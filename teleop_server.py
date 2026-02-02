"""
iPhone teleoperation server for the SO-101 arm.

Runs an async WebSocket server on 0.0.0.0:8765.  Receives phone
orientation (quaternion) + grasp state, converts tilt directly into
end-effector velocity, solves velocity IK via the Jacobian, and steps
the MuJoCo simulation with a live viewer.

Usage
-----
    uv run python teleop_server.py
    uv run python teleop_server.py --no-viewer
"""

import argparse
import asyncio
import json
import time

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

import lerobothackathonenv as _  # noqa: F401 — triggers env registration
from lerobothackathonenv.env import LeRobot
from teleop_ik import solve_velocity_ik

try:
    import websockets
    from websockets.asyncio.server import serve
except ImportError:
    raise SystemExit(
        "websockets is required.  Install with:  uv add 'websockets>=12.0'"
    )

# ── Configuration ────────────────────────────────────────────────────────

WS_HOST = "0.0.0.0"
WS_PORT = 8765
CONTROL_HZ = 30
DT = 1.0 / CONTROL_HZ

# Tilt-to-velocity mapping
VELOCITY_SCALE = 0.15          # m/s per radian of tilt
DEADZONE_DEG = 5.0             # degrees — ignore small tilts
DEADZONE_RAD = np.radians(DEADZONE_DEG)

# Workspace bounds (x forward, y left, z up in MuJoCo world frame).
WS_LO = np.array([-0.10, -0.30, 0.62])
WS_HI = np.array([0.40,   0.30, 1.00])

# Margin for workspace edge velocity damping (metres).
WS_EDGE_MARGIN = 0.02

# Gripper actuator targets (radians, from XML ctrlrange).
GRIPPER_OPEN = 1.5
GRIPPER_CLOSED = -0.15


# ── Teleop state ─────────────────────────────────────────────────────────

class TeleopState:
    """Tracks phone orientation and computes EE velocity."""

    def __init__(self):
        self.ee_velocity = np.zeros(3)
        self.ref_rot: Rotation | None = None
        self.grasp = False
        self.connected = False
        self.last_update = 0.0
        self.home_requested = False

    def calibrate(self, quat_wxyz: np.ndarray):
        """Store current phone orientation as the neutral reference."""
        # scipy uses (x, y, z, w) ordering
        self.ref_rot = Rotation.from_quat(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        )

    def update(self, quat_wxyz: np.ndarray, grasp: bool):
        """Process a new phone IMU reading."""
        self.grasp = grasp
        self.connected = True
        self.last_update = time.monotonic()

        q_cur = Rotation.from_quat(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        )

        if self.ref_rot is None:
            self.calibrate(quat_wxyz)
            return

        # Relative rotation from reference pose.
        q_rel = self.ref_rot.inv() * q_cur
        # Intrinsic Euler: pitch (X), roll (Y), yaw (Z)
        pitch, roll, yaw = q_rel.as_euler("XYZ")

        # Apply deadzone.
        pitch = _deadzone(pitch, DEADZONE_RAD)
        roll = _deadzone(roll, DEADZONE_RAD)
        yaw = _deadzone(yaw, DEADZONE_RAD)

        # Map tilt → velocity.  Conventions (phone held in landscape):
        #   pitch (tilt forward/back) → +X  (forward on table)
        #   roll  (tilt left/right)   → +Y  (left on table)
        #   yaw   (twist)             → +Z  (up/down)
        self.ee_velocity = np.array([
            pitch * VELOCITY_SCALE,
            roll * VELOCITY_SCALE,
            yaw * VELOCITY_SCALE,
        ])

    def request_home(self):
        """Flag that a home reset was requested."""
        self.home_requested = True


def _deadzone(value: float, zone: float) -> float:
    if abs(value) < zone:
        return 0.0
    return value - np.sign(value) * zone


def _clamp_velocity_at_edges(ee_pos: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Zero out velocity components that would push EE beyond workspace bounds."""
    vel = velocity.copy()
    for i in range(3):
        if ee_pos[i] <= WS_LO[i] + WS_EDGE_MARGIN and vel[i] < 0:
            vel[i] = 0.0
        elif ee_pos[i] >= WS_HI[i] - WS_EDGE_MARGIN and vel[i] > 0:
            vel[i] = 0.0
    return vel


# ── WebSocket handler ────────────────────────────────────────────────────

async def phone_handler(websocket, state: TeleopState):
    """Handle messages from one phone client."""
    print("[ws] Phone connected")
    state.connected = True
    try:
        async for raw in websocket:
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "imu":
                quat = np.array(msg["quaternion"], dtype=np.float64)
                grasp = bool(msg.get("grasp", False))
                state.update(quat, grasp)

            elif msg_type == "recalibrate":
                # Next IMU message will set a new reference.
                state.ref_rot = None
                print("[ws] Recalibrate requested")

            elif msg_type == "home":
                state.request_home()
                print("[ws] Home requested")

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        state.connected = False
        print("[ws] Phone disconnected")


# ── Main loop ────────────────────────────────────────────────────────────

async def main(use_viewer: bool = True):
    # Create environment.
    env: LeRobot = gym.make("LeRobotGoalConditioned-v0")
    obs, _ = env.reset()
    if use_viewer:
        env.unwrapped.render_to_window()

    physics = env.unwrapped.dm_control_env._physics

    # Resolve EE site id once.
    import mujoco as mj
    site_id = mj.mj_name2id(
        physics.model._model,
        mj.mjtObj.mjOBJ_SITE.value,
        "gripperframe",
    )

    state = TeleopState()

    initial_ee = physics.data.site_xpos[site_id].copy()
    print(f"[server] Initial EE position: {initial_ee}")
    print(f"[server] Starting WebSocket server on ws://{WS_HOST}:{WS_PORT}")

    # Start WebSocket server.
    connected_ws = set()

    async def handler(websocket):
        connected_ws.add(websocket)
        try:
            await phone_handler(websocket, state)
        finally:
            connected_ws.discard(websocket)

    server = await serve(handler, WS_HOST, WS_PORT)

    print("[server] Waiting for phone connection …")

    # Control loop — runs whether or not a phone is connected.
    try:
        while True:
            loop_start = time.monotonic()

            # Handle home reset.
            if state.home_requested:
                state.home_requested = False
                obs, _ = env.reset()
                if use_viewer:
                    env.unwrapped.render_to_window()
                # Recalibrate phone reference on next IMU message.
                state.ref_rot = None
                state.ee_velocity = np.zeros(3)
                print("[server] Environment reset (home)")
                # Skip this control step.
                elapsed = time.monotonic() - loop_start
                sleep_time = DT - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                continue

            # Get current EE position for workspace edge clamping.
            ee_pos = physics.data.site_xpos[site_id].copy()

            # Clamp velocity at workspace edges.
            clamped_vel = _clamp_velocity_at_edges(ee_pos, state.ee_velocity)

            # Compute velocity IK.
            gripper_cmd = GRIPPER_CLOSED if state.grasp else GRIPPER_OPEN
            action = solve_velocity_ik(physics, clamped_vel, gripper_cmd)

            # Step simulation.
            obs, reward, terminated, truncated, info = env.step(action)
            if use_viewer:
                env.unwrapped.render_to_window()

            # Send feedback to all connected phones.
            ee_now = physics.data.site_xpos[site_id].copy()
            feedback = json.dumps({
                "type": "state",
                "ee_pos": ee_now.tolist(),
                "connected": state.connected,
            })
            for ws in list(connected_ws):
                try:
                    await ws.send(feedback)
                except Exception:
                    pass

            # Maintain target rate.
            elapsed = time.monotonic() - loop_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[server] Shutting down …")
    finally:
        server.close()
        await server.wait_closed()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SO-101 iPhone teleop server")
    parser.add_argument(
        "--no-viewer", action="store_true",
        help="Run headless without the MuJoCo viewer window",
    )
    args = parser.parse_args()
    asyncio.run(main(use_viewer=not args.no_viewer))
