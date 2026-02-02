#!/usr/bin/env python3
"""Phone teleoperation script for SO101 robot in MuJoCo simulation.

Usage:
    # Test mode (hardcoded velocities, no TCP)
    uv run python -m lerobot_phone_teleop.scripts.teleoperate --test

    # With MuJoCo viewer
    uv run python -m lerobot_phone_teleop.scripts.teleoperate --test --show-viewer

    # Production mode (requires phone connection)
    uv run python -m lerobot_phone_teleop.scripts.teleoperate --host 0.0.0.0 --port 5555
"""

import argparse
import time

import gymnasium as gym

# Import the environment to register it
import lerobothackathonenv  # noqa: F401

from lerobot_phone_teleop.config import RobotConfig, TeleoperationConfig
from lerobot_phone_teleop.controller import (
    TeleoperationController,
    generate_test_phone_state,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phone teleoperation for SO101 robot"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with hardcoded velocities (no TCP)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="TCP host for phone connection",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="TCP port for phone connection",
    )
    parser.add_argument(
        "--test-steps",
        type=int,
        default=300,
        help="Number of steps to run in test mode",
    )
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Show MuJoCo viewer window",
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=30,
        help="Control frequency in Hz",
    )
    parser.add_argument(
        "--linear-scale",
        type=float,
        default=1.0,
        help="Linear velocity scale factor",
    )
    parser.add_argument(
        "--angular-scale",
        type=float,
        default=1.0,
        help="Angular velocity scale factor",
    )

    return parser.parse_args()


def run_teleoperation(args: argparse.Namespace) -> None:
    """Run the teleoperation loop.

    Args:
        args: Parsed command line arguments.
    """
    # Create configs
    robot_config = RobotConfig()
    teleop_config = TeleoperationConfig(
        tcp_host=args.host,
        tcp_port=args.port,
        control_freq=args.control_freq,
        test_mode=args.test,
        test_episode_length=args.test_steps,
        linear_velocity_scale=args.linear_scale,
        angular_velocity_scale=args.angular_scale,
        show_viewer=args.show_viewer,
    )

    # Check TCP mode before creating environment
    if not teleop_config.test_mode:
        raise NotImplementedError(
            "TCP receiver not yet implemented. Use --test flag for test mode."
        )

    # Create environment
    env = gym.make("LeRobot-v0")
    env.reset()

    # Get physics from environment
    physics = env.unwrapped.dm_control_env._physics

    # Create controller
    controller = TeleoperationController(
        physics=physics,
        robot_config=robot_config,
        teleop_config=teleop_config,
    )

    print(f"Starting teleoperation...")
    print(f"  Test mode: {teleop_config.test_mode}")
    print(f"  Control freq: {teleop_config.control_freq} Hz")
    print(f"  Show viewer: {teleop_config.show_viewer}")

    if teleop_config.test_mode:
        print(f"  Test steps: {teleop_config.test_episode_length}")
    else:
        print(f"  TCP: {teleop_config.tcp_host}:{teleop_config.tcp_port}")

    # Control loop timing
    dt = 1.0 / teleop_config.control_freq

    step = 0
    max_steps = teleop_config.test_episode_length if teleop_config.test_mode else float("inf")

    try:
        while step < max_steps:
            loop_start = time.time()

            # Get phone state (test mode uses hardcoded velocities)
            phone_state = generate_test_phone_state(
                step=step,
                total_steps=teleop_config.test_episode_length,
            )

            # Compute action
            action = controller.compute_action(phone_state)

            # Step environment
            env.step(action)

            # Render
            if teleop_config.show_viewer:
                env.unwrapped.render_to_window()

            # Print status periodically
            if step % 30 == 0:
                ee_pos, _ = controller.ik_solver.get_ee_pose()
                print(
                    f"Step {step:4d} | "
                    f"EE pos: [{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}] | "
                    f"Gripper: {phone_state.gripper_value:.2f}"
                )

            step += 1

            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nTeleoperation interrupted by user")

    finally:
        env.close()
        print(f"Teleoperation complete. Total steps: {step}")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    run_teleoperation(args)


if __name__ == "__main__":
    main()
