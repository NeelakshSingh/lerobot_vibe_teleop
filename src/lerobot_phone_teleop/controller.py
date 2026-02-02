"""Teleoperation controller for phone-based robot control."""

import numpy as np
from numpy.typing import NDArray
from typing import Optional
import time

import mujoco as mj
from dm_control import mujoco as dm_mujoco

from .config import RobotConfig, TeleoperationConfig, PhoneState
from .ik_solver import IKSolver


class TeleoperationController:
    """Controller for phone-based teleoperation of SO101 robot.

    Handles velocity mapping from phone to robot end-effector and
    computes joint commands via inverse kinematics.
    """

    def __init__(
        self,
        physics: dm_mujoco.Physics,
        robot_config: RobotConfig = None,
        teleop_config: TeleoperationConfig = None,
    ):
        """Initialize teleoperation controller.

        Args:
            physics: dm_control physics instance.
            robot_config: Robot configuration.
            teleop_config: Teleoperation configuration.
        """
        self.physics = physics
        self.robot_config = robot_config or RobotConfig()
        self.teleop_config = teleop_config or TeleoperationConfig()

        # Initialize IK solver
        self.ik_solver = IKSolver(
            model=physics.model._model,
            data=physics.data,
            config=self.robot_config,
            damping=self.teleop_config.damping,
        )

        # Home state (set on first phone data received)
        self.home_phone_position: Optional[NDArray] = None
        self.home_phone_quaternion: Optional[NDArray] = None
        self.home_ee_position: Optional[NDArray] = None
        self.home_ee_quaternion: Optional[NDArray] = None

        # Gripper joint info
        self.gripper_joint_id = mj.mj_name2id(
            physics.model._model,
            mj.mjtObj.mjOBJ_JOINT,
            self.robot_config.gripper_joint_name
        )
        self.gripper_qpos_addr = physics.model._model.jnt_qposadr[self.gripper_joint_id]

        # Control timestep
        self.dt = 1.0 / self.teleop_config.control_freq

        # State
        self.is_calibrated = False
        self.step_count = 0

    def calibrate_home(self, phone_state: PhoneState) -> None:
        """Set home position from current phone and robot state.

        Args:
            phone_state: Current phone state to use as home.
        """
        self.home_phone_position = np.array(phone_state.position)
        self.home_phone_quaternion = np.array(phone_state.quaternion)

        ee_pos, ee_quat = self.ik_solver.get_ee_pose()
        self.home_ee_position = ee_pos
        self.home_ee_quaternion = ee_quat

        self.is_calibrated = True

    def compute_action(self, phone_state: PhoneState) -> NDArray:
        """Compute robot action from phone state.

        Args:
            phone_state: Current phone state.

        Returns:
            Action array [6] with 5 arm joint positions + 1 gripper position.
        """
        if not self.is_calibrated:
            self.calibrate_home(phone_state)

        # Get velocity commands from phone
        linear_vel = np.array(phone_state.linear_velocity)
        angular_vel = np.array(phone_state.angular_velocity)

        # Apply velocity scaling
        linear_vel *= self.teleop_config.linear_velocity_scale
        angular_vel *= self.teleop_config.angular_velocity_scale

        # Compute joint velocities via IK
        joint_velocities = self.ik_solver.compute_joint_velocities(
            linear_velocity=linear_vel,
            angular_velocity=angular_vel,
            max_velocity=self.teleop_config.max_joint_velocity,
        )

        # Integrate to get target joint positions
        arm_positions = self.ik_solver.integrate_joint_positions(
            joint_velocities=joint_velocities,
            dt=self.dt,
        )

        # Map gripper slider to joint position (clamp to valid range)
        gripper_range = self.robot_config.gripper_range
        gripper_value = np.clip(phone_state.gripper_value, 0.0, 1.0)
        gripper_pos = (
            gripper_range[0] +
            gripper_value * (gripper_range[1] - gripper_range[0])
        )

        # Combine arm and gripper
        action = np.concatenate([arm_positions, [gripper_pos]])

        self.step_count += 1

        return action

    def reset(self) -> None:
        """Reset controller state."""
        self.home_phone_position = None
        self.home_phone_quaternion = None
        self.home_ee_position = None
        self.home_ee_quaternion = None
        self.is_calibrated = False
        self.step_count = 0


def generate_test_phone_state(step: int, total_steps: int = 100) -> PhoneState:
    """Generate test phone state with simple motion pattern.

    Creates a circular motion in x-y plane with sinusoidal z motion
    for testing without actual phone connection.

    Args:
        step: Current step number.
        total_steps: Total steps in test episode.

    Returns:
        PhoneState with hardcoded test velocities.
    """
    t = step / total_steps * 2 * np.pi  # One full cycle

    # Circular motion in x-y plane
    radius = 0.02  # m/s velocity magnitude
    freq = 0.5  # Hz

    linear_velocity = (
        radius * np.cos(t * freq * 2 * np.pi),
        radius * np.sin(t * freq * 2 * np.pi),
        0.01 * np.sin(t * 2),  # Small z oscillation
    )

    # Gentle rotation around z-axis
    angular_velocity = (
        0.0,
        0.0,
        0.1 * np.sin(t),
    )

    # Gripper opening and closing
    gripper_value = 0.5 + 0.5 * np.sin(t * 2)

    return PhoneState(
        position=(0.0, 0.0, 0.0),  # Position not used for velocity control
        linear_velocity=linear_velocity,
        quaternion=(1.0, 0.0, 0.0, 0.0),
        angular_velocity=angular_velocity,
        gripper_value=gripper_value,
        timestamp=time.time(),
    )
