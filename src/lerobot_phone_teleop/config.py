"""Configuration dataclasses for phone teleoperation."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class RobotConfig:
    """Configuration for the SO101 robot."""

    # Joint names in order (excluding gripper for IK)
    arm_joint_names: List[str] = field(default_factory=lambda: [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ])
    gripper_joint_name: str = "gripper"

    # End-effector site name for IK
    ee_site_name: str = "gripperframe"

    # Joint limits (rad) - from SO101 URDF
    joint_limits_lower: List[float] = field(default_factory=lambda: [
        -1.92, -1.75, -1.69, -1.66, -2.74
    ])
    joint_limits_upper: List[float] = field(default_factory=lambda: [
        1.92, 1.75, 1.69, 1.66, 2.84
    ])

    # Gripper range
    gripper_range: tuple = (-0.17, 1.75)

    # Number of arm joints (excluding gripper)
    n_arm_joints: int = 5


@dataclass
class TeleoperationConfig:
    """Configuration for teleoperation session."""

    # TCP server settings
    tcp_host: str = "0.0.0.0"
    tcp_port: int = 5555

    # Control settings
    control_freq: int = 30  # Hz

    # IK settings
    damping: float = 0.05  # Damped least squares coefficient
    max_joint_velocity: float = 2.0  # rad/s

    # Velocity scaling (phone velocity to robot velocity)
    linear_velocity_scale: float = 1.0
    angular_velocity_scale: float = 1.0

    # Test mode settings
    test_mode: bool = False
    test_episode_length: int = 100  # steps

    # Rendering
    render_camera: str = "frontview"
    render_size: tuple = (640, 480)
    show_viewer: bool = True


@dataclass
class RecordingConfig:
    """Configuration for data recording."""

    # Dataset settings
    dataset_name: str = "phone_teleop_dataset"
    output_dir: str = "data"

    # Recording settings
    fps: int = 30
    cameras: List[str] = field(default_factory=lambda: ["frontview", "agentview"])
    image_size: tuple = (256, 256)

    # LeRobot format settings
    repo_id: str = ""  # HuggingFace repo ID (empty for local only)
    push_to_hub: bool = False

    # Test mode
    test_mode: bool = False


@dataclass
class PhoneState:
    """State received from phone sensors."""

    # Position in world frame (meters)
    position: tuple = (0.0, 0.0, 0.0)

    # Linear velocity in world frame (m/s)
    linear_velocity: tuple = (0.0, 0.0, 0.0)

    # Orientation as quaternion (w, x, y, z)
    quaternion: tuple = (1.0, 0.0, 0.0, 0.0)

    # Angular velocity in body frame (rad/s)
    angular_velocity: tuple = (0.0, 0.0, 0.0)

    # Gripper value from slider (0.0 = closed, 1.0 = open)
    gripper_value: float = 0.0

    # Timestamp
    timestamp: float = 0.0
