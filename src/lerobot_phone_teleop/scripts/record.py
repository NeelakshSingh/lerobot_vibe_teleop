#!/usr/bin/env python3
"""Recording script for phone teleoperation data collection.

Records episodes in LeRobot dataset format with camera images, joint states,
and actions.

Usage:
    # Test mode (hardcoded velocities, no TCP, local storage only)
    uv run python -m lerobot_phone_teleop.scripts.record --test

    # Test mode with viewer
    uv run python -m lerobot_phone_teleop.scripts.record --test --show-viewer

    # Record multiple episodes
    uv run python -m lerobot_phone_teleop.scripts.record --test --num-episodes 5
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import gymnasium as gym

# Import the environment to register it
import lerobothackathonenv  # noqa: F401

from lerobot_phone_teleop.config import (
    RobotConfig,
    TeleoperationConfig,
    RecordingConfig,
)
from lerobot_phone_teleop.controller import (
    TeleoperationController,
    generate_test_phone_state,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Record phone teleoperation episodes"
    )

    # Mode
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with hardcoded velocities (no TCP)",
    )

    # Recording settings
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=300,
        help="Steps per episode",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="phone_teleop_dataset",
        help="Name for the dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default="Phone teleoperation demonstration",
        help="Task description for the dataset",
    )

    # Control settings
    parser.add_argument(
        "--control-freq",
        type=int,
        default=30,
        help="Control frequency in Hz",
    )
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Show MuJoCo viewer window",
    )

    # Camera settings
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=["frontview", "agentview"],
        help="Camera names to record",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Image size (width height)",
    )

    return parser.parse_args()


class EpisodeRecorder:
    """Records a single episode of teleoperation data."""

    def __init__(
        self,
        env: gym.Env,
        cameras: List[str],
        image_size: tuple,
        robot_config: RobotConfig,
    ):
        """Initialize episode recorder.

        Args:
            env: Gymnasium environment.
            cameras: List of camera names to record.
            image_size: Image size as (width, height).
            robot_config: Robot configuration.
        """
        self.env = env
        self.cameras = cameras
        self.image_size = image_size
        self.robot_config = robot_config

        # Get camera IDs
        self.camera_ids = {}
        physics = env.unwrapped.dm_control_env._physics
        for cam_name in cameras:
            try:
                import mujoco
                cam_id = mujoco.mj_name2id(
                    physics.model._model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    cam_name
                )
                if cam_id >= 0:
                    self.camera_ids[cam_name] = cam_id
            except Exception:
                pass

        # Episode data storage
        self.frames: List[Dict[str, Any]] = []
        self.episode_start_time = 0.0

    def reset(self) -> None:
        """Reset recorder for new episode."""
        self.frames = []
        self.episode_start_time = time.time()

    def record_frame(
        self,
        action: np.ndarray,
        phone_state: Any,
    ) -> None:
        """Record a single frame of data.

        Args:
            action: Action taken (joint positions).
            phone_state: Current phone state.
        """
        physics = self.env.unwrapped.dm_control_env._physics
        timestamp = time.time() - self.episode_start_time

        # Get robot state
        qpos = physics.data.qpos.copy()
        qvel = physics.data.qvel.copy()

        # Get end-effector pose
        import mujoco
        ee_site_id = mujoco.mj_name2id(
            physics.model._model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.robot_config.ee_site_name
        )
        ee_pos = physics.data.site_xpos[ee_site_id].copy()
        ee_mat = physics.data.site_xmat[ee_site_id].reshape(3, 3).copy()

        # Capture camera images
        images = {}
        for cam_name, cam_id in self.camera_ids.items():
            img = physics.render(
                width=self.image_size[0],
                height=self.image_size[1],
                camera_id=cam_id,
            )
            images[cam_name] = img

        # Build frame data (following LeRobot naming convention)
        frame = {
            "timestamp": timestamp,
            "observation.state": qpos[:7].astype(np.float32),  # 6 arm + 1 gripper
            "observation.velocity": qvel[:6].astype(np.float32),  # 6 arm joints
            "observation.ee_pos": ee_pos.astype(np.float32),
            "action": action.astype(np.float32),
        }

        # Add camera images
        for cam_name, img in images.items():
            frame[f"observation.images.{cam_name}"] = img

        # Add phone state for reference
        frame["phone.linear_velocity"] = np.array(
            phone_state.linear_velocity, dtype=np.float32
        )
        frame["phone.angular_velocity"] = np.array(
            phone_state.angular_velocity, dtype=np.float32
        )
        frame["phone.gripper_value"] = np.float32(phone_state.gripper_value)

        self.frames.append(frame)

    def get_episode_data(self) -> Dict[str, Any]:
        """Get recorded episode data.

        Returns:
            Dictionary with episode data and metadata.
        """
        if not self.frames:
            return {}

        # Stack arrays for each key
        episode_data = {}
        for key in self.frames[0].keys():
            values = [f[key] for f in self.frames]
            if isinstance(values[0], np.ndarray):
                episode_data[key] = np.stack(values)
            else:
                episode_data[key] = np.array(values)

        return episode_data


def save_dataset_metadata(
    output_path: Path,
    dataset_name: str,
    task_description: str,
    num_episodes: int,
    fps: int,
    cameras: List[str],
    image_size: tuple,
    robot_config: RobotConfig,
) -> None:
    """Save dataset metadata files.

    Args:
        output_path: Path to dataset directory.
        dataset_name: Name of the dataset.
        task_description: Task description.
        num_episodes: Number of episodes.
        fps: Frames per second.
        cameras: List of camera names.
        image_size: Image dimensions.
        robot_config: Robot configuration.
    """
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # info.json - dataset schema and settings
    info = {
        "codebase_version": "3.0",
        "dataset_name": dataset_name,
        "robot_type": "so101",
        "fps": fps,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [7],
                "description": "Joint positions (6 arm + 1 gripper)",
            },
            "observation.velocity": {
                "dtype": "float32",
                "shape": [6],
                "description": "Joint velocities (6 arm joints)",
            },
            "observation.ee_pos": {
                "dtype": "float32",
                "shape": [3],
                "description": "End-effector position in base frame",
            },
            "action": {
                "dtype": "float32",
                "shape": [6],
                "description": "Target joint positions (5 arm + 1 gripper)",
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [],
                "description": "Time since episode start (seconds)",
            },
        },
        "cameras": cameras,
        "image_size": list(image_size),
        "total_episodes": num_episodes,
        "robot_config": asdict(robot_config),
    }

    # Add camera features
    for cam_name in cameras:
        info["features"][f"observation.images.{cam_name}"] = {
            "dtype": "uint8",
            "shape": [image_size[1], image_size[0], 3],  # H, W, C
            "description": f"RGB image from {cam_name} camera",
        }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # tasks.jsonl - task descriptions
    with open(meta_dir / "tasks.jsonl", "w") as f:
        task_entry = {"task_index": 0, "task": task_description}
        f.write(json.dumps(task_entry) + "\n")


def save_episode(
    output_path: Path,
    episode_idx: int,
    episode_data: Dict[str, Any],
) -> None:
    """Save a single episode to disk.

    Args:
        output_path: Path to dataset directory.
        episode_idx: Episode index.
        episode_data: Episode data dictionary.
    """
    data_dir = output_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save episode data as numpy archive
    episode_file = data_dir / f"episode_{episode_idx:04d}.npz"

    # Separate images from other data
    images_data = {}
    scalar_data = {}

    for key, value in episode_data.items():
        if "images" in key:
            images_data[key] = value
        else:
            scalar_data[key] = value

    # Save scalar data
    np.savez_compressed(episode_file, **scalar_data)

    # Save images separately (could be converted to video later)
    if images_data:
        videos_dir = output_path / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        for cam_key, frames in images_data.items():
            cam_name = cam_key.split(".")[-1]
            cam_dir = videos_dir / cam_name
            cam_dir.mkdir(parents=True, exist_ok=True)

            # Save as numpy array (can be converted to MP4 later)
            np.save(
                cam_dir / f"episode_{episode_idx:04d}.npy",
                frames
            )


def run_recording(args: argparse.Namespace) -> None:
    """Run the recording loop.

    Args:
        args: Parsed command line arguments.
    """
    # Check test mode
    if not args.test:
        raise NotImplementedError(
            "TCP receiver not yet implemented. Use --test flag for test mode."
        )

    # Create configs
    robot_config = RobotConfig()
    teleop_config = TeleoperationConfig(
        control_freq=args.control_freq,
        test_mode=args.test,
        test_episode_length=args.episode_length,
        show_viewer=args.show_viewer,
    )
    recording_config = RecordingConfig(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        fps=args.control_freq,
        cameras=args.cameras,
        image_size=tuple(args.image_size),
        test_mode=args.test,
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"{args.dataset_name}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Recording dataset to: {output_path}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Steps per episode: {args.episode_length}")
    print(f"  Cameras: {args.cameras}")
    print(f"  Control freq: {args.control_freq} Hz")

    # Create environment
    env = gym.make("LeRobot-v0")

    # Create recorder
    recorder = EpisodeRecorder(
        env=env,
        cameras=args.cameras,
        image_size=tuple(args.image_size),
        robot_config=robot_config,
    )

    # Record episodes
    for episode_idx in range(args.num_episodes):
        print(f"\nRecording episode {episode_idx + 1}/{args.num_episodes}...")

        # Reset environment and recorder
        env.reset()
        recorder.reset()

        # Get physics and create controller
        physics = env.unwrapped.dm_control_env._physics
        controller = TeleoperationController(
            physics=physics,
            robot_config=robot_config,
            teleop_config=teleop_config,
        )

        # Control loop timing
        dt = 1.0 / teleop_config.control_freq

        for step in range(args.episode_length):
            loop_start = time.time()

            # Get phone state (test mode)
            phone_state = generate_test_phone_state(
                step=step,
                total_steps=args.episode_length,
            )

            # Compute action
            action = controller.compute_action(phone_state)

            # Record frame before stepping
            recorder.record_frame(action=action, phone_state=phone_state)

            # Step environment
            env.step(action)

            # Render
            if teleop_config.show_viewer:
                env.unwrapped.render_to_window()

            # Print progress
            if step % 50 == 0:
                print(f"  Step {step}/{args.episode_length}")

            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Save episode
        episode_data = recorder.get_episode_data()
        save_episode(output_path, episode_idx, episode_data)
        print(f"  Saved episode {episode_idx + 1} ({len(recorder.frames)} frames)")

    # Save metadata
    save_dataset_metadata(
        output_path=output_path,
        dataset_name=args.dataset_name,
        task_description=args.task_description,
        num_episodes=args.num_episodes,
        fps=args.control_freq,
        cameras=args.cameras,
        image_size=tuple(args.image_size),
        robot_config=robot_config,
    )

    env.close()
    print(f"\nRecording complete!")
    print(f"Dataset saved to: {output_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    run_recording(args)


if __name__ == "__main__":
    main()
