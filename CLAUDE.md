# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Branch: lerobot_phone_teleop**

This branch focuses on phone-based teleoperation and data collection for the SO101 robot arm. The system receives pose estimates from smartphone sensors (IMU, gyroscope, accelerometer) via TCP and maps phone movements to robot end-effector velocities in simulation.

**Supported devices:** iPhone 13, Pixel 8a

**Core concept:** Velocity mirroring - phone velocities are directly mapped to robot end-effector velocities:
- Phone linear velocities → end-effector linear velocities in base link frame
- Phone angular velocities → end-effector angular velocities in end-effector frame
- Gripper controlled via slider value from phone UI

The initial pose when a session starts defines the "home" relative state for both phone and robot.

## Common Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_env.py

# Interactive MuJoCo viewer (Linux)
uv run tests/mj_viewer_rendering.py

# Interactive MuJoCo viewer (macOS - requires special binary)
uv run mjpython tests/mj_viewer_rendering.py

# Run teleoperation script (test mode with hardcoded velocities)
uv run python -m lerobot_phone_teleop.scripts.teleoperate --test

# Run recording script (test mode)
uv run python -m lerobot_phone_teleop.scripts.record --test
```

## Development Rules

### Testing After Every Edit

**CRITICAL:** After making any code changes, always run the main scripts to verify they execute without errors:

```bash
# Test teleoperation script
uv run python -m lerobot_phone_teleop.scripts.teleoperate --test

# Test recording script
uv run python -m lerobot_phone_teleop.scripts.record --test
```

The `--test` flag enables test mode which:
- Bypasses TCP connection requirement
- Uses hardcoded velocity values for end-effector control
- Runs a short episode (100 steps) to verify functionality
- Skips dataset upload/push operations

### Code Style (following lerobot conventions)

1. **Script organization**: Main scripts go in `src/lerobot_phone_teleop/scripts/` with entry points like `teleoperate.py`, `record.py`

2. **Configuration**: Use dataclasses with `@dataclass` decorator for configs:
   ```python
   @dataclass
   class TeleoperationConfig:
       tcp_host: str = "0.0.0.0"
       tcp_port: int = 5555
       test_mode: bool = False
       fps: int = 30
   ```

3. **Feature naming**: Follow lerobot's `observation.images.<camera_name>`, `observation.state`, `action` convention

4. **Type hints**: All functions must have type hints

5. **Docstrings**: Use Google-style docstrings for public functions

6. **Test mode pattern**: All main scripts must support `--test` flag that uses mock/hardcoded data:
   ```python
   def get_phone_data(config: TeleoperationConfig) -> PhoneState:
       if config.test_mode:
           return _get_hardcoded_test_data()
       return _receive_from_tcp()
   ```

## Architecture

### Phone Teleoperation Data Flow

```
Phone (state estimation) → TCP → Teleoperation Server → IK Solver → Robot Simulation → Data Storage
```

**Data received from phone (TCP):**
- Position (world frame)
- Velocity (world frame)
- Quaternion orientation
- Angular velocity (body frame)
- Gripper slider value (0-1)

**Note:** Phone state estimation scripts are in separate branches (not yet available).

### SO101 Robot Configuration

**Joint chain (6-DOF arm + gripper):**
| Joint | Type | Range (rad) |
|-------|------|-------------|
| shoulder_pan | hinge | -1.92 to 1.92 |
| shoulder_lift | hinge | -1.75 to 1.75 |
| elbow_flex | hinge | -1.69 to 1.69 |
| wrist_flex | hinge | -1.66 to 1.66 |
| wrist_roll | hinge | -2.74 to 2.84 |
| gripper | hinge | -0.17 to 1.75 |

**End-effector site:** `gripperframe` - used for IK target and position sensing

**Actuators:** Position-controlled STS3215 servos with force limits ±3.35 N

### Camera Configuration

Cameras defined in `src/lerobothackathonenv/models/xml/arenas/table_arena.xml`:
- `frontview` - pos=(1.6, 0, 1.45) - main front-facing view
- `birdview` - pos=(-0.2, 0, 3.0) - top-down view
- `agentview` - pos=(0.5, 0, 1.35) - agent perspective
- `sideview` - pos=(-0.056, 1.276, 1.488) - side perspective

Additional camera in main scene (`so101_tabletop_manipulation.xml`):
- `front` - pos=(0.9, 0, 1.2) - targets robot base

Rendering uses `dm_control.physics.render(camera_id=...)` with default size 256x256.

### Data Collection

Data is stored in **LeRobot dataset format**. Each episode should include:
- Camera images (from MuJoCo render)
- Joint positions (qpos)
- Joint velocities (qvel)
- End-effector pose
- Actions (target joint positions)
- Timestamps

### Core Components

**Environment (`src/lerobothackathonenv/env.py`)**
- `LeRobot(Env)`: Gymnasium environment wrapping dm_control
- `render()`: Returns numpy array from specified camera
- `sim_state` property: Returns `MujocoState` for trajectory recording

**Task System (`src/lerobothackathonenv/tasks.py`)**
- `ExtendedTask`: Base class for environment variations
- Defines `XML_PATH`, `ACTION_SPACE`, `OBSERVATION_SPACE`
- `get_observation(physics)`: Extracts robot state

**State Recording (`src/lerobothackathonenv/structs.py`)**
- `MujocoState`: Dataclass with qpos, qvel, xpos, xquat, mocap_pos, mocap_quat

### Key Implementation Requirements

1. **TCP Server**: Receive pose data from phone
2. **Velocity Mapping**: Transform phone velocities to end-effector frame
3. **Inverse Kinematics**: Compute joint velocities/positions from EE velocity commands
4. **Data Recording**: Store episodes in LeRobot format with camera images
5. **Session Management**: Handle connection establishment, home pose calibration

### Physics Access Patterns

```python
# Get end-effector position
gripper_site_id = mujoco.mj_name2id(physics.model._model, mujoco.mjtObj.mjOBJ_SITE.value, "gripperframe")
ee_pos = physics.data.site_xpos[gripper_site_id]

# Joint positions/velocities
qpos = physics.data.qpos  # shape (27,) - includes free joints of objects
qvel = physics.data.qvel  # shape (24,)

# Robot joint indices: 0-5 for arm, 6 for gripper (after base)
```
