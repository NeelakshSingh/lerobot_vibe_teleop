# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobotHackathonEnv is a minimal, extendable gym environment for robotic manipulation built on MuJoCo/dm_control. It provides gymnasium-compatible environments for training RL agents on tabletop manipulation tasks with the SO-101 robot arm.

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

# Throughput benchmarking
uv run python -m tests.vec_env_throughput --backend gym --sweep-num-envs 2,4,8,16,32
uv run python -m tests.vec_env_throughput --backend puffer --sweep-num-envs 2,4,8,16,32 --puffer-num-workers 8
```

## Architecture

### Core Components

**Environment (`src/lerobothackathonenv/env.py`)**
- `LeRobot(Env)`: Main gymnasium environment class wrapping dm_control physics
- Key methods: `step()`, `reset()`, `render()`, `render_to_window()`
- `sim_state` property: Returns `MujocoState` for trajectory recording/dataset creation

**Task System (`src/lerobothackathonenv/tasks.py`)**
- `ExtendedTask`: Abstract base class defining environment variations
- Each task specifies: `XML_PATH`, `ACTION_SPACE`, `OBSERVATION_SPACE`
- Required methods: `get_reward()`, `get_observation()`, `get_sim_metadata()`
- Optional: `get_success()` for task completion detection

**Concrete Tasks:**
- `ExampleTask`: Base task with common observation extraction (qpos, qvel, actuator_force, gripper_pos)
- `ExampleReachTask`: Gripper reaches target position (Gaussian reward)
- `GoalConditionedObjectPlaceTask`: Pick-and-place with 3 manipulatable objects (milk_0, bread_1, cereal_2), goal-conditioned learning, reward shaping, and randomized spawning

**Registered Environments:**
- `LeRobot-v0`: Default reach task
- `LeRobotGoalConditioned-v0`: Goal-conditioned pick-and-place

### Creating New Tasks

Subclass `ExtendedTask` and define:
1. `XML_PATH`: Path to MuJoCo scene XML
2. `ACTION_SPACE` / `OBSERVATION_SPACE`: Gymnasium spaces
3. `get_reward(physics)`: Reward function using dm_control physics
4. `get_observation(physics)`: Observation function
5. `get_sim_metadata()`: Episode metadata for trajectory recording

Register with `gymnasium.register()` in `__init__.py`.

### Physics Assets

`src/lerobothackathonenv/models/` contains:
- `xml/`: MuJoCo scene descriptions and component includes
- `meshes/`: 3D geometry (STL) for robots (fr3, leap, SO-101)
- `textures/`: Visual textures

### Vectorization

Supports both Gymnasium `AsyncVectorEnv` (~28k steps/s) and PufferLib multiprocessing (~60k steps/s at 16 envs, 8 workers).

## Key Patterns

- Physics accessed via `physics.data` (positions, velocities) and `physics.model` (model parameters)
- Site/body IDs retrieved with `mujoco.mj_name2id()`
- Observations clipped to defined ranges before returning
- `initialize_episode()` in tasks handles per-episode randomization (goal sampling, object placement)
