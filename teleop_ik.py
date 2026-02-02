"""
Jacobian-based IK solver for the SO-101 arm.

Computes joint-position targets that move the gripperframe site toward
a desired Cartesian end-effector position using damped least-squares.
Returns a 6-element action array (5 arm joint targets in radians + 1
gripper target in radians) that can be passed directly to env.step().

The actuators are position-controlled; MuJoCo clamps ctrl values to
each actuator's ctrlrange automatically.
"""

import numpy as np
import mujoco

# Arm joint names in kinematic-chain order (excludes gripper).
ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# Damped least-squares damping factor.
DAMPING = 0.05

# Maximum joint-position step per IK call (radians).
MAX_JOINT_STEP = 0.15


def solve_ik(
    physics,
    target_pos: np.ndarray,
    gripper_action: float = 0.0,
) -> np.ndarray:
    """Return a 6-element action array for ``env.step()``.

    Parameters
    ----------
    physics : dm_control Physics instance
        Access to model/data for the current simulation state.
    target_pos : (3,) ndarray
        Desired Cartesian position for the gripperframe site.
    gripper_action : float
        Gripper command in radians.  ~1.7 = fully open, ~-0.17 = closed.

    Returns
    -------
    action : (6,) ndarray — joint-position targets in radians.
    """
    model = physics.model._model
    data_wrapper = physics.data
    # Raw MjData needed for mj_jacSite; the wrapper is used for array access.
    data_raw = data_wrapper._data

    # ---- resolve indices ------------------------------------------------
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE.value, "gripperframe"
    )
    arm_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, n)
        for n in ARM_JOINT_NAMES
    ]
    arm_dof_ids = [model.jnt_dofadr[jid] for jid in arm_joint_ids]
    arm_qpos_ids = [model.jnt_qposadr[jid] for jid in arm_joint_ids]
    n_arm = len(arm_dof_ids)

    # ---- current EE position -------------------------------------------
    ee_pos = data_wrapper.site_xpos[site_id].copy()  # (3,)

    # ---- positional error ----------------------------------------------
    err = target_pos - ee_pos  # (3,)

    # ---- Jacobian at gripperframe (full 3×nv) --------------------------
    jacp_full = np.zeros((3, model.nv), dtype=np.float64)
    jacr_full = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data_raw, jacp_full, jacr_full, site_id)

    # Extract columns for the arm DOFs only.
    J = jacp_full[:, arm_dof_ids]  # (3, n_arm)

    # ---- damped least-squares ------------------------------------------
    JtJ = J.T @ J + (DAMPING ** 2) * np.eye(n_arm)
    dq = np.linalg.solve(JtJ, J.T @ err)  # (n_arm,)

    # Clamp step magnitude.
    max_abs = np.abs(dq).max()
    if max_abs > MAX_JOINT_STEP:
        dq *= MAX_JOINT_STEP / max_abs

    # ---- new joint targets (position-controlled actuators) --------------
    q_current = np.array([data_wrapper.qpos[i] for i in arm_qpos_ids])
    q_target = q_current + dq

    # Build full 6-DOF action: 5 arm joints + gripper.
    action = np.zeros(6, dtype=np.float64)
    action[:5] = q_target
    action[5] = gripper_action

    return action
