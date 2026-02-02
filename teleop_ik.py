"""
Jacobian-based velocity IK solver for the SO-101 arm.

Computes joint velocities that move the gripperframe site at a desired
Cartesian end-effector velocity using damped least-squares on the Jacobian.
Returns a 6-element action array (5 arm joint velocities in rad/s + 1
gripper position target in radians) that can be passed directly to env.step().

The arm actuators are velocity-controlled; the gripper is position-controlled.
MuJoCo clamps ctrl values to each actuator's ctrlrange automatically.
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
DAMPING = 0.01

# Maximum joint velocity (rad/s).
MAX_JOINT_VEL = 2.5


def solve_velocity_ik(
    physics,
    ee_velocity: np.ndarray,
    gripper_action: float = 0.0,
) -> np.ndarray:
    """Return a 6-element action array for ``env.step()``.

    Parameters
    ----------
    physics : dm_control Physics instance
        Access to model/data for the current simulation state.
    ee_velocity : (3,) ndarray
        Desired Cartesian velocity for the gripperframe site (m/s).
    gripper_action : float
        Gripper position command in radians.  ~1.7 = fully open, ~-0.17 = closed.

    Returns
    -------
    action : (6,) ndarray — [joint_vel_1..5, gripper_position].
    """
    model = physics.model._model
    data_raw = physics.data._data

    # ---- resolve indices ------------------------------------------------
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE.value, "gripperframe"
    )
    arm_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, n)
        for n in ARM_JOINT_NAMES
    ]
    arm_dof_ids = [model.jnt_dofadr[jid] for jid in arm_joint_ids]
    n_arm = len(arm_dof_ids)

    # ---- Jacobian at gripperframe (full 3×nv) --------------------------
    jacp_full = np.zeros((3, model.nv), dtype=np.float64)
    jacr_full = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data_raw, jacp_full, jacr_full, site_id)

    # Extract columns for the arm DOFs only.
    J = jacp_full[:, arm_dof_ids]  # (3, n_arm)

    # ---- damped least-squares: joint_vel = J^+ * ee_velocity -----------
    JtJ = J.T @ J + (DAMPING ** 2) * np.eye(n_arm)
    dq = np.linalg.solve(JtJ, J.T @ ee_velocity)  # (n_arm,)

    # Clamp joint velocities.
    max_abs = np.abs(dq).max()
    if max_abs > MAX_JOINT_VEL:
        dq *= MAX_JOINT_VEL / max_abs

    # Build full 6-DOF action: 5 arm joint velocities + gripper position.
    action = np.zeros(6, dtype=np.float64)
    action[:5] = dq
    action[5] = gripper_action

    return action
