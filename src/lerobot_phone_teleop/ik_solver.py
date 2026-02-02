"""Velocity-based inverse kinematics solver for SO101 robot."""

import numpy as np
import mujoco
from numpy.typing import NDArray
from typing import Tuple

from .config import RobotConfig


class IKSolver:
    """Velocity-based IK solver using damped least squares.

    Computes joint velocities from end-effector velocity commands using
    the Jacobian pseudo-inverse with damping for singularity robustness.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data,
        config: RobotConfig = None,
        damping: float = 0.05,
    ):
        """Initialize IK solver.

        Args:
            model: MuJoCo model (raw mujoco.MjModel).
            data: MuJoCo data (dm_control wrapper with .ptr attribute).
            config: Robot configuration.
            damping: Damping coefficient for pseudo-inverse.
        """
        self.model = model
        # Store dm_control wrapper for array access
        self._data_wrapper = data
        # Get raw pointer for mujoco functions
        self.data = data.ptr if hasattr(data, 'ptr') else data
        self.config = config or RobotConfig()
        self.damping = damping

        # Cache site and body IDs
        self.ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, self.config.ee_site_name
        )
        if self.ee_site_id < 0:
            raise ValueError(f"Site '{self.config.ee_site_name}' not found")

        # Get the body that the site belongs to
        self.ee_body_id = model.site_bodyid[self.ee_site_id]

        # Cache joint IDs and qpos/qvel addresses
        self.joint_ids = []
        self.qpos_addrs = []
        self.qvel_addrs = []

        for joint_name in self.config.arm_joint_names:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise ValueError(f"Joint '{joint_name}' not found")
            self.joint_ids.append(joint_id)
            self.qpos_addrs.append(model.jnt_qposadr[joint_id])
            self.qvel_addrs.append(model.jnt_dofadr[joint_id])

        self.n_joints = len(self.joint_ids)

        # Pre-allocate Jacobian arrays (3 x nv for position, 3 x nv for rotation)
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def get_ee_pose(self) -> Tuple[NDArray, NDArray]:
        """Get current end-effector position and orientation.

        Returns:
            Tuple of (position [3], quaternion [4] as wxyz).
        """
        pos = self._data_wrapper.site_xpos[self.ee_site_id].copy()
        # Site orientation is stored as 3x3 rotation matrix
        rot_mat = self._data_wrapper.site_xmat[self.ee_site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot_mat.flatten())
        return pos, quat

    def get_jacobian(self) -> Tuple[NDArray, NDArray]:
        """Compute the Jacobian for the end-effector.

        Returns:
            Tuple of (Jp [3 x n_joints], Jr [3 x n_joints]) for position
            and rotation Jacobians, extracted for the arm joints only.
        """
        # Compute full Jacobian
        mujoco.mj_jacSite(
            self.model, self.data,
            self.jacp, self.jacr,
            self.ee_site_id
        )

        # Extract columns for arm joints only
        jp = np.zeros((3, self.n_joints))
        jr = np.zeros((3, self.n_joints))

        for i, dof_addr in enumerate(self.qvel_addrs):
            jp[:, i] = self.jacp[:, dof_addr]
            jr[:, i] = self.jacr[:, dof_addr]

        return jp, jr

    def compute_joint_velocities(
        self,
        linear_velocity: NDArray,
        angular_velocity: NDArray,
        max_velocity: float = 2.0,
    ) -> NDArray:
        """Compute joint velocities from end-effector velocity command.

        Uses damped least squares pseudo-inverse for robustness near
        singularities.

        Args:
            linear_velocity: Desired EE linear velocity [3] in base frame (m/s).
            angular_velocity: Desired EE angular velocity [3] in EE frame (rad/s).
            max_velocity: Maximum joint velocity magnitude (rad/s).

        Returns:
            Joint velocities [n_joints] in rad/s.
        """
        jp, jr = self.get_jacobian()

        # Stack Jacobians: [6 x n_joints]
        J = np.vstack([jp, jr])

        # Stack velocity commands: [6]
        v_des = np.concatenate([linear_velocity, angular_velocity])

        # Damped least squares: q_dot = J^T (J J^T + λ²I)^{-1} v
        JJT = J @ J.T
        damping_matrix = (self.damping ** 2) * np.eye(6)
        q_dot = J.T @ np.linalg.solve(JJT + damping_matrix, v_des)

        # Clip to max velocity
        max_vel = np.max(np.abs(q_dot))
        if max_vel > max_velocity:
            q_dot = q_dot * (max_velocity / max_vel)

        return q_dot

    def get_current_joint_positions(self) -> NDArray:
        """Get current arm joint positions.

        Returns:
            Joint positions [n_joints] in rad.
        """
        positions = np.zeros(self.n_joints)
        for i, addr in enumerate(self.qpos_addrs):
            positions[i] = self._data_wrapper.qpos[addr]
        return positions

    def integrate_joint_positions(
        self,
        joint_velocities: NDArray,
        dt: float,
    ) -> NDArray:
        """Integrate joint velocities to get target positions.

        Args:
            joint_velocities: Joint velocities [n_joints] in rad/s.
            dt: Time step in seconds.

        Returns:
            Target joint positions [n_joints] in rad, clamped to limits.
        """
        current_pos = self.get_current_joint_positions()
        target_pos = current_pos + joint_velocities * dt

        # Clamp to joint limits
        lower = np.array(self.config.joint_limits_lower)
        upper = np.array(self.config.joint_limits_upper)
        target_pos = np.clip(target_pos, lower, upper)

        return target_pos
