"""Robot-arm scenes for robot_tracking (end-effector tracking).

* :class:`KukaTrajectoryGenerator` — samples randomized 7-DoF KUKA IIWA
  pick-and-place trajectories in MuJoCo, solving inverse kinematics through
  approach, transit, and descent waypoints.
* :func:`end_effector_scene` — given a trajectory, builds a :class:`SceneSDF`
  consisting of a small sphere placed at every end-effector pose. This is the
  ``target`` geometry for tracking.
* :func:`robot_arm_scene` — given a trajectory, builds a :class:`SceneSDF`
  consisting of the robot arm's full link geometry at each pose. This is the
  ``occluder`` geometry used by the visibility term to penalize rays that
  pass through the arm.

All three share a single MuJoCo model loaded from the bundled KUKA IIWA 14
description in :data:`KUKA_MJCF_PATH`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..core import SceneSDF

# Default MuJoCo scene XML for the KUKA IIWA 14 (vendored under assets/).
KUKA_MJCF_PATH: Path = (
    Path(__file__).resolve().parents[2] / "assets" / "kuka_iiwa_14" / "scene.xml"
)


# Quaternion helpers


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    q_inv = np.array([current[0], -current[1], -current[2], -current[3]])
    err = _quat_mul(target, q_inv)
    if err[0] < 0:
        err = -err
    return 2.0 * err[1:]


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


# Trajectory generation


class KukaTrajectoryGenerator:
    """Sample randomized pick-and-place trajectories for the KUKA IIWA 14.

    Each trajectory follows the same six-waypoint template (approach pick,
    pick, retract, transit, descent, place), with a
    Gaussian perturbation on the target *place* location to inject motion
    variability across the dataset.

    Solving IK at each waypoint is done by damped-least-squares gradient
    descent inside MuJoCo (no external IK library required).
    """

    def __init__(
        self,
        model_xml: str | Path = KUKA_MJCF_PATH,
        *,
        ik_step_size: float = 0.01,
        ik_tol: float = 1e-3,
        ik_damping: float = 0.2,
        ik_orientation_weight: float = 0.1,
        ik_max_iters: int = 1000,
    ):
        import mujoco
        self.model_xml = str(model_xml)
        self.model = mujoco.MjModel.from_xml_path(self.model_xml)
        self.data = mujoco.MjData(self.model)
        self.ik = _DampedLeastSquaresIK(
            self.model,
            self.data,
            step_size=ik_step_size,
            tol=ik_tol,
            damping=ik_damping,
            orientation_weight=ik_orientation_weight,
            max_iters=ik_max_iters,
        )

    def sample(
        self,
        *,
        seed: int | None = None,
        place_noise_xy: tuple[float, float] = (0.05, 0.05),
        steps_per_segment: int = 10,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Sample one randomized pick-and-place trajectory.

        Returns
        -------
        joint_traj : ``(T, 7)`` array of joint configurations along the trajectory.
        end_effector_traj : list of ``(position, quaternion)`` pairs at every
            timestep, with the end-effector "carrying" the picked object after
            it grasps and releasing at the place location.
        """
        import mujoco
        rng = np.random.RandomState(seed) if seed is not None else np.random

        pick = np.array([0.3, 0.3, 0.05])
        place = np.array([-0.3, -0.3, 0.05]) + rng.normal(
            scale=np.array([place_noise_xy[0], place_noise_xy[1], 0.0])
        )
        waypoints = _pick_place_waypoints(pick, place)

        end_id = self.model.body("link7").id
        q_init = np.deg2rad([0, -45, 0, 90, 0, 45, 0])
        down_quat = np.array([0, 1, 0, 0])

        joint_configs: list[np.ndarray] = []
        q_curr = q_init.copy()
        for i, wp in enumerate(waypoints):
            target_quat = down_quat if i in (0, len(waypoints) - 1) else None
            q_sol = self.ik.solve(wp, q_curr, end_id, target_quat=target_quat)
            joint_configs.append(q_sol)
            q_curr = q_sol.copy()

        segments = [
            np.linspace(joint_configs[i], joint_configs[i + 1], num=steps_per_segment)
            for i in range(len(joint_configs) - 1)
        ]
        joint_traj = np.concatenate(segments, axis=0)

        # Carry the "object" at the end-effector position once we grasp.
        # The object's resting pose comes from the MuJoCo mocap body, which
        # sits a small offset above the pick location.
        end_effector_traj = []
        attached = False
        local_offset = np.zeros(3)
        threshold = 0.01
        mocap_idx = 0
        mujoco.mj_resetData(self.model, self.data)
        self.data.mocap_pos[mocap_idx] = pick.copy()
        for q in joint_traj:
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            ee_pos = self.data.body(end_id).xpos.copy()
            ee_quat = self.data.body(end_id).xquat.copy()
            if not attached and np.linalg.norm(ee_pos - pick) < threshold:
                attached = True
                local_offset = self.data.mocap_pos[mocap_idx].copy() - ee_pos
            if attached:
                self.data.mocap_pos[mocap_idx] = ee_pos + local_offset
                self.data.mocap_quat[mocap_idx] = ee_quat
            end_effector_traj.append(
                (self.data.mocap_pos[mocap_idx].copy(), ee_quat.copy())
            )
        return joint_traj, end_effector_traj


# SceneSDFs: end-effector targets, robot-arm occluder


def end_effector_scene(
    end_effector_traj: list[tuple[np.ndarray, np.ndarray]],
    *,
    sphere_radius: float = 0.02,
    device: torch.device | str = "cpu",
) -> SceneSDF:
    """Sphere SDFs placed at every end-effector pose.

    Each timestep contributes one rigidly-transformed sphere SDF over a
    tessellated sphere mesh; the resulting :class:`SceneSDF` is the *target*
    geometry for tracking.
    """
    base_sdf = _build_meshed_sphere_sdf(sphere_radius)
    return SceneSDF([
        _PosedSphereSDF(base_sdf, position=pos, quat=quat).to(device)
        for pos, quat in end_effector_traj
    ])


def robot_arm_scene(
    joint_traj: np.ndarray,
    *,
    end_effector_link: str = "lbr_iiwa_link_7",
    device: torch.device | str = "cpu",
) -> SceneSDF:
    """Robot-arm link SDFs at every joint configuration along the trajectory.

    The resulting :class:`SceneSDF` is the *occluder* geometry: the visibility
    term in the target density penalizes rays whose path passes through the
    arm's links. The arm geometry is loaded from the KUKA URDF bundled with
    `pybullet_data` (which ``pip install pybullet`` provides), then wrapped as
    a `pytorch_volumetric.RobotSDF`.
    """
    import pybullet_data
    import pytorch_kinematics as pk
    import pytorch_volumetric as pv

    asset_root = Path(pybullet_data.getDataPath()) / "kuka_iiwa"
    urdf_path = asset_root / "model.urdf"
    chain = pk.build_serial_chain_from_urdf(urdf_path.read_text(), end_effector_link)
    chain = chain.to(device=device)
    base = pv.RobotSDF(chain, path_prefix=str(asset_root))
    return SceneSDF([_PosedRobotSDF(base, q, device=device) for q in joint_traj])


class _PosedRobotSDF(nn.Module):
    """Wrap a `pv.RobotSDF` so it always evaluates at a fixed joint configuration."""

    def __init__(self, robot_sdf, q: np.ndarray, *, device: torch.device | str):
        super().__init__()
        self._robot_sdf = robot_sdf
        self.register_buffer("q", torch.tensor(np.asarray(q), dtype=torch.float32))
        self.device = torch.device(device)

    def forward(self, pts: torch.Tensor):
        self._robot_sdf.set_joint_configuration(self.q.to(self.device))
        return self._robot_sdf(pts)


# Internal modules


class _PosedSphereSDF(nn.Module):
    """A meshed sphere SDF rigidly transformed to a given pose.

    Wraps a shared base ``pv.MeshSDF`` (built once per radius via
    :func:`_build_meshed_sphere_sdf`) and applies a fixed rotation/translation
    when forwarding queries.
    """

    def __init__(self, base_sdf, position: np.ndarray, quat: np.ndarray):
        super().__init__()
        self._base_sdf = base_sdf
        self.register_buffer("pos", torch.tensor(position, dtype=torch.float32))
        self.register_buffer("R", torch.tensor(_quat_to_rot(quat), dtype=torch.float32))

    def forward(self, pts: torch.Tensor):
        local = (pts - self.pos) @ self.R.T
        return self._base_sdf(local)


_SPHERE_SDF_CACHE: dict[float, object] = {}


def _build_meshed_sphere_sdf(radius: float):
    """Return a cached ``pv.MeshSDF`` over a tessellated sphere of given radius.

    Meshes are written to ``outputs/data/sdf_cache/sphere_r<radius>.obj`` and
    cached in-process so all sphere instances share one SDF.
    """
    import os

    import pytorch_volumetric as pv

    if radius in _SPHERE_SDF_CACHE:
        return _SPHERE_SDF_CACHE[radius]

    cache_dir = os.path.join("outputs", "data", "sdf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    obj_path = os.path.join(cache_dir, f"sphere_r{radius}.obj")
    if not os.path.exists(obj_path):
        _write_tessellated_sphere_obj(obj_path, radius=radius, segments=20, rings=20)
    sdf = pv.MeshSDF(pv.MeshObjectFactory(obj_path))
    _SPHERE_SDF_CACHE[radius] = sdf
    return sdf


def _write_tessellated_sphere_obj(path: str, *, radius: float, segments: int, rings: int) -> None:
    """Write a UV-sphere OBJ (20 segments × 20 rings)."""
    vertices = []
    for i in range(rings + 1):
        theta = math.pi * i / rings
        for j in range(segments):
            phi = 2 * math.pi * j / segments
            vertices.append((
                radius * math.sin(theta) * math.cos(phi),
                radius * math.sin(theta) * math.sin(phi),
                radius * math.cos(theta),
            ))
    faces = []
    for i in range(rings):
        for j in range(segments):
            next_j = (j + 1) % segments
            idx0 = i * segments + j
            idx1 = i * segments + next_j
            idx2 = (i + 1) * segments + j
            idx3 = (i + 1) * segments + next_j
            if i != 0:
                faces.append((idx0 + 1, idx2 + 1, idx1 + 1))
            if i != rings - 1:
                faces.append((idx1 + 1, idx2 + 1, idx3 + 1))
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


class _DampedLeastSquaresIK:
    """Damped least-squares IK solver against a MuJoCo model."""

    def __init__(self, model, data, *, step_size, tol, damping, orientation_weight, max_iters):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.damping = damping
        self.orientation_weight = orientation_weight
        self.max_iters = max_iters
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def solve(
        self,
        goal_pos: np.ndarray,
        init_q: np.ndarray,
        body_id: int,
        target_quat: np.ndarray | None = None,
    ) -> np.ndarray:
        import mujoco
        self.data.qpos[:] = init_q
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.max_iters):
            current_pos = self.data.body(body_id).xpos
            pos_err = goal_pos - current_pos
            if target_quat is not None:
                ori_err = _quat_error(target_quat, self.data.body(body_id).xquat)
                err = np.concatenate([pos_err, self.orientation_weight * ori_err])
            else:
                err = pos_err
            if np.linalg.norm(err) <= self.tol:
                break
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal_pos, body_id)
            J = (
                np.vstack([self.jacp, self.orientation_weight * self.jacr])
                if target_quat is not None else self.jacp
            )
            A = J.T @ J + self.damping * np.eye(self.model.nv)
            delta_q = np.linalg.pinv(A) @ J.T @ err
            self.data.qpos[:] += self.step_size * delta_q
            for i in range(len(self.data.qpos)):
                self.data.qpos[i] = np.clip(
                    self.data.qpos[i],
                    self.model.jnt_range[i][0],
                    self.model.jnt_range[i][1],
                )
            mujoco.mj_forward(self.model, self.data)
        return self.data.qpos.copy()


def _pick_place_waypoints(pick: np.ndarray, place: np.ndarray) -> list[np.ndarray]:
    """Return the six waypoints of an approach / pick / transit / place trajectory."""
    return [
        pick + np.array([0, 0, 0.3]),
        pick.copy(),
        pick + np.array([0, 0, 0.4]),
        np.array([0.0, 0.6, 0.65]),
        place + np.array([0, 0, 0.4]),
        place.copy(),
    ]
