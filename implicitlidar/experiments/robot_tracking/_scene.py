"""Build the target and (optional) occluder scenes for the robot tracking experiment.

Trajectory-generation helpers shared by the train and evaluate scripts;
identical seeds yield the same trajectory family.
"""

from __future__ import annotations

import torch

from implicitlidar.core import SceneSDF
from implicitlidar.scenes.robot_arm import (
    KukaTrajectoryGenerator,
    end_effector_scene,
    robot_arm_scene,
)


def build_robot_scenes(
    config: dict,
    *,
    device: torch.device | str,
) -> tuple[SceneSDF, SceneSDF | None, list[tuple]]:
    """Generate trajectories and return ``(target_scene, occluder_scene, trajectories)``.

    The target scene is the union of small-sphere SDFs at every end-effector
    pose along every trajectory; this is what the implicit density learns to
    "see". The occluder scene is the union of robot arm SDFs over the same
    trajectories — populated only when ``config.scene.occluder == 'robot'``.
    """
    scene_cfg = config["scene"]
    if scene_cfg["source"] != "sphere":
        raise NotImplementedError(f"Unsupported scene source: {scene_cfg['source']!r}")

    tg = KukaTrajectoryGenerator()
    trajectories: list[tuple] = []
    target_sdfs: list = []
    occluder_sdfs: list = []

    use_occluder = scene_cfg.get("occluder") == "robot"
    base_seed = int(scene_cfg.get("seed", 42))
    for i in range(int(scene_cfg["num_trajectories"])):
        joint_traj, ee_traj = tg.sample(
            seed=base_seed + i,
            place_noise_xy=tuple(scene_cfg.get("trajectory_variance", (0.0, 0.0))),
            steps_per_segment=int(scene_cfg.get("steps_per_segment", 10)),
        )
        trajectories.append((joint_traj, ee_traj))
        target_sdfs.extend(end_effector_scene(
            ee_traj, sphere_radius=float(scene_cfg.get("sphere_radius", 0.02)), device=device,
        ).sdfs)
        if use_occluder:
            occluder_sdfs.extend(robot_arm_scene(joint_traj, device=device).sdfs)

    target_scene = SceneSDF(target_sdfs).to(device)
    occluder_scene = SceneSDF(occluder_sdfs).to(device) if use_occluder else None
    return target_scene, occluder_scene, trajectories
