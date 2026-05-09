# robot_tracking

Distributed ceiling-mounted LiDAR design for tracking the end-effector of
a 7-DoF KUKA IIWA performing randomized pick-and-place trajectories. Trains
the implicit density with learnable origin covariances, synthesizes 2, 4, and 8-sensor designs via EM, and evaluates trajectory-reconstruction
quality (Fréchet distance) against uniform and random baselines.

No external dataset is required: trajectories are sampled deterministically
from a seed using the bundled KUKA IIWA model
([`assets/kuka_iiwa_14`](../../../assets/kuka_iiwa_14)).

## Run

```bash
python -m implicitlidar.experiments.robot_tracking.train      --config implicitlidar/experiments/robot_tracking/configs/default.yaml
python -m implicitlidar.experiments.robot_tracking.synthesize --config implicitlidar/experiments/robot_tracking/configs/default.yaml --components 2,4,8
python -m implicitlidar.experiments.robot_tracking.evaluate   --config implicitlidar/experiments/robot_tracking/configs/default.yaml
```

## Visibility-term ablation

The default config uses `use_visibility: false`. The paired
`with_visibility.yaml` enables the visibility term and the robot arm as
an occluder; running both side-by-side quantifies the value of modeling
view-dependent occlusion.

```bash
python -m implicitlidar.experiments.robot_tracking.train      --config implicitlidar/experiments/robot_tracking/configs/with_visibility.yaml
python -m implicitlidar.experiments.robot_tracking.synthesize --config implicitlidar/experiments/robot_tracking/configs/with_visibility.yaml
python -m implicitlidar.experiments.robot_tracking.evaluate   --config implicitlidar/experiments/robot_tracking/configs/with_visibility.yaml
```

## Hyperparameters

| Quantity | Value |
|---|---|
| Sensor placement | ceiling (z = 1 m), x ∈ [−1, 1], y ∈ [−1, 1] |
| Azimuth range | [0, 2π] |
| Elevation range | [−π/2, 0] |
| Time-of-flight range | [0.1, 2.0] m |
| Surface-proximity σ | 0.025 |
| Visibility | off (default) / on (ablation) |
| Trajectories sampled | 20 (training), 40 (eval) |
| Trajectory variance | (0.10 m, 0.10 m) on the place location |
| Sensor origin covariance | full 3×3, clamped to 0.1 m thickness |
| Flow layers / hidden / bins | 8 / 64 / 16 |
| Training iterations | 5 000 |
| Batch size | 1 024 |
| Entropy regularization | 0.5 |
