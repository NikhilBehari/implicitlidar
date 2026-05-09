# warehouse_detection

Motion-adaptive scanning design for a ground robot moving along a
warehouse aisle. Trains the implicit density conditioned on the robot's
motion `x ∈ [0, X]` and synthesizes a per-position scanning pattern (a
small-budget ray mixture) at each point along the trajectory. Evaluates
box-detection miss rate against even and random scanning baselines on a
held-out front-plane test set.

## Prepare

```bash
python -m implicitlidar.experiments.warehouse_detection.prepare \
    --out outputs/data/warehouse \
    --n-scenes 100 --seed 0
```

The generator writes paired `target/` and `occluder/` mesh directories
plus a `boxes.json` metadata file enumerating each scene's box AABBs (the
detection targets).

## Run

```bash
python -m implicitlidar.experiments.warehouse_detection.train       --config implicitlidar/experiments/warehouse_detection/configs/default.yaml
python -m implicitlidar.experiments.warehouse_detection.synthesize  --config implicitlidar/experiments/warehouse_detection/configs/default.yaml --rays-per-position 10
python -m implicitlidar.experiments.warehouse_detection.evaluate    --config implicitlidar/experiments/warehouse_detection/configs/default.yaml
```

Repeat the synthesize step for additional ray budgets (e.g.
`--rays-per-position 5` and `--rays-per-position 20`); each writes its
own `sensors_motion_adaptive_r<n>.csv` and the evaluator picks them up
automatically.

## Outputs

```
outputs/runs/warehouse_detection/default/
├── config.yaml                                    snapshot of the config used
├── checkpoints/                                   periodic and final flow checkpoints
├── sensors/sensors_motion_adaptive_r<n>.csv       per-position rays at each ray-budget
└── results/miss_rate.csv                          miss rate and bandwidth per method × budget
```

## Hyperparameters

| Quantity | Value |
|---|---|
| Robot motion range | x ∈ [−0.5, 4.0] |
| Sensor location | y = z = 0 (ground robot) |
| Azimuth | fixed at −π/2 (looks toward shelf row) |
| Elevation range | [0, π] (full upper hemisphere) |
| Time-of-flight range | [0.1, 5.0] m |
| Surface-proximity σ | 0.04 |
| Visibility | on (occluder: shelf metal) |
| Flow layers / hidden / bins | 8 / 128 / 16 |
| Training iterations | 5 000 |
| Batch size | 4 096 |
| Entropy regularization | 0.5 |
| Query positions for synthesis | x ∈ {0.0, 1.2, 2.4, 3.6} |
| Position window | 0.1 m |
| Held-out test scenes | 40 (seed 84) |
