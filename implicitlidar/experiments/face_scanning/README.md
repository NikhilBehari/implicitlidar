# face_scanning

Smartphone flash-LiDAR design for 3D face mesh reconstruction. Trains the
implicit density on 50 face meshes sampled from the Basel Face Model,
synthesizes 2, 4, and 10-sensor designs via EM, and evaluates Chamfer
distance against uniform and random scanning baselines across a sweep of
ray budgets.

## Prepare

Register and download `model2019_fullHead.h5` from
[faces.dmi.unibas.ch](https://faces.dmi.unibas.ch), then:

```bash
python -m implicitlidar.experiments.face_scanning.prepare \
    --bfm-h5 path/to/model2019_fullHead.h5 \
    --train-out outputs/data/faces/train --n-train 50 \
    --test-out  outputs/data/faces/test  --n-test  50
```

## Run

```bash
python -m implicitlidar.experiments.face_scanning.train      --config implicitlidar/experiments/face_scanning/configs/default.yaml
python -m implicitlidar.experiments.face_scanning.synthesize --config implicitlidar/experiments/face_scanning/configs/default.yaml --components 2,4,10
python -m implicitlidar.experiments.face_scanning.evaluate   --config implicitlidar/experiments/face_scanning/configs/default.yaml
```

CLI overrides apply without editing the YAML, e.g.
`--override training.iterations=200 output.run_dir=outputs/runs/face_scanning/smoke`.

## Outputs

```
outputs/runs/face_scanning/default/
├── config.yaml          snapshot of the config used
├── checkpoints/         periodic and final flow checkpoints
├── sensors/             per-component-count sensor CSVs
└── results/chamfer.csv  Chamfer distance and bandwidth per ray budget
```

## Hyperparameters

| Quantity | Value |
|---|---|
| Sensor origin | (3.0 m, 0, 0) |
| Azimuth range | [3π/4, 5π/4] |
| Elevation range | [−π/4, π/4] |
| Time-of-flight range | [0.1, 5.0] m |
| Surface-proximity σ | 0.075 |
| Visibility β | 100.0 |
| Flow layers / hidden / bins | 6 / 64 / 16 |
| Training iterations | 2 000 |
| Batch size | 2 048 |
| Learning rate | 1 × 10⁻³ |
| Entropy regularization | 0.75 |
