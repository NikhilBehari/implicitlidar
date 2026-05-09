# emitter_design

Constraint-aware emitter synthesis for a fixed smartphone-LiDAR detector.
Trains the implicit density over emitter placement and direction,
synthesizes single-flood, multi-flood, and bistatic emitter
configurations via EM, and evaluates each via Mitsuba 3 transient
light-transport rendering. The transient peak per detector pixel gives
the optical path length, from which the surface point is recovered by
solving the bistatic geometry; reconstructed face meshes are scored by
mesh-vs-mesh squared Chamfer.

## Install

```bash
pip install -e ".[emitter]"
```

This pulls in `mitsuba >= 3.5` and
[`mitransient`](https://github.com/diegoroyo/mitransient).

## Run

```bash
python -m implicitlidar.experiments.emitter_design.train      --config implicitlidar/experiments/emitter_design/configs/default.yaml
python -m implicitlidar.experiments.emitter_design.synthesize --config implicitlidar/experiments/emitter_design/configs/default.yaml --components 1,2,4
python -m implicitlidar.experiments.emitter_design.evaluate   --config implicitlidar/experiments/emitter_design/configs/default.yaml --components 1,2,4 \
    --temporal-bins 20 --resolution 256 --spp 2048
```

The renderer's variant defaults to `llvm_ad_rgb` (CPU). Switch to
`cuda_ad_rgb` from a wrapping script if you have a CUDA-capable GPU and
want GPU rendering.

## Bundled scene

The Mitsuba scene template (perspective sensor, transient HDR film, one
or more projector emitters, and a shape slot for the target mesh) lives
at
[`assets/scenes/lidar_template.xml`](../../../assets/scenes/lidar_template.xml).
The evaluator patches the projector list, the target object path, the
resolution, and the temporal bins for every render.

## Hyperparameters

| Quantity | Value |
|---|---|
| Detector origin | (3 m, 0, 0) |
| Detector FoV | 60° (h) × 50° (w) |
| Emitter placement region | (x, y, z) ∈ [−1, 1]² × [0, 1] m |
| Emitter azimuth range | [3π/4, 5π/4] |
| Emitter elevation range | [−π/4, π/4] |
| Emitter origin covariance | full 3×3, clamped to 0.05 m thickness |
| Surface-proximity σ | 0.05 |
| Mitsuba spp | 2 048 |
| Temporal bins | 20 |
| Sensor resolution | 256 × 256 |
