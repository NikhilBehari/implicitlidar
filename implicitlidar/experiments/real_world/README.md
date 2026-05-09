# real_world

Real-world face-scanning validation on a single-photon-avalanche-diode
(SPAD) capture. Peak-finds per-pixel time-of-flight to recover a 3D point
cloud, then samples that cloud through the rays implied by the
synthesized sensor designs from `face_scanning` (or by uniform / random
scanning baselines) and reconstructs a surface mesh.

## Capture setup

The measurement file `depth_transient.mat` was acquired with:

| Component | Spec |
|---|---|
| Detector | Micro Photon Devices PDM SPAD module (timing jitter 50 ps FWHM) |
| Emitter | pulsed laser (≤ 100 fs FWHM pulse width) |
| Steering | two-axis galvanometer (±20°/axis ⇒ 40° total FoV) |
| Bin width | 8 ps; instrument response 50 ps FWHM |
| Beam diameter | ~1.2 mm at exit; circularity > 90 % |

## Run

```bash
# 1. Process the SPAD measurement into a 3D point cloud.
python -m implicitlidar.experiments.real_world.process_measurement \
    --transient outputs/data/real_world/depth_transient.mat \
    --out       outputs/data/real_world/point_cloud.npy \
    --t0 3925 --t-min 4000 --t-max 4350 \
    --y0 60 --y1 210 --x0 78 --x1 178 \
    --depth-scale 0.009

# 2. Reconstruct a face mesh by sampling the cloud through the
#    synthesized sensor design (sensor CSV from face_scanning).
python -m implicitlidar.experiments.real_world.reconstruct \
    --point-cloud outputs/data/real_world/point_cloud.npy \
    --sensors     outputs/runs/face_scanning/default/sensors/sensors_2.csv \
    --rays 576 \
    --out outputs/runs/real_world/sensors_2_576.obj

# 3. Reconstruct using the uniform baseline at the same ray budget.
python -m implicitlidar.experiments.real_world.reconstruct \
    --point-cloud outputs/data/real_world/point_cloud.npy --baseline uniform \
    --rays 576 --out outputs/runs/real_world/uniform_576.obj
```

## Hyperparameters

| Quantity | Value |
|---|---|
| Pulse-onset bin (`t0`) | 3925 |
| Accept window (`t_min, t_max`) | (4000, 4350) bins |
| Pixel ROI | rows 60–210, cols 78–178 |
| Depth scale | 0.009 m / bin |
| Match tolerance | 0.025 m perpendicular distance |
| Total rays per design | 576 (24 × 24 grid) |
