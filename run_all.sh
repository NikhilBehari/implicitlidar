#!/usr/bin/env bash
# End-to-end driver: face_scanning, robot_tracking, warehouse_detection,
# emitter_design, the visibility-term ablation, and the real-world SPAD demo.
# Pass --dry-run to print commands without executing.

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

run() {
    echo "+ $*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}

cd "$(dirname "$0")"


# Dataset preparation (skipped when data already exists).

if [[ ! -d outputs/data/faces/train ]]; then
    echo "Basel face dataset not found; populate it via:"
    echo "    python -m implicitlidar.experiments.face_scanning.prepare \\"
    echo "        --bfm-h5 path/to/model2019_fullHead.h5"
    exit 1
fi

if [[ ! -d outputs/data/warehouse/target ]]; then
    run python -m implicitlidar.experiments.warehouse_detection.prepare \
        --out outputs/data/warehouse \
        --n-scenes 100 --seed 0
fi


# face_scanning
EXP_A=implicitlidar/experiments/face_scanning
run python -m implicitlidar.experiments.face_scanning.train      --config "$EXP_A/configs/default.yaml"
run python -m implicitlidar.experiments.face_scanning.synthesize --config "$EXP_A/configs/default.yaml" --components 2,4,10
run python -m implicitlidar.experiments.face_scanning.evaluate   --config "$EXP_A/configs/default.yaml"


# robot_tracking
EXP_B=implicitlidar/experiments/robot_tracking
run python -m implicitlidar.experiments.robot_tracking.train      --config "$EXP_B/configs/default.yaml"
run python -m implicitlidar.experiments.robot_tracking.synthesize --config "$EXP_B/configs/default.yaml" --components 2,4,8
run python -m implicitlidar.experiments.robot_tracking.evaluate   --config "$EXP_B/configs/default.yaml"


# warehouse_detection
EXP_C=implicitlidar/experiments/warehouse_detection
run python -m implicitlidar.experiments.warehouse_detection.train      --config "$EXP_C/configs/default.yaml"
for r in 5 10 20; do
    run python -m implicitlidar.experiments.warehouse_detection.synthesize \
        --config "$EXP_C/configs/default.yaml" --rays-per-position "$r"
done
run python -m implicitlidar.experiments.warehouse_detection.evaluate    --config "$EXP_C/configs/default.yaml"


# emitter_design (Mitsuba 3 transient evaluation)
EXP_D=implicitlidar/experiments/emitter_design
run python -m implicitlidar.experiments.emitter_design.train      --config "$EXP_D/configs/default.yaml"
run python -m implicitlidar.experiments.emitter_design.synthesize --config "$EXP_D/configs/default.yaml" --components 1,2,4
run python -m implicitlidar.experiments.emitter_design.evaluate   --config "$EXP_D/configs/default.yaml" --components 1,2,4


# robot_tracking with the visibility term enabled
run python -m implicitlidar.experiments.robot_tracking.train      --config "$EXP_B/configs/with_visibility.yaml"
run python -m implicitlidar.experiments.robot_tracking.synthesize --config "$EXP_B/configs/with_visibility.yaml" --components 2,4,8
run python -m implicitlidar.experiments.robot_tracking.evaluate   --config "$EXP_B/configs/with_visibility.yaml"


# real_world (SPAD validation; needs a captured transient .mat)
if [[ -f outputs/data/real_world/depth_transient.mat ]]; then
    run python -m implicitlidar.experiments.real_world.process_measurement \
        --transient outputs/data/real_world/depth_transient.mat \
        --out       outputs/data/real_world/point_cloud.npy \
        --t0 3925 --t-min 4000 --t-max 4350 \
        --y0 60 --y1 210 --x0 78 --x1 178 \
        --depth-scale 0.009
    for n in 2 4 10; do
        run python -m implicitlidar.experiments.real_world.reconstruct \
            --point-cloud outputs/data/real_world/point_cloud.npy \
            --sensors     outputs/runs/face_scanning/default/sensors/sensors_${n}.csv \
            --rays 576 \
            --out outputs/runs/real_world/sensors_${n}_576.obj
    done
    run python -m implicitlidar.experiments.real_world.reconstruct \
        --point-cloud outputs/data/real_world/point_cloud.npy --baseline uniform \
        --rays 576 --out outputs/runs/real_world/uniform_576.obj
else
    echo "Skipping real_world: outputs/data/real_world/depth_transient.mat not found"
fi

echo
echo "All experiments complete. Numerical results are in:"
echo "  outputs/runs/<experiment>/results/*.csv"
