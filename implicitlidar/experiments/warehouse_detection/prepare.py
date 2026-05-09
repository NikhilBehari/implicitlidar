"""Generate the procedural warehouse dataset used by warehouse_detection.

Produces ``n_scenes`` watertight OBJ meshes of multi-shelf warehouse rows
with randomized shelf heights and box sizes.

Usage::

    python -m implicitlidar.experiments.warehouse_detection.prepare \\
        --out outputs/data/warehouse \\
        --n-scenes 100 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from implicitlidar.scenes.warehouse import DEFAULT_LAYOUT, MultiShelfGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/data/warehouse"),
                        help="Output directory for generated scenes.")
    parser.add_argument("--n-scenes", type=int, default=100,
                        help="Number of warehouse scenes to generate.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the scene generator.")
    parser.add_argument("--shelf-spacing-x", type=float, default=0.2,
                        help="Spacing in meters between adjacent shelf rows. "
                             "Must match the value the evaluation config expects "
                             "(default 0.2, see configs/default.yaml).")
    parser.add_argument("--sdf-grid-size", type=int, default=128,
                        help="Resolution of the mesh2sdf grid used to make scenes watertight.")
    args = parser.parse_args()

    generator = MultiShelfGenerator(
        layout=DEFAULT_LAYOUT,
        shelf_spacing_x=args.shelf_spacing_x,
        sdf_grid_size=args.sdf_grid_size,
    )
    written = generator.generate_dataset(
        args.out, n_scenes=args.n_scenes, seed=args.seed,
        split_targets_and_occluders=True,
    )
    for category, paths in written.items():
        print(f"Wrote {len(paths)} {category} mesh(es) to {args.out / category}")


if __name__ == "__main__":
    main()
