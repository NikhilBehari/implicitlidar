"""Generate the face-mesh dataset used by face_scanning.

Samples random faces from the Basel Face Model 2009 PCA basis and exports
them as OBJ meshes, normalized to the height and orientation expected by
the training pipeline. Register and download ``model2019_fullHead.h5``
from https://faces.dmi.unibas.ch first.

Usage::

    python -m implicitlidar.experiments.face_scanning.prepare \\
        --bfm-h5 path/to/model2019_fullHead.h5 \\
        --train-out outputs/data/faces/train --n-train 50 \\
        --test-out  outputs/data/faces/test  --n-test  50
"""

from __future__ import annotations

import argparse
from pathlib import Path

from implicitlidar.scenes.faces import generate_face_meshes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfm-h5", type=Path, required=True,
                        help="Path to model2019_fullHead.h5 (Basel Face Model 2009).")
    parser.add_argument("--train-out", type=Path, default=Path("outputs/data/faces/train"),
                        help="Output directory for training faces.")
    parser.add_argument("--n-train", type=int, default=50,
                        help="Number of training faces to sample.")
    parser.add_argument("--test-out", type=Path, default=Path("outputs/data/faces/test"),
                        help="Output directory for held-out test faces.")
    parser.add_argument("--n-test", type=int, default=50,
                        help="Number of held-out test faces to sample.")
    parser.add_argument("--n-shape-components", type=int, default=199,
                        help="Number of PCA components used when sampling shape coefficients.")
    parser.add_argument("--decimation-ratio", type=float, default=0.10,
                        help="Quadric decimation ratio applied to the raw BFM mesh.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed used for the training set; +1 is used for the test set.")
    args = parser.parse_args()

    train_paths = generate_face_meshes(
        args.bfm_h5, args.train_out,
        n_meshes=args.n_train,
        n_shape_components=args.n_shape_components,
        decimation_ratio=args.decimation_ratio,
        seed=args.seed,
    )
    print(f"Wrote {len(train_paths)} training meshes to {args.train_out}")

    test_paths = generate_face_meshes(
        args.bfm_h5, args.test_out,
        n_meshes=args.n_test,
        n_shape_components=args.n_shape_components,
        decimation_ratio=args.decimation_ratio,
        seed=args.seed + 1,
    )
    print(f"Wrote {len(test_paths)} test meshes to {args.test_out}")


if __name__ == "__main__":
    main()
