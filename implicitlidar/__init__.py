"""Task-Driven Implicit Representations for Automated Design of LiDAR Systems.

Reference implementation of the CVPR 2026 paper by Behari, Young,
Klinghoffer, Dave, and Raskar.

Subpackages:
    core         Core methodology (target density, flow, EM sensor synthesis, constraints).
    scenes       Task-scene SDF construction (faces, robot arm, warehouse).
    eval         Ray intersection, reconstruction, metrics, baselines.
    utils        Config loading, GPU selection, I/O helpers.
    experiments  Per-experiment train / synthesize / evaluate scripts.
"""

__version__ = "0.1.0"
