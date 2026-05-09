"""Task scenes used to train and evaluate the implicit density.

Each module packages one task's geometry as a
:class:`~implicitlidar.core.SceneSDF` plus, where applicable, a
dataset-preparation helper:

* :mod:`.faces` — Basel Face Model meshes (used by face_scanning).
* :mod:`.robot_arm` — KUKA IIWA pick-and-place trajectories with
  end-effector targets and arm-link occluders (used by robot_tracking).
* :mod:`.warehouse` — procedurally generated multi-shelf rack scenes
  (used by warehouse_detection).
"""

from . import faces, robot_arm, warehouse

__all__ = ["faces", "robot_arm", "warehouse"]
