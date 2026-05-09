"""real_world — Real-world SPAD validation.

Two scripts here:

* :mod:`.process_measurement` — load a real single-photon transient
  measurement (HDF5 ``.mat``), peak-find each pixel's time-of-flight, extract
  a region of interest, and emit a 3D point cloud.
* :mod:`.reconstruct` — given a point cloud and a set of synthesized sensor
  rays, find each ray's closest cloud point and triangulate the resulting
  hits into a surface mesh.
"""
