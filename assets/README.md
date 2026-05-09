# assets

Shared resources used by the package and bundled with the repository.

```
assets/
├── README.md
├── images/                 README and project-page figures
│   └── teaser.png
├── kuka_iiwa_14/           KUKA LBR IIWA 14 model (used by robot_tracking)
└── scenes/                 Mitsuba scene templates
    └── lidar_template.xml      transient LiDAR scene (used by emitter_design)
```

## `kuka_iiwa_14/`

The KUKA LBR IIWA 14 model from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
Used both for trajectory generation (via MuJoCo) and per-link SDFs (via
`pytorch_kinematics` + `pytorch_volumetric`). Released under Apache 2.0
by DeepMind.

```bibtex
@software{menagerie2022github,
  author = {Zakka, Kevin and Tassa, Yuval and {MuJoCo Menagerie Contributors}},
  title  = {{MuJoCo Menagerie}},
  url    = {http://github.com/google-deepmind/mujoco_menagerie},
  year   = {2022}
}
```

## `scenes/lidar_template.xml`

Mitsuba 3 scene template: perspective sensor + transient HDR film + one
or more projector emitters + a shape slot for the target mesh. The
emitter-design evaluator patches the projector list, the target object
path, the resolution, and the temporal bins for every render.
