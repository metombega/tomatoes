Tomato Dataset Schema

This document describes the recommended JSON structure for the tomato data collection project.

## Folder Structure

```text
dataset/
├── dataset_schema.md
└── batches/
    ├── batch_001.json
    └── tomatoes/
        ├── tomato_0001/
        │   ├── tomato_0001.json
        │   ├── raw/
        │   │   ├── depth/
        │   │   │   ├── stem_down/
        │   │   │   │   ├── deg_0/
        │   │   │   │       ├── rgb.png
        │   │   │   │       └── point_cloud.ply
        │   │   │   │   ├── deg_120/
        │   │   │   │   └── deg_240/
        │   │   │   └── stem_facing_camera/
        │   │   │       ├── deg_0/
        │   │   │       ├── deg_120/
        │   │   │       └── deg_240/
        │   │   ├── video/
        │   │   │   ├── stem_down/
        │   │   │   │   └── recording.bag
        │   │   │   └── stem_facing_camera/
        │   │   │       └── recording.bag
        │   │   └── slices/
        │   │       ├── slice_001.png
        │   │       ├── slice_002.png
        │   │       └── slice_003.png
        │   └── processed/
        │       ├── combined_point_cloud.ply
        │       └── mesh.ply
        ├── tomato_0002/
        └── tomato_0003/
```
