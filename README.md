## Homework 1 Guide
- This repo contains two scripts. The tasks are projecting the BEV view (`bev.py`) and 3D Reconstruction (`reconstruct.py`).

### Requirements
- Python 3.10+, Open3D, OpenCV (Python), NumPy, Matplotlib (optional)
- Place Replica dataset in `replica_v1` and organize `data_collection/<floor>` as required. Both scripts expect consistent file paths. Use `intrinsics.yaml` for camera parameters.

### Task 1 – BEV Projection (`bev.py`)
- This script projects selected BEV pixels to front-view.
- Prepare top-view image (default `bev_data/bev2.png`) and front-view image (`bev_data/front2.png`).
- Run `python bev.py` and draw polygon by left-clicking the BEV image window (right-click shows RGB). After selecting, close window.
- Adjust the camera parameters in main function if needed: `bev_orientation`, `bev_position`, `front_orientation`, `front_position`, `fov`. Output saved as `projection.png`.

### Task 2 – Reconstruction (`reconstruct.py`)
- This script uses RGB-D frames to reconstruct point cloud.
- Run `python reconstruct.py --floor first_floor` to process the dataset in `data_collection/first_floor`. Optionally use `--data-root` to provide full paths.
- Provide camera intrinsics in YAML/JSON using `--intrinsics`. The `intrinsics.yaml` is a sample.
- Key arguments: `--voxel-size`, `--icp-mode {custom, open3d}`, `--icp-iters`, `--use-fgr`, `--disable-global-registration`, `--output-dir`. Most are optional.
- Example: `python reconstruct.py --floor first_floor --intrinsics intrinsics.yaml --icp-mode custom --output-dir ./out`.
- Outputs in `<output-dir>/<floor>/` include:
  - `reconstruction_coarse_<icp_mode>.ply` with globally aligned point cloud.
  - `camera_trajectory_<icp_mode>.json` describing estimated poses.
  - `bev_scene_<icp_mode>.png` bird’s-eye projection (written when GT poses exist).
