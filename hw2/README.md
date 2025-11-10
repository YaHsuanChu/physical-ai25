# Running the Pipeline

Execute the scripts in sequence from the repository root. Each stage consumes artifacts saved by the previous one.

1. `python build_map.py`  
   Generates the 2D semantic map and occupancy data in `results/`. No CLI arguments are required.

2. `python rrt_planner.py --target <TARGET>`  
    <TARGET> is the desired goal.Loads the generated map, lets you pick a start location (default: mouse click), and computes an RRT path saved under `results/`.

3. `python habitat_navigator.py --target <TARGET> --path results/path_<TARGET>_habitat.npy`  
   Runs the Habitat simulator and replays the saved trajectory. Replace `<TARGET>` with the goal label you planned for in Step 2.

If you changed the target or output filenames manually, update the `--target` and `--path` values accordingly.
