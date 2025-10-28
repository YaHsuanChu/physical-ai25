# Navigation Workflow

Run the three scripts in order; each step reuses the artifacts produced by the previous one.

1. `python build_map.py` – loads the environment and waits for you to click once to set the start position; save and close when done.
2. `python rrt_planner.py --target <TARGET>` – replace `<TARGET>` with your goal label; the planner reads the saved map and generates a path.
3. `python habitat_navigator.py --target <TARGET>` – use the same `<TARGET>` value to execute the planned trajectory inside Habitat.

Minimal input required:
- A single mouse click in `build_map.py` to mark the start.
- The textual `<TARGET>` you pass to both planner and navigator.
