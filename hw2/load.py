"""
Utilities for configuring and interactively exploring the Habitat simulator.

This module exposes helpers that the homework scripts can import without
triggering side-effects, and still retains the keyboard-driven demo when run
directly.  The discrete action amounts are configurable so that downstream
navigation code can control the agent with predictable step sizes.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import cv2
import habitat_sim
import numpy as np
from habitat_sim.agent import ActuationSpec
from habitat_sim.utils.common import d3_40_colors_rgb
from PIL import Image

# Default scene path shipped with the homework starter.
_DEFAULT_SCENE = "replica_v1/apartment_0/habitat/mesh_semantic.ply"

DEFAULT_SIM_SETTINGS: Dict[str, object] = {
    "scene": _DEFAULT_SCENE,
    "default_agent": 0,
    "sensor_height": 1.5,
    "width": 512,
    "height": 512,
    "sensor_pitch": 0.0,
}

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH_KEY = "f"


def transform_rgb_bgr(image: np.ndarray) -> np.ndarray:
    """Convert Habitat's RGB (or RGBA) image to BGR for OpenCV display/write."""
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image[..., ::-1]


def transform_depth(image: np.ndarray) -> np.ndarray:
    """Normalize depth image to an 8-bit range for visualization."""
    depth_img = (image / 10.0 * 255).clip(0, 255).astype(np.uint8)
    return depth_img


def transform_semantic(semantic_obs: np.ndarray) -> np.ndarray:
    """Render semantic observation with the Habitat color palette."""
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    return cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)


def make_simple_cfg(
    settings: Dict[str, object],
    forward_step_m: float = 0.25,
    turn_step_deg: float = 10.0,
) -> habitat_sim.Configuration:
    """
    Build a Habitat simulator + agent configuration based on the provided settings.

    Discrete actions are wired to the requested magnitudes so downstream code knows
    exactly how far the agent will move or rotate per step.
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [
        rgb_sensor_spec,
        depth_sensor_spec,
        semantic_sensor_spec,
    ]

    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", ActuationSpec(amount=forward_step_m)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", ActuationSpec(amount=turn_step_deg)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", ActuationSpec(amount=turn_step_deg)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def make_env(
    override_settings: Optional[Dict[str, object]] = None,
    forward_step_m: float = 0.25,
    turn_step_deg: float = 10.0,
) -> Tuple[habitat_sim.Simulator, habitat_sim.agent.Agent, Dict[str, object]]:
    """
    Instantiate the simulator and default agent with the provided action magnitudes.

    Returns the simulator, the default agent handle, and the resolved settings dict.
    """
    sim_settings = copy.deepcopy(DEFAULT_SIM_SETTINGS)
    if override_settings:
        sim_settings.update(override_settings)

    cfg = make_simple_cfg(sim_settings, forward_step_m, turn_step_deg)
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(sim_settings["default_agent"])
    return sim, agent, sim_settings


def _interactive_demo() -> None:
    """Keyboard-controlled demo preserved from the original starter script."""
    sim, agent, sim_settings = make_env()
    cfg = sim.config
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([0.0, 0.0, 0.0])
    agent.set_state(agent_state)

    print("Discrete action space:", action_names)
    print("#############################")
    print("use keyboard to control the agent")
    print(" w for go forward")
    print(" a for turn left")
    print(" d for turn right")
    print(" f for finish and quit the program")
    print("#############################")

    def navigate_and_see(action: str = "") -> None:
        if action not in action_names:
            print("INVALID ACTION")
            return
        observations = sim.step(action)
        cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
        cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
        cv2.imshow("semantic", transform_semantic(observations["semantic_sensor"]))
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states["color_sensor"]
        print("camera pose: x y z rw rx ry rz")
        print(
            sensor_state.position[0],
            sensor_state.position[1],
            sensor_state.position[2],
            sensor_state.rotation.w,
            sensor_state.rotation.x,
            sensor_state.rotation.y,
            sensor_state.rotation.z,
        )

    try:
        navigate_and_see("move_forward")
        while True:
            keystroke = cv2.waitKey(0)
            if keystroke == ord(FORWARD_KEY):
                navigate_and_see("move_forward")
                print("action: FORWARD")
            elif keystroke == ord(LEFT_KEY):
                navigate_and_see("turn_left")
                print("action: LEFT")
            elif keystroke == ord(RIGHT_KEY):
                navigate_and_see("turn_right")
                print("action: RIGHT")
            elif keystroke == ord(FINISH_KEY):
                print("action: FINISH")
                break
            else:
                print("INVALID KEY")
    finally:
        cv2.destroyAllWindows()
        sim.close()


if __name__ == "__main__":
    _interactive_demo()
