# scripts/src/run.py
from web_ui.app import WebUI
import argparse
from queue import Queue
import os
import time
import math
import gzip, json
from datetime import datetime
from omni.isaac.lab.app import AppLauncher
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from log_manager import LogManager
import torch
import cv2
import numpy as np
from typing import Dict, Any
import cli_args
import torch.nn.functional as F
from PIL import Image

from vlnbert.vlnbert_init import get_vlnbert_models
from transformers import BertTokenizer
import glob
import os
import torch.nn.functional as F
from utils import pad_instr_tokens  # Assuming utils.py is in src/

# Add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to collect data from the matterport dataset.")
parser.add_argument("--episode_index", default=0, type=int, help="Episode index.")
parser.add_argument("--num_episodes", default=1, type=int, help="Number of episodes to run.")
parser.add_argument("--task", type=str, default="go2_matterport", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2500, help="Length of the recorded video (in steps).")
parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--arm_fixed", action="store_true", default=False, help="Fix the robot's arms.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")
parser.add_argument("--action_repeat", default=20, type=int, help="Number of simulation steps to repeat each action.")
parser.add_argument("--vlnbert_model_path", type=str, required=True, help="Path to the VLNBert model checkpoint.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--draw_pointcloud", action="store_true", default=1, help="Draw pointcloud.")
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.core.utils.prims as prim_utils
import torch
from omni.isaac.core.objects import VisualCuboid
import gymnasium as gym
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.markers.config import CUBOID_MARKER_CFG
from omni.isaac.lab.markers import VisualizationMarkers
import omni.isaac.lab.utils.math as math_utils
from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
from omni.isaac.vlnce.config import *
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper, VLNEnvWrapper

def find_latest_model(checkpoints_dir, experiment_name):
    """
    Find the latest best model or checkpoint from the most recent datetime subfolder.

    Args:
        checkpoints_dir (str): Base directory for checkpoints (e.g., "checkpoints").
        experiment_name (str): Experiment name (e.g., "navigation_PREVALENT").

    Returns:
        str: Path to the latest model file.

    Raises:
        FileNotFoundError: If no datetime directories or checkpoint files are found.
    """
    experiment_path = os.path.join(checkpoints_dir, experiment_name)
    datetime_dirs = sorted(glob.glob(os.path.join(experiment_path, "*/")))
    if not datetime_dirs:
        raise FileNotFoundError(f"No datetime directories found in {experiment_path}.")
    latest_datetime_dir = datetime_dirs[-1]
    best_model_path = os.path.join(latest_datetime_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        return best_model_path
    else:
        checkpoint_files = sorted(
            glob.glob(os.path.join(latest_datetime_dir, "checkpoint_*.pt")),
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {latest_datetime_dir}.")
        return checkpoint_files[-1]

class Planner:
    def __init__(self, env, env_cfg, args_cli, simulation_app):
        self.env = env
        self.env_cfg = env_cfg
        self.args_cli = args_cli
        self.simulation_app = simulation_app
        self.k_steps = args_cli.action_repeat
        self.action_queue = Queue()
        self.web_ui = WebUI(action_queue=self.action_queue)
        self.log_manager = LogManager(log_dir="logs/experiments", web_ui=self.web_ui)
        self.step_counter = 0
        self.max_steps = 2000
        self.action_map = {
            "Move forward": (0.6, 0.0, 0.0),
            "Turn left": (0.0, 0.0, 0.5),
            "Turn right": (0.0, 0.0, -0.5)
        }
        self.action_to_idx = {"Move forward": 0, "Turn left": 1, "Turn right": 2}
        self.idx_to_action = {0: "Move forward", 1: "Turn left", 2: "Turn right"}
        self.action_velocities = [
            (0.6, 0.0, 0.0),  # Move forward
            (0.0, 0.0, 0.5),  # Turn left
            (0.0, 0.0, -0.5)  # Turn right
        ]

        # Expert path visualization (unchanged)
        # self.marker_cfg = CUBOID_MARKER_CFG.copy()
        # self.marker_cfg.prim_path = "/Visuals/Command/pos_goal_command"
        # self.marker_cfg.markers["cuboid"].scale = (0.1, 0.1, 0.1)
        # self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.env.unwrapped.device).repeat(1, 1)
        # for i in range(self.env_cfg.expert_path_length):
        #     expert_point_visualizer = VisualizationMarkers(self.marker_cfg)
        #     expert_point_visualizer.set_visibility(True)
        #     point = np.array(self.env_cfg.expert_path[i]).reshape(1, 3)
        #     default_scale = expert_point_visualizer.cfg.markers["cuboid"].scale
        #     larger_scale = 2.0 * torch.tensor(default_scale, device=self.env.unwrapped.device).repeat(1, 1)
        #     expert_point_visualizer.visualize(point, self.identity_quat, larger_scale)

        # Initialize VLNBert model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vln_bert = get_vlnbert_models().to(self.env.unwrapped.device)
        self.vln_bert.load_state_dict(torch.load(args_cli.vlnbert_model_path, map_location=self.env.unwrapped.device))
        self.vln_bert.eval()
        print(f"Loaded VLNBert model from {args_cli.vlnbert_model_path}")

        # Define RGB transformation for VLNBert
        self.rgb_transform = Compose([
            ToPILImage(),
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_instruction(self, instruction):
        """Tokenize and process the instruction to get initial language output."""
        tokens = self.tokenizer.tokenize(instruction)
        padded_tokens, _ = pad_instr_tokens(tokens, maxlength=80)  # Match maxInput from param.py
        instr_encoding = self.tokenizer.convert_tokens_to_ids(padded_tokens)
        instr_encoding = torch.tensor([instr_encoding], device=self.env.unwrapped.device)
        lang_mask = (instr_encoding != self.tokenizer.pad_token_id).float()
        with torch.no_grad():
            _, lang_output = self.vln_bert("language", instr_encoding, lang_mask=lang_mask)
        return lang_output, lang_mask

    def process_depth(self, depth_map):
        """Process depth map to the format expected by VLNBert."""
        depth = np.where(np.isfinite(depth_map), depth_map, 1.94)
        depth = torch.from_numpy(depth).float().to(self.env.unwrapped.device)
        depth = (depth - 1.94) / 1.43  # Standardize using stats from compute_depth_stats.py
        depth = depth.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        depth = F.interpolate(depth, size=(224, 224), mode='bilinear', align_corners=False)
        return depth  # Shape: (1, 1, 224, 224)

    def run_episode(self, episode_idx: int) -> None:
        obs, infos = self.env.reset()
        # Compute initial distance to goal
        initial_robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        initial_distance = np.linalg.norm(initial_robot_pos[:2] - self.env_cfg.goals[0]['position'][:2])
        print("initial_distance: ", initial_distance)
        current_instruction = self.env_cfg.instruction_text
        self.log_manager.start_episode(episode_idx, current_instruction)        
        # Training data buffers (unchanged)
        instructions = []
        rgbs = []
        depths = []
        actions = []
        
        self.step_counter = 0
        self.robot_path = []
        success = False
        done = False
        
        print("Started episode. Visualization and action input at http://localhost:5000")

        # Inference mode with VLNBert
        lang_output, lang_mask = self.process_instruction(current_instruction)
        robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        rgb_image_np = infos['observations']['camera_obs'][0, :, :, :3].clone().detach().cpu().numpy().astype(np.uint8)
        depth_map = infos['observations']['depth_obs'][0].cpu().numpy().squeeze()
        while self.step_counter < self.max_steps and not done:
            rgb_t = self.rgb_transform(rgb_image_np).unsqueeze(0).to(self.env.unwrapped.device)  # (1, 3, 224, 224)
            depth_t = self.process_depth(depth_map)  # (1, 1, 224, 224)
            vis_mask = torch.ones(1, 1, device=self.env.unwrapped.device)
            goal_distance = np.linalg.norm(robot_pos[:2] - self.env_cfg.goals[0]['position'][:2])
            with torch.no_grad():
                pooled_output, action_logits, next_lang_output = self.vln_bert(
                    "visual", lang_output, lang_mask=lang_mask, vis_mask=vis_mask, rgb=rgb_t, depth=depth_t
                )
            action_idx = torch.argmax(action_logits, dim=1).item()
            action = self.idx_to_action[action_idx]
            vel_command = torch.tensor(self.action_velocities[action_idx], device=self.env.unwrapped.device)
            for _ in range(self.k_steps):
                if self.step_counter >= self.max_steps:
                    break
                obs, _, done, infos = self.env.step(vel_command.clone())
                self.step_counter += 1
                robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                rgb_image_np = infos['observations']['camera_obs'][0, :, :, :3].clone().detach().cpu().numpy().astype(np.uint8)
                depth_map = infos['observations']['depth_obs'][0].cpu().numpy().squeeze()
                bgr_image_np = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR)
                goal_distance = np.linalg.norm(robot_pos[:2] - self.env_cfg.goals[0]['position'][:2])
                self.robot_path.append((robot_pos[0], robot_pos[1]))
                # Update UI with predicted action
                resized_rgb = cv2.resize(rgb_image_np, (256, 256))
                resized_depth = cv2.resize(depth_map, (256, 256), interpolation=cv2.INTER_NEAREST)
                self.log_manager.log_step(
                    self.step_counter, self.robot_path, bgr_image_np, depth_map, action,
                    goal_distance, self.max_steps - self.step_counter
                )
                self.web_ui.current_data.update({
                    'vlm_prompt': current_instruction,
                    'vlm_response': action,
                    'waiting_for_action': False
                })
                self.web_ui.update_data(self.web_ui.current_data)
                # Check if goal is reached
                if goal_distance < 1.0:
                    success = True
                    done = True
                    break
            lang_output = next_lang_output  # Update language output for recurrence

        final_distance = goal_distance
        print("initial_distance: ", initial_distance)
        print("final_distance: ", final_distance)
        progress = (initial_distance - final_distance + 1) / initial_distance if not success else 1 #even reaching 1 counts as success, therefore + 1 by default for the progress
        status = "Success: Reached the goal within 1 meter" if success else "Failure: Did not reach the goal" if done else "Failure: Reached maximum steps"
        self.log_manager.set_status(status)
        print(status)
        print(f"Progress: {progress:.2f}")

    def start_loop(self):
        """Start running multiple episodes."""
        self.run_episode(self.args_cli.episode_index)
        time.sleep(3)

if __name__ == "__main__":
    # Environment Configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    vel_command = torch.tensor([0, 0, 0])
    episode_idx = args_cli.episode_index 
    dataset_file_name = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")
    with gzip.open(dataset_file_name, "rt") as f:
        deserialized = json.loads(f.read())
        episode = deserialized["episodes"][episode_idx]
        if "go2" in args_cli.task:
            env_cfg.scene.robot.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+0.4)
        elif "h1" in args_cli.task:
            env_cfg.scene.robot.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+1.0)
        else:
            env_cfg.scene.robot.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+0.5)
        env_cfg.scene.disk_1.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+2.5)
        env_cfg.scene.disk_2.init_state.pos = (episode["reference_path"][-1][0], episode["reference_path"][-1][1], episode["reference_path"][-1][2]+2.5)
        wxyz_rot = episode["start_rotation"]
        init_rot = wxyz_rot
        env_cfg.scene.robot.init_state.rot = (init_rot[0], init_rot[1], init_rot[2], init_rot[3])
        env_cfg.goals = episode["goals"]
        env_cfg.episode_id = episode["episode_id"]
        env_cfg.scene_id = episode["scene_id"].split('/')[1]
        env_cfg.traj_id = episode["trajectory_id"]
        env_cfg.instruction_text = episode["instruction"]["instruction_text"]
        env_cfg.instruction_tokens = episode["instruction"]["instruction_tokens"]
        env_cfg.reference_path = np.array(episode["reference_path"])
        expert_locations = np.array(episode["gt_locations"])
        env_cfg.expert_path = expert_locations
        env_cfg.expert_path_length = len(env_cfg.expert_path)
        env_cfg.expert_time = np.arange(env_cfg.expert_path_length) * 1.0
    udf_file = os.path.join(ASSETS_DIR, f"matterport_usd/{env_cfg.scene_id}/{env_cfg.scene_id}.usd")
    if os.path.exists(udf_file):
        env_cfg.scene.terrain.obj_filepath = udf_file
    else:
        raise ValueError(f"No USD file found in scene directory: {udf_file}")  

    print("scene_id: ", env_cfg.scene_id)
    print("robot_init_pos: ", env_cfg.scene.robot.init_state.pos)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # Low Level Policy
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)
    
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.join(os.path.dirname(__file__), "logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    low_level_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
    env = VLNEnvWrapper(env, low_level_policy, args_cli.task, episode, high_level_obs_key="camera_obs",
                        measure_names=all_measures)

    planner = Planner(env, env_cfg, args_cli, simulation_app)
    planner.start_loop()
    simulation_app.close()
    print("closed!!!")