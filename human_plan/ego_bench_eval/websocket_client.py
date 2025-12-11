"""
WebSocket Client for Environment
"""

import os
import tqdm
import pickle
import random
import numpy as np
import cv2
import websockets.sync.client
import json
import torch
from collections import deque
from pathlib import Path

from transformers import HfArgumentParser
from human_plan.vila_train.args import (
    VLATrainingArguments, VLAModelArguments, VLADataArguments
)
from omni.isaac.lab.app import AppLauncher

seed_map = {
    "Humanoid-Pour-Balls-v0": 0, "Humanoid-Sort-Cans-v0": 1,
    "Humanoid-Insert-Cans-v0": 2, "Humanoid-Unload-Cans-v0": 3,
    "Humanoid-Insert-And-Unload-Cans-v0": 4, "Humanoid-Push-Box-v0": 5,
    "Humanoid-Open-Drawer-v0": 6, "Humanoid-Close-Drawer-v0": 7,
    "Humanoid-Open-Laptop-v0": 8, "Humanoid-Flip-Mug-v0": 9,
    "Humanoid-Stack-Can-v0": 10, "Humanoid-Stack-Can-Into-Drawer-v0": 11,
}

parser = HfArgumentParser((VLAModelArguments, VLADataArguments, VLATrainingArguments))
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--room_idx", type=int, default=None)
parser.add_argument("--table_idx", type=int, default=None)
parser.add_argument("--num_episodes", type=int, default=None)
parser.add_argument("--num_trials", type=int, default=None)
parser.add_argument("--result_saving_path", type=str, default=None)
parser.add_argument("--video_saving_path", type=str, default=None)
parser.add_argument("--save_frames", type=int, default=0)
parser.add_argument("--project_trajs", type=int, default=0)
parser.add_argument("--additional_label", type=str, default=None)
parser.add_argument("--websocket_url", type=str, default="ws://localhost:8765")

AppLauncher.add_app_launcher_args(parser)
app_launcher = AppLauncher(enable_cameras=True, device="cuda", headless=True)
simulation_app = app_launcher.app

from human_plan.ego_bench_eval.utils import (
    ik_solve, TASK_MAX_HORIZON, TASK_INIT_EPISODE
)
import gymnasium as gym
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from humanoid.tasks.base_env import BaseEnv, BaseEnvCfg


class BaseClient:
    """Base client for WebSocket communication
    
    This is the lowest-level client that receives direct qpos signals
    (arm joints + hand joints) for control.
    """
    
    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self._connected = False
        self.env = None
        self.action_tensor = None
        self.cam_intrinsics = None
        
        # Configuration variables
        self.task_name = None
        self.room_idx = None
        self.table_idx = None
        self.randomize_idxes = None
        self.curr_random_idx = None
        self.task_name_short = None
        self.load_name = None
        self.episode_list = None
        self.hist_len = None
        self.padding = 0
        self.init_poses = None
        self.save_path = None
        self.smooth_weight = None
        self.hand_smooth_weight = None
        self.future_index = None
        self.data_args = None
        self.max_horizon = None
        self.save_frames = False
        self.project_trajs = False
        self.result_saving_path = None
    
    def setup_config(self, task_args, data_args, model_args, num_episodes, video_saving_path, additional_label):
        """Setup all configuration variables"""
        self.task_name = task_args.task
        self.room_idx = task_args.room_idx
        self.table_idx = task_args.table_idx
        self.smooth_weight = model_args.smooth_weight
        self.hand_smooth_weight = model_args.hand_smooth_weight
        self.save_frames = task_args.save_frames
        self.project_trajs = task_args.project_trajs
        self.result_saving_path = task_args.result_saving_path
        self.data_args = data_args
        self.future_index = data_args.future_index
        
        # Setup randomization
        assert self.task_name in seed_map
        random.seed(seed_map[self.task_name])
        self.randomize_idxes = list(range(10000))
        random.shuffle(self.randomize_idxes)
        self.curr_random_idx = 100 + (self.room_idx * 5 + self.table_idx) * task_args.num_trials * num_episodes
        
        # Task config
        self.task_name_short = self.task_name[9:-3]
        self.load_name = self.task_name_short
        self.episode_list = TASK_INIT_EPISODE[self.task_name_short][:num_episodes]
        self.hist_len = data_args.predict_future_step * data_args.future_index
        self.cam_intrinsics = np.array([[488.6662, 0.0000, 640.0000], 
                                       [0.0000, 488.6662, 360.0000], 
                                       [0.0000, 0.0000, 1.0000]])
        
        # Paths
        self.save_path = os.path.join(video_saving_path, additional_label,
                                      f"inference_{self.smooth_weight}_{self.hand_smooth_weight}")
        Path(self.save_path).mkdir(exist_ok=True, parents=True)
        
        # Load init poses
        with open("init_poses_fixed_set_100traj.pkl", "rb") as f:
            self.init_poses = pickle.load(f)
        
        # Max horizon
        self.max_horizon = TASK_MAX_HORIZON[self.task_name]
    
    def setup_env(self):
        """Setup environment (general setup, no IK dependency)"""
        env_cfg = parse_env_cfg(self.task_name, num_envs=1)
        env_cfg.episode_length_s = 60
        env_cfg.randomize = True
        env_cfg.spawn_background = True
        env_cfg.room_idx = self.room_idx
        env_cfg.table_idx = self.table_idx
        self.env = gym.make(self.task_name, cfg=env_cfg)
        self.env.unwrapped.cfg.randomize_idx = self.randomize_idxes[self.curr_random_idx]
        self.env.reset()
        
        # Create action tensor
        self.action_tensor = torch.zeros((self.env.unwrapped.scene.num_envs, self.env.unwrapped.num_actions), device=self.env.unwrapped.robot.device)
    
    def reset(self, episode_idx, trial_idx):
        """Reset environment for a new episode
        
        Args:
            episode_idx: Episode index
            trial_idx: Trial index
            
        Returns:
            tuple: (env_results, rgb_obs, frames_output_path, out)
        """
        self.curr_random_idx += 1
        self.env.unwrapped.cfg.randomize_idx = self.randomize_idxes[self.curr_random_idx]
        env_results = self.env.reset()
        
        seq_save_path = os.path.join(self.save_path, self.task_name_short, f"room_{self.room_idx}", f"table_{self.table_idx}")
        Path(seq_save_path).mkdir(exist_ok=True, parents=True)
        output_path = os.path.join(seq_save_path, f"{self.task_name_short}_room_{self.room_idx}_table_{self.table_idx}_episode_{episode_idx}_{trial_idx}.mp4")
        
        frames_output_path = None
        if self.save_frames:
            frames_output_path = os.path.join(seq_save_path, f"{self.task_name_short}_room_{self.room_idx}_table_{self.table_idx}_episode_{episode_idx}_{trial_idx}")
            Path(frames_output_path).mkdir(exist_ok=True, parents=True)
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 720))
        
        rgb_obs = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, :]
        self.reset_inference(rgb_obs)

        return (env_results, rgb_obs, frames_output_path, out)
    
    def connect(self):
        if not self._connected:
            self.websocket = websockets.sync.client.connect(
                self.server_url, 
                max_size=50 * 1024 * 1024,
                open_timeout=10
            )
            self._connected = True
    
    def disconnect(self):
        if self.websocket and self._connected:
            self.websocket.close()
            self._connected = False
    
    def infer(self, rgb_obs, env_results, task_name, config=None):
        """Call model inference via WebSocket
            
        Returns:
            dict with keys:
                - action: dict with action signals (format depends on policy type)
                  For BaseClient (qpos-based): left_arm_qpos, right_arm_qpos, left_hand_qpos, right_hand_qpos
                  For EEFClient (eef-based): left_ee_pose, right_ee_pose, left_qpos_multi_step, right_qpos_multi_step
                - vis: dict with visualization data
        """
        if not self._connected:
            self.connect()
        
        request = {
            "type": "inference",
            "rgb_obs": rgb_obs.tolist(),  # Convert numpy array to list for JSON serialization
            "proprio": {
                "left_finger_tip": env_results["left_finger_tip_pos"].cpu().numpy().tolist(),
                "right_finger_tip": env_results["right_finger_tip_pos"].cpu().numpy().tolist(),
                "left_ee_pose": env_results["left_ee_pose"].cpu().numpy().tolist(),
                "right_ee_pose": env_results["right_ee_pose"].cpu().numpy().tolist(),
                "qpos": env_results["qpos"].cpu().numpy().tolist(),
                "cam_intrinsics": self.cam_intrinsics.tolist(),
            },
            "task_name": task_name,
        }
        if config:
            request["config"] = config
        
        self.websocket.send(json.dumps(request))
        response = json.loads(self.websocket.recv())
        return {
            "action": response.get("action", {}),
            "vis": response.get("vis", {})
        }
    
    def reset_inference(self, rgb_obs):
        """Reset inference history on server side"""
        if not self._connected:
            self.connect()
        
        request = {"type": "reset"}
        request["rgb_obs"] = rgb_obs.tolist()
        self.websocket.send(json.dumps(request))
        response = json.loads(self.websocket.recv())
        return response.get("type") == "reset_ack"
    
    def get_action(self, left_arm_qpos, right_arm_qpos, left_hand_qpos, right_hand_qpos):
        """Get action tensor from arm and hand joint positions
        
        Args:
            left_arm_qpos: Left arm joint positions
            right_arm_qpos: Right arm joint positions
            left_hand_qpos: Left hand joint positions
            right_hand_qpos: Right hand joint positions
            
        Returns:
            torch.Tensor: Action tensor ready for env.step()
        """
        if self.env is None or self.action_tensor is None:
            raise ValueError("BaseClient requires env and action_tensor for get_action")
        
        self.action_tensor[:, :] = 0
        self.action_tensor[:, self.env.unwrapped.cfg.left_arm_cfg.joint_ids] = torch.FloatTensor(left_arm_qpos).to(self.action_tensor.device)
        self.action_tensor[:, self.env.unwrapped.cfg.right_arm_cfg.joint_ids] = torch.FloatTensor(right_arm_qpos).to(self.action_tensor.device)
        self.action_tensor[:, self.env.unwrapped.cfg.left_hand_cfg.joint_ids] = torch.FloatTensor(left_hand_qpos).to(self.action_tensor.device)
        self.action_tensor[:, self.env.unwrapped.cfg.right_hand_cfg.joint_ids] = torch.FloatTensor(right_hand_qpos).to(self.action_tensor.device)
        return self.action_tensor
    
    def _postprocess_action(self, action_left_arm_qpos, action_right_arm_qpos, 
                        action_left_hand, action_right_hand):
        return self.get_action(action_left_arm_qpos, action_right_arm_qpos, 
                              action_left_hand, action_right_hand)
    
    def _extract_action_from_response(self, response):
        """Extract action values from response
        
        BaseClient expects qpos format: left_arm_qpos, right_arm_qpos, left_hand_qpos, right_hand_qpos
        Subclasses can override to extract different formats (e.g., EEF poses)
        
        Args:
            response: Raw response from infer()
            
        Returns:
            tuple: (left_arm_qpos, right_arm_qpos, left_hand_qpos, right_hand_qpos)
        """
        action_data = response
        return (
            np.array(action_data["left_arm_qpos"]),
            np.array(action_data["right_arm_qpos"]),
            np.array(action_data["left_hand_qpos"]),
            np.array(action_data["right_hand_qpos"])
        )
    
    def run_inference_loop(self, env_results, rgb_obs, frames_output_path, out):
        """Run main inference loop
        
        Args:
            env_results: Initial environment results
            rgb_obs: Initial RGB observation
            frames_output_path: Path to save frames
            out: Video writer
            
        Returns:
            tuple: (result, env_results)
        """
        result = False
        for i in tqdm.tqdm(range(self.max_horizon)):
            rgb_obs = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, :]
            
            response = self.infer(
                rgb_obs, env_results[0], self.task_name,
                config={"input_hand_dof": self.data_args.input_hand_dof}
            )
            left_arm_action, right_arm_action, left_hand_action, right_hand_action = self._extract_action_from_response(response)
            
            action = self._postprocess_action(
                left_arm_action, right_arm_action, 
                left_hand_action, right_hand_action
            )
            env_results = self.env.step(action)
            if env_results[0]["success"].sum().item() == 1:
                result = True
                break
            
            result_img_3d = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, ::-1].copy()
            if self.project_trajs == 1:
                from human_plan.utils.visualization import project_points
                pred_3d = np.array(response["vis"]["pred_3d"])
                proj_2d = project_points(pred_3d, self.cam_intrinsics).reshape(-1, 2, 2)
                for fi in range(proj_2d.shape[0]-1):
                    for j in range(2):
                        result_img_3d = cv2.circle(result_img_3d, (int(proj_2d[fi, j, 0]), int(proj_2d[fi, j, 1])), 5, (0, 255, 0), thickness=-1)
                        if fi < proj_2d.shape[0] - 1:
                            result_img_3d = cv2.line(result_img_3d, (int(proj_2d[fi, j, 0]), int(proj_2d[fi, j, 1])),
                                                   (int(proj_2d[fi + 1, j, 0]), int(proj_2d[fi + 1, j, 1])), (0, 255, 0), thickness=2)
            out.write(result_img_3d)
            if frames_output_path is not None:
                cv2.imwrite(os.path.join(frames_output_path, f"{i}.jpg"), result_img_3d)
        
        return result, env_results
    
    def save_results(self, episode_idx, trial_idx, result, env_results):
        """Save episode results to file"""
        with open(self.result_saving_path, "a") as f:
            f.write(f"Task: {self.task_name_short}, Room Idx: {self.room_idx}, Table Idx: {self.table_idx}, Episode Label: {episode_idx[0]}, Trial Label: {trial_idx}, Result: {result} \n")
            subtask_string = ""
            for key in env_results[0].keys():
                if "success" in key:
                    subtask_string += f"{key}: {env_results[0][key].sum().item()} "
            subtask_string += "\n"
            f.write(subtask_string)


class EEFClient(BaseClient):
    """EEF client that receives eef_pose and hand_qpos, 
    converts them to qpos via IK, then passes to BaseClient.
    
    This is the upper-level client for models that output EEF poses.
    """
    
    def __init__(self, server_url: str = "ws://localhost:8765"):
        super().__init__(server_url)
        self.left_ik_controller = None
        self.right_ik_controller = None
        self.left_ik_commands_world = None
        self.right_ik_commands_world = None
        self.left_ik_commands_robot = None
        self.right_ik_commands_robot = None
    
    def setup_env(self):
        """Setup environment (calls parent, then adds IK setup)"""
        super().setup_env()
        self._setup_ik()
    
    def _setup_ik(self):
        """Setup IK controllers and command buffers (IK-specific setup)"""
        command_type = "pose"
        left_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="dls")
        self.left_ik_controller = DifferentialIKController(left_ik_cfg, num_envs=self.env.unwrapped.scene.num_envs, device=self.env.unwrapped.sim.device)
        right_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="pinv")
        self.right_ik_controller = DifferentialIKController(right_ik_cfg, num_envs=self.env.unwrapped.scene.num_envs, device=self.env.unwrapped.robot.device)
        
        self.left_ik_commands_world = torch.zeros(self.env.unwrapped.scene.num_envs, self.left_ik_controller.action_dim, device=self.env.unwrapped.robot.device)
        self.left_ik_commands_robot = torch.zeros(self.env.unwrapped.scene.num_envs, self.left_ik_controller.action_dim, device=self.env.unwrapped.robot.device)
        self.right_ik_commands_world = torch.zeros(self.env.unwrapped.scene.num_envs, self.right_ik_controller.action_dim, device=self.env.unwrapped.robot.device)
        self.right_ik_commands_robot = torch.zeros(self.env.unwrapped.scene.num_envs, self.right_ik_controller.action_dim, device=self.env.unwrapped.robot.device)
    
    def infer(self, rgb_obs, env_results, task_name, config=None):
        """Call model inference and return EEF poses + hand qpos"""
        response = super().infer(rgb_obs, env_results, task_name, config)
        
        return {
            "action": response["action"],
            "vis": response["vis"]
        }
    
    def reset(self, episode_idx, trial_idx, num_warmup_steps=100):
        """Reset environment and run warmup using EEF poses via IK
        
        Args:
            episode_idx: Episode index
            trial_idx: Trial index
            num_warmup_steps: Number of warmup steps (default 100)
            
        Returns:
            tuple: (env_results, rgb_obs, frames_output_path, out)
        """
        env_results, rgb_obs, frames_output_path, out = super().reset(episode_idx, trial_idx)
        
        seq_name = episode_idx[0]
        left_dof = self.init_poses[self.load_name][seq_name][self.padding]["left_dof"]
        right_dof = self.init_poses[self.load_name][seq_name][self.padding]["right_dof"]
        left_ee_pose_traj_gt = self.init_poses[self.load_name][seq_name][self.padding]["left_ee"]
        right_ee_pose_traj_gt = self.init_poses[self.load_name][seq_name][self.padding]["right_ee"]
        
        self.left_ik_controller.reset()
        self.right_ik_controller.reset()
        
        for idx in range(num_warmup_steps):
            action = self.get_action_from_eef(
                left_ee_pose_traj_gt,
                right_ee_pose_traj_gt,
                left_dof,
                right_dof
            )
            env_results = self.env.step(action)
        
        rgb_obs = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, :]
        self.reset_inference(rgb_obs)
        
        return (env_results, rgb_obs, frames_output_path, out)
    
    def get_action_from_eef(self, left_ee_pose, right_ee_pose, left_hand_qpos, right_hand_qpos):
        """Convert EEF poses and hand qpos to full action via IK
        
        First solves IK to get arm qpos, then calls BaseClient.get_action()
        
        Args:
            left_ee_pose: (7,) array - left end-effector pose
            right_ee_pose: (7,) array - right end-effector pose
            left_hand_qpos: (12,) array - left hand joint positions
            right_hand_qpos: (12,) array - right hand joint positions
            
        Returns:
            torch.Tensor: action tensor ready for env.step()
        """
        left_arm_qpos, right_arm_qpos = ik_solve(
            self.env,
            self.left_ik_controller,
            self.right_ik_controller,
            self.left_ik_commands_world,
            self.right_ik_commands_world,
            self.left_ik_commands_robot,
            self.right_ik_commands_robot,
            left_ee_pose,
            right_ee_pose,
        )
        
        # Call base get_action with computed arm qpos
        return self.get_action(left_arm_qpos, right_arm_qpos, left_hand_qpos, right_hand_qpos)
    
    def _extract_action_from_response(self, response):
        """Extract EEF poses and hand qpos from response (override for EEF-based policies)
        
        Returns EEF poses as "arm" values and hand qpos as "hand" values for compatibility
        with run_inference_loop's smoothing logic.
        """
        action_data = response["action"]
        return (
            np.array(action_data["left_ee_pose"]),  # as left_arm
            np.array(action_data["right_ee_pose"]),  # as right_arm
            np.array(action_data["left_qpos_multi_step"]),  # as left_hand
            np.array(action_data["right_qpos_multi_step"])  # as right_hand
        )
    
    def _postprocess_action(self, action_left_ee, action_right_ee, 
                        action_left_hand, action_right_hand):
        """Compute action from smoothed EEF poses via IK"""
        return self.get_action_from_eef(
            action_left_ee,
            action_right_ee,
            action_left_hand,
            action_right_hand
        )


def main():
    model_args, data_args, training_args, task_args = parser.parse_args_into_dataclasses()
    
    # Create and setup client
    client = EEFClient(task_args.websocket_url)
    client.setup_config(task_args, data_args, model_args, task_args.num_episodes, 
                       task_args.video_saving_path, task_args.additional_label)
    client.setup_env()
    client.connect()
    
    for episode_idx in client.episode_list:
        for trial_idx in range(task_args.num_trials):
            (env_results, rgb_obs, frames_output_path, out) = client.reset(
                episode_idx, trial_idx
            )
            
            result, env_results = client.run_inference_loop(
                env_results, rgb_obs, frames_output_path, out
            )
                
            client.save_results(episode_idx, trial_idx, result, env_results)
            out.release()
        
    client.disconnect()
    client.env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
