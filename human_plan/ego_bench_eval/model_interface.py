import base64
import numpy as np
import cv2
import torch
from collections import deque
from human_plan.vila_eval.utils.load_model import load_model_eval
from human_plan.ego_bench_eval.utils import (
    process_proprio_input,
    process_input,
    ik_eval_single_step,
    get_language_instruction,
    smooth_action,
    repeat_action
)


class ModelInterface:
    """Base interface for model inference and data processing"""
    
    @staticmethod
    def encode_image(image: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer.tobytes()).decode('utf-8')
    
    @staticmethod
    def decode_image(image_data: str) -> np.ndarray:
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    @staticmethod
    def tensor_to_numpy(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, np.ndarray):
            return obj
        return np.array(obj)
    
    @classmethod
    def setup(cls, *args, **kwargs):
        """Setup and create model interface instance
        
        This method should be overridden by subclasses to provide
        model-specific initialization logic.
        """
        raise NotImplementedError("Subclasses must implement setup()")
    
    def infer(self, rgb_obs_hist, left_finger_tip, right_finger_tip, 
              left_ee_pose, right_ee_pose, qpos, cam_intrinsics,
              language_instruction=None, task_name=None, 
              input_hand_dof=None, raw_width=1280, raw_height=720):
        """Perform model inference and return formatted response
        
        Args:
            rgb_obs_hist: List of RGB images (numpy arrays)
            left_finger_tip: Left finger tip position
            right_finger_tip: Right finger tip position
            left_ee_pose: Left end-effector pose
            right_ee_pose: Right end-effector pose
            qpos: Joint positions
            cam_intrinsics: Camera intrinsics matrix
            language_instruction: Optional language instruction
            task_name: Optional task name
            input_hand_dof: Optional input hand DOF flag
            raw_width: Image width
            raw_height: Image height
            
        Returns:
            dict: Formatted response with 'action' and 'vis' keys
        """
        raise NotImplementedError("Subclasses must implement infer()")


class EgoVLAInterface(ModelInterface):
    """EgoVLA model inference implementation"""
    
    def __init__(self, model, tokenizer, model_args, data_args, training_args):
        self.model = model.to("cuda")
        self.model.eval()
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.data_args.sep_query_token = self.model_args.sep_query_token
        self.smooth_weight = model_args.smooth_weight
        self.hand_smooth_weight = model_args.hand_smooth_weight
        self.hist_len = data_args.predict_future_step * data_args.future_index
        self.future_index = data_args.future_index
        
        self._init_history_buffers()
    
    def _init_history_buffers(self):
        """Initialize history buffers for images and actions"""
        self.rgb_obs_hist = deque(maxlen=120)  # Image history
        self.action_hist_left_arm = deque(maxlen=self.hist_len)
        self.action_hist_right_arm = deque(maxlen=self.hist_len)
        self.action_hist_left_hand = deque(maxlen=self.hist_len)
        self.action_hist_right_hand = deque(maxlen=self.hist_len)
    
    def reset_history(self, rgb_obs=None):
        """Reset all history buffers and optionally initialize with an observation
        
        Args:
            rgb_obs: Optional initial RGB observation to add to history after reset
        """
        self._init_history_buffers()
        rgb_obs_resized = cv2.resize(rgb_obs, (384, 384))
        self.rgb_obs_hist.append(rgb_obs_resized)
    
    @classmethod
    def setup(cls, model_path, model_args, data_args, training_args):
        """Create and initialize EgoVLAInterface"""
        model_args.model_name_or_path = model_path
        model, tokenizer, model_args, data_args, training_args = load_model_eval(
            model_args, data_args, training_args
        )
        return cls(model, tokenizer, model_args, data_args, training_args)
    
    def infer(self, rgb_obs, left_finger_tip, right_finger_tip, 
              left_ee_pose, right_ee_pose, qpos, cam_intrinsics,
              language_instruction=None, task_name=None, 
              input_hand_dof=None, raw_width=1280, raw_height=720):
        """Perform model inference with internal history management and action smoothing
        
        Args:
            rgb_obs: Single RGB image (numpy array), will be resized and added to history
            left_finger_tip: Left finger tip position
            right_finger_tip: Right finger tip position
            left_ee_pose: Left end-effector pose
            right_ee_pose: Right end-effector pose
            qpos: Joint positions
            cam_intrinsics: Camera intrinsics matrix
            language_instruction: Optional language instruction
            task_name: Optional task name
            input_hand_dof: Optional input hand DOF flag
            raw_width: Image width
            raw_height: Image height
            
        Returns:
            dict: Formatted response with smoothed 'action' and 'vis' keys
        """
        # Resize and add to image history (before inference)
        rgb_obs_resized = cv2.resize(rgb_obs, (384, 384))
        self.rgb_obs_hist.append(rgb_obs_resized)
        
        # Process proprioceptive input
        input_hand_dof_val = input_hand_dof if input_hand_dof is not None else self.data_args.input_hand_dof
        proprio_input, raw_proprio_inputs = process_proprio_input(
            left_finger_tip, right_finger_tip, left_ee_pose, right_ee_pose,
            qpos, cam_intrinsics, input_hand_dof=input_hand_dof_val
        )
        
        # Get language instruction
        if language_instruction is None:
            if task_name is None:
                raise ValueError("Either language_instruction or task_name must be provided")
            language_instruction = get_language_instruction(task_name)
        
        # Process input
        raw_data_dict = process_input(
            list(self.rgb_obs_hist), proprio_input.to("cuda"), language_instruction,
            self.data_args, self.model_args, self.tokenizer
        )
        raw_data_dict.update(raw_proprio_inputs)
        raw_data_dict["raw_width"] = raw_width
        raw_data_dict["raw_height"] = raw_height

        # Model inference
        with torch.inference_mode():
            action_dict = ik_eval_single_step(raw_data_dict, self.model, self.tokenizer)
        
        # Extract raw actions
        left_ee_pose = self.tensor_to_numpy(action_dict["left_ee_pose"])
        right_ee_pose = self.tensor_to_numpy(action_dict["right_ee_pose"])
        left_qpos_multi_step = self.tensor_to_numpy(action_dict["left_qpos_multi_step"])
        right_qpos_multi_step = self.tensor_to_numpy(action_dict["right_qpos_multi_step"])
        
        # Apply repeat_action and add to history
        self.action_hist_left_arm.append(repeat_action(left_ee_pose, self.future_index))
        self.action_hist_right_arm.append(repeat_action(right_ee_pose, self.future_index))
        self.action_hist_left_hand.append(repeat_action(left_qpos_multi_step, self.future_index))
        self.action_hist_right_hand.append(repeat_action(right_qpos_multi_step, self.future_index))
        
        # Smooth actions
        smoothed_left_ee_pose = smooth_action(self.hist_len, self.smooth_weight, self.action_hist_left_arm)
        smoothed_right_ee_pose = smooth_action(self.hist_len, self.smooth_weight, self.action_hist_right_arm)
        smoothed_left_qpos = smooth_action(self.hist_len, self.hand_smooth_weight, self.action_hist_left_hand)
        smoothed_right_qpos = smooth_action(self.hist_len, self.hand_smooth_weight, self.action_hist_right_hand)
        
        return {
            "action": {
                "left_ee_pose": smoothed_left_ee_pose.tolist(),
                "right_ee_pose": smoothed_right_ee_pose.tolist(),
                "left_qpos_multi_step": smoothed_left_qpos.tolist(),
                "right_qpos_multi_step": smoothed_right_qpos.tolist(),
            },
            "vis": {
                "left_ee_trans_cam": self.tensor_to_numpy(action_dict["left_ee_trans_cam"]).tolist(),
                "right_ee_trans_cam": self.tensor_to_numpy(action_dict["right_ee_trans_cam"]).tolist(),
                "pred_3d": self.tensor_to_numpy(action_dict["pred_3d"]).tolist(),
            }
        }
