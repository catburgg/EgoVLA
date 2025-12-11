"""
WebSocket Server Layer
Only handles data transmission, no data processing
"""

import asyncio
import websockets
import json
import numpy as np
import argparse
from transformers import HfArgumentParser

from human_plan.vila_train.args import (
    VLATrainingArguments, VLAModelArguments, VLADataArguments
)
from human_plan.ego_bench_eval.model_interface import ModelInterface, EgoVLAInterface


class WebSocketServer:
    """WebSocket server that only handles data transmission"""
    
    def __init__(self, interface: ModelInterface):
        self.interface = interface
    
    async def handle_client(self, websocket):
        print(f"Client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                request = json.loads(message)
                if request.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
                    continue
                
                if request.get("type") == "reset":
                    rgb_obs = np.array(request["rgb_obs"], dtype=np.uint8)
                    self.interface.reset_history(rgb_obs)
                    await websocket.send(json.dumps({"type": "reset_ack"}))
                    continue
                
                proprio = request["proprio"]
                formatted_response = self.interface.infer(
                    np.array(request["rgb_obs"], dtype=np.uint8),
                    np.array(proprio["left_finger_tip"]),
                    np.array(proprio["right_finger_tip"]),
                    np.array(proprio["left_ee_pose"]),
                    np.array(proprio["right_ee_pose"]),
                    np.array(proprio["qpos"]),
                    np.array(proprio["cam_intrinsics"]),
                    language_instruction=request.get("language_instruction"),
                    task_name=request.get("task_name"),
                    input_hand_dof=request.get("config", {}).get("input_hand_dof"),
                    raw_width=request.get("config", {}).get("raw_width", 1280),
                    raw_height=request.get("config", {}).get("raw_height", 720),
                )
                
                response = {
                    "action": formatted_response.get("action", {}),
                    "vis": formatted_response.get("vis", {})
                }
                    
                await websocket.send(json.dumps(response))
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="localhost")
    
    hf_parser = HfArgumentParser((VLAModelArguments, VLADataArguments, VLATrainingArguments))
    args, remaining_args = parser.parse_known_args()
    
    model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses(remaining_args)
    
    print("Loading model...")
    interface = EgoVLAInterface.setup(args.model_path, model_args, data_args, training_args)
    print("Model loaded")
    
    server = WebSocketServer(interface)
    print(f"Server starting on ws://{args.host}:{args.port}")
    async with websockets.serve(server.handle_client, args.host, args.port, max_size=50 * 1024 * 1024):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
