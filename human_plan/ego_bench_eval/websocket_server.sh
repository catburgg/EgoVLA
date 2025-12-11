#!/bin/bash
# WebSocket Server Startup Script
# Usage: bash websocket_server.sh [model_path] [port] [host] [model_args...]

MODEL_PATH=${1:-$(find checkpoints/ego_vla_checkpoint -type d -name "ckpt-*" -print -quit)}
PORT=${2:-8765}
HOST=${3:-localhost}

# Source environment if needed
# source /home/rchal97/code/clean_egovla/isaacsim/setup_conda_env.sh

echo "Starting WebSocket Server..."
echo "Model path: $MODEL_PATH"
echo "Port: $PORT"
echo "Host: $HOST"

# Shift to get remaining model arguments
shift 3 2>/dev/null || shift

python human_plan/ego_bench_eval/websocket_server.py \
    --model_path "$MODEL_PATH" \
    --port "$PORT" \
    --host "$HOST" \
    --output_dir tmp/ \
    --version qwen2 \
    --vision_tower google/siglip-so400m-patch14-384 \
    --data_mixture otv_sim_fixed_set_aug_AUG_SHIFT_30Hz_train \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --group_by_modality_length False \
    --future_index 1 \
    --predict_future_step 30 \
    --max_action 1 \
    --min_action 0 \
    --add_his_obs_step 5 \
    --add_his_imgs True \
    --add_his_img_skip 6 \
    --num_action_bins 256 \
    --action_tokenizer uniform \
    --invalid_token_weight 0.1 \
    --mask_input True \
    --add_current_language_description False \
    --traj_decoder_type transformer_split_action_v2 \
    --raw_action_label True \
    --traj_action_output_dim 48 \
    --input_placeholder_diff_index True \
    --ee_loss_coeff 20.0 \
    --hand_loss_coeff 5.0 \
    --hand_loss_dim 6 \
    --ee_2d_loss_coeff 0.0 \
    --ee_rot_loss_coeff 5.0 \
    --hand_kp_loss_coeff 0.0 \
    --next_token_loss_coeff 0.0 \
    --traj_action_output_ee_2d_dim 0 \
    --traj_action_output_ee_dim 6 \
    --traj_action_output_hand_dim 30 \
    --traj_action_output_ee_rot_dim 12 \
    --ee_rot_representation rot6d \
    --correct_transformation True \
    --include_2d_label True \
    --include_rot_label True \
    --use_short_language_label True \
    --no_norm_ee_label True \
    --lazy_preprocess True \
    --tf32 True \
    --merge_hand True \
    --use_mano True \
    --sep_proprio True \
    --sep_query_token True \
    --loss_use_l1 True \
    "$@"

