DATASET_ROOT=data/EgoVLA-Humanoid-Sim
SAVE_PATH=processed_data_otv

additional_tag=FIXED_SET_MIX

frame_skip=1

# Now 30 Hz
CHUNKS=16
for IDX in {0..15}; do
    python human_plan/dataset_preprocessing/otv_isaaclab/hf_dataset_fixed_set/generate_hand_parquets.py \
        --num_chunk $CHUNKS  --additional_tag $additional_tag  \
        --clip_starting 0  \
        --chunk_id $IDX --frame_skip $frame_skip --sample_skip 1 --future_len 61 \
        --dataset_root $DATASET_ROOT \
        --save_path $SAVE_PATH &
done
wait

python human_plan/dataset_preprocessing/otv_isaaclab/parquet_to_dataset_hand.py --additional_tag $additional_tag --save_path $SAVE_PATH
