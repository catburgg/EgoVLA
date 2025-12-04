DATASET_ROOT=data/EgoVLA-Humanoid-Sim
SAVE_PATH=data/processed_data_otv

CHUNKS=32
for IDX in {0..31}; do
    CUDA_VISIBLE_DEVICES=$(($IDX % 8)) python human_plan/dataset_preprocessing/otv_isaaclab/generate_img_parquets.py \
        --num_chunk $CHUNKS  --clip_starting 0 \
        --chunk_id $IDX --frame_skip 1 \
        --dataset_root $DATASET_ROOT \
        --save_path $SAVE_PATH &
done
wait

python human_plan/dataset_preprocessing/otv_isaaclab/parquet_to_dataset_images.py --save_path $SAVE_PATH

python human_plan/dataset_preprocessing/otv_isaaclab/generate_images_mapping.py --save_path $SAVE_PATH