import subprocess

# Define an array of strings (arguments)
task_names = [
  "Humanoid-Push-Box-v0",
  "Humanoid-Open-Drawer-v0",
  "Humanoid-Close-Drawer-v0",
  "Humanoid-Pour-Balls-v0",
  "Humanoid-Flip-Mug-v0",
  "Humanoid-Unload-Cans-v0",
  "Humanoid-Insert-Cans-v0",
  "Humanoid-Open-Laptop-v0",
  "Humanoid-Stack-Can-v0",
  "Humanoid-Stack-Can-Into-Drawer-v0",
  "Humanoid-Sort-Cans-v0",
  "Humanoid-Insert-And-Unload-Cans-v0",
]

room_indexes = ["1", "2", "3"]
# room_indexes = ["1"]
# room_indexes = ["3"]
# table_indexes = ["1", "2", "3"]
table_indexes = ["1"]
# table_indexes = ["2", "3"]
# episode_indexes = ["1", "2", "3", "4", "5"]
num_episodes = 9
num_trials = 1

smooth_weight = "0.2"
hand_smooth_weight = "0.8"

# Path to your Bash script
script_path = "human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh"  # Update this to your script's path

project_trajs = "1"
save_frames = "0"

video_saving_path = f"playground/diverse_eval_aug_flip_mug_700e_transv2_updated_data_100e_finetune_difftableroom_v2/"
additional_label = f"transv2_p30_30hz_flip_mug_700e_updated_data_100e_finetune_difftableroom_v2"
result_saving_path = f"playground/eval_result_diverse/transv2_30hz_p30_flip_mug_700e_updated_data_100e_finetune_difftableroom_v2_smooth_{smooth_weight}_{hand_smooth_weight}.txt"


video_saving_path = f"playground/diverse_eval_aug_flip_mug_700e_transv2_updated_data_100e_finetune_table1room123/"
additional_label = f"transv2_p30_30hz_flip_mug_700e_updated_data_100e_finetune_table1room123"
result_saving_path = f"playground/eval_result_diverse/transv2_30hz_p30_flip_mug_700e_updated_data_100e_finetune_table1room123_smooth_{smooth_weight}_{hand_smooth_weight}.txt"

video_saving_path = f"playground/diverse_eval_aug_flip_mug_700e_transv2_updated_data_nopretrain_table1room123/"
additional_label = f"transv2_p30_30hz_flip_mug_700e_updated_data_nopretrain_table1room123"
result_saving_path = f"playground/eval_result_diverse/transv2_30hz_p30_flip_mug_700e_updated_data_nopretrain_table1room123_smooth_{smooth_weight}_{hand_smooth_weight}.txt"

video_saving_path = f"playground/diverse_eval_aug_transv2_updated_data_100e_hoi4dhot3dholotaco_6720_fix/"
additional_label = f"transv2_p30_30hz_updated_data_100e_hoi4dhot3dholotaco_6720_fix"
result_saving_path = f"playground/eval_result_diverse/transv2_30hz_p30_updated_data_100e_hoi4dhot3dholotaco_6720_fix_smooth_{smooth_weight}_{hand_smooth_weight}.txt"

# Iterate through the array and run the script with each argument
from tqdm import tqdm
for task_name in tqdm(task_names, desc="Task Name"):
  # print("Task Name:", task_name)
  for room_idx in tqdm(room_indexes, desc="Room Idx"):
    for table_idx in tqdm(table_indexes, desc="Table Idx"):
      try:
          # Run the Bash script with the current argument
          print(
            "bash", script_path, task_name, room_idx, table_idx, smooth_weight, str(num_episodes), str(num_trials), result_saving_path, save_frames, project_trajs, hand_smooth_weight, video_saving_path, additional_label
          )
          result = subprocess.run(
              ["bash", script_path, task_name, room_idx, table_idx, smooth_weight, str(num_episodes), str(num_trials), result_saving_path, save_frames, project_trajs, hand_smooth_weight, video_saving_path, additional_label ],  # Command to run the Bash script
              # capture_output=True,          # Capture the output
              # text=True,                    # Return output as string
              check=True                    # Raise an error if the command fails
          )
          print(f"Output for '{task_name} {room_idx}':\n{result.stdout}")
      except subprocess.CalledProcessError as e:
          print(f"Error for '{task_name} {room_idx}':\n{e.stderr}")
