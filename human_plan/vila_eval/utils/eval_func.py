import torch
import numpy as np

def to_ndarray(sample):
  skip_keys = [
    "rgb_obs", "language_", "frame_count",
    "raw_width","raw_height",
    "raw_w","raw_h",
  ]
  for key in sample.keys():
    if key in skip_keys or "language_" in key:
      continue
    try:
      sample[key] = [v if v is not None else 0 for v in sample[key]]
    except Exception:
      print(key)
      exit()

    sample[key] = [v if v is not None else 0 for v in sample[key]]
    sample[key] = np.array(sample[key])
    # sample[key][~np.isfinite(sample[key])] = 0

EPIC_KITCHEN_HEIGHT = 256
EPIC_KITCHEN_WIDTH = 456


def eval_single_sample(
  raw_data_dict,
  tokenizer,
  model,
  image_width=EPIC_KITCHEN_WIDTH,
  image_height=EPIC_KITCHEN_HEIGHT,
):
  model_device = next(model.parameters()).device
  data_dict = {}
  data_dict["images"] = raw_data_dict["image"].to(model.dtype).to(model_device)

  num_images = data_dict["images"].shape[0]

  input_ids_tensor = raw_data_dict["input_ids"]
  if isinstance(input_ids_tensor, torch.Tensor):
    if input_ids_tensor.device != model_device:
      input_ids_tensor = input_ids_tensor.to(model_device)
  else:
    input_ids_tensor = torch.tensor(input_ids_tensor, device=model_device)
  
  labels_tensor = raw_data_dict["labels"]
  if isinstance(labels_tensor, torch.Tensor):
    if labels_tensor.device != model_device:
      labels_tensor = labels_tensor.to(model_device)
  else:
    labels_tensor = torch.tensor(labels_tensor, device=model_device)

  data_dict["input_ids"] = torch.nn.utils.rnn.pad_sequence(
      [input_ids_tensor], batch_first=True, padding_value=tokenizer.pad_token_id
  )
  data_dict["attention_mask"] = data_dict["input_ids"].ne(tokenizer.pad_token_id)
  data_dict["labels"] = torch.nn.utils.rnn.pad_sequence(
      [labels_tensor], batch_first=True, padding_value=-100
  )

  data_dict["inference"] = True
  data_dict["raw_action_labels"] = raw_data_dict["raw_action_label"].to(model.dtype).to(model_device) # No Need to Unsqueeze
  data_dict["raw_action_masks"] = raw_data_dict["raw_action_mask"].to(model_device) # No Need to Unsqueeze

  data_dict["raw_proprio_inputs"] = raw_data_dict["proprio_input"].to(model.dtype).to(model_device)
  data_dict["raw_proprio_inputs_2d"] = raw_data_dict["proprio_input_2d"].to(model.dtype).to(model_device)
  data_dict["raw_proprio_inputs_3d"] = raw_data_dict["proprio_input_3d"].to(model.dtype).to(model_device)
  data_dict["raw_proprio_inputs_rot"] = raw_data_dict["proprio_input_rot"].to(model.dtype).to(model_device)
  data_dict["raw_proprio_inputs_handdof"] = raw_data_dict["proprio_input_handdof"].to(model.dtype).to(model_device)
  data_dict["raw_proprio_inputs_hand_finger_tip"] = raw_data_dict["proprio_input_hand_finger_tip"].to(model.dtype).to(model_device)
  data_dict["raw_ee_movement_masks"] = raw_data_dict["ee_movement_mask"].to(model.dtype).to(model_device)

  for k, v in data_dict.items():
    if isinstance(v, torch.Tensor):
      if v.device != model_device:
        data_dict[k] = v.to(model_device)
    elif isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
      for i, t in enumerate(v):
        if t.device != model_device:
          v[i] = t.to(model_device)

  with torch.inference_mode():
    output = model.forward(**data_dict)
  
  return output.prediction.cpu().numpy(), \
    raw_data_dict["raw_image_obs"], \
    raw_data_dict["raw_action_label"].cpu().numpy(), \
    data_dict["raw_action_masks"].cpu().numpy(), \
    output.loss.item()
