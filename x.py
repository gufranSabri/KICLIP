# import torch
# from transformers import VivitModel

# # Load and modify the VivitModel
# model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", torch_dtype=torch.float16)
# # model.encoder = torch.nn.Identity()
# # model.layernorm = torch.nn.Identity()
# # model.pooler = torch.nn.Identity()

# # Ensure the model is in evaluation mode
# model.eval()

# # Create dummy input: (batch_size, num_frames, height, width, num_channels)
# batch_size = 2
# num_frames = 16
# height = 224
# width = 224
# num_channels = 3
# dummy_input = torch.randn(batch_size, num_frames, num_channels, height, width, dtype=torch.float16)


# # Forward pass through the model
# with torch.no_grad():
#     output = model(pixel_values=dummy_input)

# # Print the shape of the output
# print("Output shape:", output.last_hidden_state.shape)

# ==========================

# import av
# import numpy as np
# import torch

# from transformers import VivitImageProcessor, VivitModel
# from huggingface_hub import hf_hub_download

# np.random.seed(0)


# def read_video_pyav(container, indices):
#     '''
#     Decode the video with PyAV decoder.
#     Args:
#         container (`av.container.input.InputContainer`): PyAV container.
#         indices (`List[int]`): List of frame indices to decode.
#     Returns:
#         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
#     '''
#     frames = []
#     container.seek(0)
#     start_index = indices[0]
#     end_index = indices[-1]
#     for i, frame in enumerate(container.decode(video=0)):
#         if i > end_index:
#             break
#         if i >= start_index and i in indices:
#             frames.append(frame)
#     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
#     '''
#     Sample a given number of frame indices from the video.
#     Args:
#         clip_len (`int`): Total number of frames to sample.
#         frame_sample_rate (`int`): Sample every n-th frame.
#         seg_len (`int`): Maximum allowed index of sample's last frame.
#     Returns:
#         indices (`List[int]`): List of sampled frame indices
#     '''
#     converted_len = int(clip_len * frame_sample_rate)
#     end_idx = np.random.randint(converted_len, seg_len)
#     start_idx = end_idx - converted_len
#     indices = np.linspace(start_idx, end_idx, num=clip_len)
#     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
#     return indices


# file_path = hf_hub_download(repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset")
# container = av.open(file_path)

# indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
# video = read_video_pyav(container=container, indices=indices)

# image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
# model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
# # model.encoder = torch.nn.Identity()
# # model.layernorm = torch.nn.Identity()
# # model.pooler = torch.nn.Identity()
# print(model)

# inputs = image_processor(list(video), return_tensors="pt")

# print(inputs.keys(), len(inputs), inputs["pixel_values"].shape)

# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state
# print(list(last_hidden_states.shape))

# ============================

import av
import numpy as np
import torch
from transformers import VivitImageProcessor, VivitModel, VivitConfig
from huggingface_hub import hf_hub_download

np.random.seed(0)

# Define utility functions
def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# Load video
file_path = hf_hub_download(repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset")
container = av.open(file_path)
indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container=container, indices=indices)

# Load model and processor
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
# model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")

# Print model architecture
# print(model)


configuration = VivitConfig()
model = VivitModel(configuration)
print(model)

# Prepare inputs
inputs = image_processor(list(video), return_tensors="pt")
print("Input keys:", inputs.keys())
print("Input shape:", inputs["pixel_values"].shape)

# Define a hook to print shapes
def hook_fn(module, input, output):
    print(f"{module.__class__.__name__}:")
    if isinstance(input, tuple) and input:
        print(f"  Input shape: {[i.shape for i in input if isinstance(i, torch.Tensor)]}")
    if isinstance(output, torch.Tensor):
        print(f"  Output shape: {output.shape}")
    elif isinstance(output, (list, tuple)):
        print(f"  Output shape: {[o.shape for o in output if isinstance(o, torch.Tensor)]}")

# Register hooks for each module in the model
hooks = []
for name, module in model.named_modules():
    # print(name)
    hooks.append(module.register_forward_hook(hook_fn))

# Perform a forward pass
print("\nPerforming forward pass...")
with torch.no_grad():
    outputs = model(**inputs)

# Remove hooks after use
for hook in hooks:
    hook.remove()

# Print final output shape
print("\nFinal Output shape:", outputs.last_hidden_state.shape)

