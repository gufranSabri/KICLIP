import torch
import os
from pprint import pprint

raw_clip = '/home/g202302610/.cache/clip/ViT-B-16.pt'
source_dir = './basetraining/B2N_ucf101_KICLIP_10/checkpoints'
output_dir = './basetraining/B2N_ucf101_KICLIP_10/wa_checkpoints'

wa_start = 2
wa_end = 22

def average_checkpoint(checkpoint_list):
    ckpt_list = []
    
    # raw clip
    raw_clip_weight = {}
    clip_ori_state = torch.jit.load(raw_clip, map_location='cpu').state_dict() 
    _ = [clip_ori_state.pop(i) for i in ['input_resolution', 'context_length', 'vocab_size']]
    for key in clip_ori_state:
        raw_clip_weight['model.' + key] = clip_ori_state[key]

    ckpt_list.append((0, raw_clip_weight))
    for name, ckpt_id in checkpoint_list:
        ckpt_list.append((ckpt_id, torch.load(name, map_location='cpu')['model_state']))
    
    modded_model_keys = []
    for k, v in ckpt_list[-1][1].items():
        if 'projector' in k:
            modded_model_keys.append(k)
        elif 'adapter' in k:
            modded_model_keys.append(k)
        elif 'post_prompt' in k:
            modded_model_keys.append(k)
        elif "tempx_blocks" in k:
            modded_model_keys.append(k)
        elif "dca" in k:
            modded_model_keys.append(k)
        elif "lstm" in k:
            modded_model_keys.append(k)
        elif "spatial_model" in k:
            modded_model_keys.append(k)
        elif "temporal_model" in k:
            modded_model_keys.append(k)
        elif "resblocks_f" in k:
            modded_model_keys.append(k)

    # threshold filter
    new_ckpt_list = []
    ckpt_id_list = []
    for i in ckpt_list:
        if int(i[0]) >= wa_start and int(i[0]) <= wa_end:
            new_ckpt_list.append(i)
            ckpt_id_list.append(int(i[0]))
    
    print("Files with the following paths will participate in the parameter averaging")
    print(ckpt_id_list)

    pprint(modded_model_keys)

    state_dict = {}
    for key in raw_clip_weight:
        state_dict[key] = []
        for ckpt in new_ckpt_list:
            state_dict[key].append(ckpt[1][key])
    
    for key in modded_model_keys:
        state_dict[key] = []
        for ckpt in new_ckpt_list:
            state_dict[key].append(ckpt[1][key])
        if "agg_weights" in key:
            print(state_dict[key])
    
    for key in state_dict:
        try:
            state_dict[key] = torch.mean(torch.stack(state_dict[key], 0), 0)
        except:
            print(key)

    return state_dict


os.makedirs(output_dir, exist_ok=True)
checkpoint_list = os.listdir(source_dir)
    
checkpoint_list = [(os.path.join(source_dir, i), int(i.split('.')[0].split('_')[-1])) for i in checkpoint_list]
checkpoint_list = sorted(checkpoint_list, key=lambda d: d[1])

swa_state_dict = average_checkpoint(checkpoint_list)
torch.save({'model_state': swa_state_dict}, os.path.join(output_dir, 'swa_%d_%d.pth'%(wa_start, wa_end)))

