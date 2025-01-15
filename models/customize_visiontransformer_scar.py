from collections import OrderedDict
from typing import Tuple, Union
import copy

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from clip.model import LayerNorm, QuickGELU
from models.torch_utils import activation
from models.scar_components import *

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, num_experts=0, record_routing=False, routing_type='patch-level'):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.num_experts = num_experts
        self.record_routing = record_routing
        self.routing_type = routing_type

        if num_experts > 0:    
            self.experts_head = nn.Sequential(*[nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    # ("c_proj", nn.Linear(d_model * 4, d_model))
                ])) for _ in range(num_experts)])
            
            self.experts_tail = nn.Sequential(*[nn.Sequential(OrderedDict([
                    ("c_proj", nn.Linear(d_model * 4, d_model))
                ])) for _ in range(num_experts)])
            
            self.routing1 = nn.Linear(d_model, self.num_experts + 1)
            self.routing2 = nn.Linear(d_model*4, self.num_experts + 1)
        
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        if type(x) == tuple:
            x, routing_state = x
        else:
            routing_state = None

        x = x + self.attention(self.ln_1(x))
        ln_x = self.ln_2(x)
        # x = x + self.mlp(self.ln_2(x))
        if self.num_experts > 0:
            # output = self.experts_tail[0](self.experts_head[0][1](self.experts_head[0][0](ln_x)))
             
            output_head = [self.mlp[0](ln_x)]
            [output_head.append(self.experts_head[i][0](ln_x)) for i in range(self.num_experts)]
            
            if self.routing_type == 'patch-level':
                rout1 = torch.nn.functional.softmax(self.routing1(ln_x), -1).unsqueeze(-1)
            elif self.routing_type == 'image-level':
                rout1 = torch.nn.functional.softmax(self.routing1(ln_x[0].unsqueeze(0)), -1).unsqueeze(-1)

            output_head = torch.stack(output_head, 0).permute(1,2,0,3)
            output_head = (rout1 * output_head).sum(-2)
            output_head = self.mlp[1](output_head)
            
            """
            output_head = [self.mlp[1](self.mlp[0](ln_x))]
            [output_head.append(self.experts_head[i](ln_x)) for i in range(self.num_experts)]
            rout1 = torch.nn.functional.softmax(self.routing1(ln_x), -1).unsqueeze(-1)
            output_head = torch.stack(output_head, 0).permute(1,2,0,3)
            output_head = (rout1 * output_head).sum(-2)
            """    
            
            output = [self.mlp[2](output_head)]
            [output.append(self.experts_tail[i](output_head)) for i in range(self.num_experts)]
            # rout2 = torch.nn.functional.softmax(self.routing2(output_head), -1).unsqueeze(-1)
            if self.routing_type == 'patch-level':
                rout2 = torch.nn.functional.softmax(self.routing2(output_head), -1).unsqueeze(-1)
            elif self.routing_type == 'image-level':
                rout2 = torch.nn.functional.softmax(self.routing2(output_head[0].unsqueeze(0)), -1).unsqueeze(-1)
            output = torch.stack(output, 0).permute(1,2,0,3)
            output = (rout2 * output).sum(-2)
            
        else:
            output = self.mlp(ln_x)
        
        x = x + output
        # x = x + self.experts[0](self.ln_2(x))
        if self.record_routing:
            if self.num_experts > 0:
                current_rout = torch.stack([rout1.squeeze(-1), rout2.squeeze(-1)], 0)    
                if routing_state == None:
                    routing_state = current_rout
                else:
                    routing_state = torch.cat([routing_state, current_rout], 0)
             
            return x, routing_state
        
        return x

class TimesAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, T=0, temporal_modeling_type='expand_temporal_view'):
        super().__init__()
        self.T = T
        
        self.attn = activation.MultiheadAttention(d_model, n_head, temporal_shift=temporal_modeling_type, T=T)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))

        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        l, bt, d = x.size()
        b = bt // self.T
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
         
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class TSTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, use_checkpoint=False, T=8, temporal_modeling_type=None, num_experts=0, expert_insert_layers=[], record_routing=False, routing_type='patch-level'):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.width = width
        self.layers = layers
        self.use_checkpoint = use_checkpoint
        self.T = T
        self.temporal_modeling_type = temporal_modeling_type
        self.record_routing = record_routing
        self.routing_type = routing_type

        if self.temporal_modeling_type == None:
            print("INITIALIZING RESIDUAL ATTENTION")
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, num_experts, record_routing, routing_type) if layer_id in expert_insert_layers else ResidualAttentionBlock(width, heads, attn_mask, record_routing=record_routing, routing_type=routing_type) for layer_id in range(layers)])
        elif self.temporal_modeling_type == "expand_temporal_view":
            print("INITIALIZING TIMES ATTENTION")
            self.resblocks = nn.Sequential(*[TimesAttentionBlock(width, heads, attn_mask, T=T, temporal_modeling_type=self.temporal_modeling_type) for _ in range(layers)])
        self.layers = layers

    def forward(self, x, prompt_token=None):
        if not self.use_checkpoint:
            for block in self.resblocks:
                x = block(x)
            return x
        else:
            return checkpoint_sequential(self.resblocks, 3, x)

class TemporalVisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, T = 8, temporal_modeling_type = None, use_checkpoint = False, num_experts=0, expert_insert_layers=[], record_routing=False, routing_type='patch-level', vil=True, add_spatial_model=True, add_temporal_model=True):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.temporal_modeling_type = temporal_modeling_type
        self.T = T
        self.use_checkpoint = use_checkpoint
        self.record_routing = record_routing
        self.routing_type = routing_type

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = TSTransformer(width, layers, heads, use_checkpoint=self.use_checkpoint, T=self.T, temporal_modeling_type=self.temporal_modeling_type, num_experts=num_experts, expert_insert_layers=expert_insert_layers, record_routing=record_routing, routing_type=routing_type)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.tuned_model = False
        self.add_spatial_model = add_spatial_model
        self.add_temporal_model = add_temporal_model
        self.vil = vil

        self.width = width
        self.layers = layers
        self.heads = heads
        self.num_experts = num_experts
        self.expert_insert_layers = expert_insert_layers
        self.record_routing = record_routing
        self.routing_type = routing_type

    def construct_scar_components(self):
        if self.add_spatial_model and self.vil:
            self.spatial_model = torch.hub.load("nx-ai/vision-lstm", "vil2-base")
            self.spatial_model.patch_embed = torch.nn.Identity()
            self.spatial_model.pos_embed = torch.nn.Identity()
            self.spatial_model.head = torch.nn.Linear(1536, 768)
        elif self.add_spatial_model:
            self.spatial_model = TSTransformer(self.width, self.layers, self.heads, use_checkpoint=self.use_checkpoint, T=self.T, temporal_modeling_type=None, num_experts=self.num_experts, expert_insert_layers=self.expert_insert_layers, record_routing=self.record_routing, routing_type=self.routing_type)
            self.ln_post_s = copy.deepcopy(self.ln_post)

        if self.add_temporal_model and self.vil:
            self.temporal_model = torch.hub.load("nx-ai/vision-lstm", "vil2-base")
            self.temporal_model.patch_embed = torch.nn.Identity()
            self.temporal_model.pos_embed = torch.nn.Identity()
            self.temporal_model.head = torch.nn.Linear(1536, 768)
        elif self.add_temporal_model:
            self.temporal_model = copy.deepcopy(self.transformer)
            self.ln_post_t = copy.deepcopy(self.ln_post)

        self.tuned_model = True

        print("Add spatial model:", self.add_spatial_model, "|", "Add temporal model:", self.add_temporal_model, "|", "Model Type is ViL:", self.vil)


    def temporize_patches(self, x):
        bT,p,d = x.shape
        x = x.reshape(bT//self.T, self.T, p, d) #[b, T, 196, 768]

        x = x.view(4, 8, 14, 14, 768) #[b, T, 14, 14, 768]
        # x = x[:, :, :8, :, :] #[b, T, 8, 14, 768]
        x = x[:, :, 3:11, :, :] #[b, T, 8, 14, 768]
        x = x.contiguous().view(4, 8, 8 * 14, 768) #[b, T, 112, 768]

        last_frame = x[:, -1:, :, :] #[b, 1, 112, 768]
        repeat_count = 14 - x.shape[1]
        last_frame_repeated = last_frame.repeat(1, repeat_count, 1, 1)

        x_temporized = torch.cat([x, last_frame_repeated], dim=1)
        x_temporized = x_temporized.contiguous().view(4, 14, 8, 14, 768)  # [b, T, 8, 14, 768]
        x_temporized = x_temporized.permute(0,2,1,3,4)  # [b, H=8, T=14, 14, 768]
        x_temporized = x_temporized.contiguous().view(4 * 8, 14 * 14, 768)  # Shape [b, T, 196, 768]
        x_temporized = x_temporized.reshape(bT//self.T * self.T, p, d) # Shape [b*T, 196, 768]

        return x_temporized
        
    def forward(self, x):
        s_x, temp_x = None, None

        x, [maskf, mask] = x
        x = self.conv1(x) #[b*T, 768, 14, 14]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [b*T, 768, 196]
        x = x.permute(0, 2, 1).contiguous()  # [b*T, 196, 768]

        if self.add_spatial_model and self.tuned_model:
            if not self.vil:
                s_x = self.spatial_model(x.permute(1, 0, 2))
            else:
                s_x = self.spatial_model(x)
            if not self.vil:
                s_x = s_x.permute(1, 0, 2)  # [b*T, 196, 768]
                s_x = self.ln_post_s(s_x[:, 0, :]) #[b*T, 196, 768] -> taking CLS token -> [b*T, 768]

        if self.add_temporal_model and self.tuned_model:
            if not self.vil:
                temp_x = self.temporal_model(self.temporize_patches(x).permute(1, 0, 2))
            else:
                temp_x = self.temporal_model(self.temporize_patches(x))
            if not self.vil:
                temp_x = temp_x.permute(1, 0, 2)  # [b*T, 196, 768]
                temp_x = self.ln_post_t(temp_x[:, 0, :]) #[b*T, 196, 768] -> taking CLS token -> [b*T, 768]

        # TSTransformer
        cls_token = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device) #[b*T, 1, 768]
        x = torch.cat([cls_token, x], dim=1)  # [b*T, 197, 768]
        x = x + self.positional_embedding.to(x.dtype) #[b*T, 197, 768]
        x = self.ln_pre(x) #[b*T, 197, 768]

        x = x.permute(1, 0, 2)  # [197, b*T, 768]
        x = self.transformer(x) # [197, b*T, 768]
        ts_x = x.permute(1, 0, 2)  # [b*T, 197, 768]
        ts_x = self.ln_post(ts_x[:, 0, :]) #[b*T, 197, 768] -> taking CLS token -> [b*T, 768]

        if self.add_spatial_model and self.add_temporal_model and self.tuned_model:
            x = (ts_x + s_x + temp_x)/3
        elif self.add_spatial_model and self.tuned_model:
            x = (ts_x + s_x)/2
        elif self.add_temporal_model and self.tuned_model:
            x = (ts_x + temp_x)/2
        else:
            x = ts_x

        if self.proj is not None:
            x = x @ self.proj #[b*T, num_classes]
        
        return [x, ts_x]