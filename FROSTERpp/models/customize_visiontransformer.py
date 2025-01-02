from collections import OrderedDict
from typing import Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from clip.model import LayerNorm, QuickGELU
from models.torch_utils import activation

# TYPE 1: expand temporal attention view
class TimesAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, T=0, temporal_modeling_type='expand_temporal_view'):
        super().__init__()
        self.T = T
        
        # type: channel_shift or expand_temporal_view
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
        # x = x.view(l, b, self.T, d)
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
         
        return x

# ORIGIN Type
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

# ORIGIN
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

# TYPE
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
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, num_experts, record_routing, routing_type) if layer_id in expert_insert_layers else ResidualAttentionBlock(width, heads, attn_mask, record_routing=record_routing, routing_type=routing_type) for layer_id in range(layers)])
        elif self.temporal_modeling_type == 'expand_temporal_view' or self.temporal_modeling_type == 'expand_temporal_view_step2' or self.temporal_modeling_type == 'expand_temporal_view_step3':# TimesAttentionBlock
            self.resblocks = nn.Sequential(*[TimesAttentionBlock(width, heads, attn_mask, T=T, temporal_modeling_type=self.temporal_modeling_type) for _ in range(layers)])

    def forward(self, x, prompt_token=None):
        if not self.use_checkpoint:
            if not self.record_routing:
                # return self.resblocks(x)
                for block in self.resblocks:
                    x = block(x)
                    l, b, c = x.size()
                    if prompt_token is not None:
                        x[-1, :, :] = x[-1, :, :] + prompt_token.view(b, c)
                return x
            else:
                # return self.resblocks(x)
                for block in self.resblocks:
                    x = block(x)
                    l, b, c = x.size()
                    if prompt_token is not None:
                        x[-1, :, :] = x[-1, :, :] + prompt_token.view(b, c)
                return x

        else:
            return checkpoint_sequential(self.resblocks, 3, x)

# TYPE
class TemporalVisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, T = 8, temporal_modeling_type = None, use_checkpoint = False, num_experts=0, expert_insert_layers=[], record_routing=False, routing_type='patch-level'):
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

    def forward(self, x):
        x, [maskf, mask] = x
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1).contiguous()  # shape = [*, grid ** 2, width] # (bxt) l c
        B, l, c = x.shape
        b = B // self.T

        cls_token = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        
        if self.record_routing:
            x, routing_state = self.transformer(x)
        else:
            x = self.transformer(x)
        
        feature = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(feature[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        
        # if self.record_routing:
        #     return [x, feature], routing_state
        # else:
        return [x, feature]