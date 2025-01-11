from collections import OrderedDict
from typing import Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from clip.model import LayerNorm, QuickGELU
from models.torch_utils import activation
from models.scar_components import *

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
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
         
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

        self.resblocks = nn.Sequential(*[TimesAttentionBlock(width, heads, attn_mask, T=T, temporal_modeling_type=self.temporal_modeling_type) for _ in range(layers)])
        self.layers = layers

    def forward(self, x, prompt_token=None):
        if not self.use_checkpoint:
            # preliminaries = []
            for block in self.resblocks:
                x = block(x)
                # preliminaries.append(x)
            # return x, preliminaries
            return x
        else:
            return checkpoint_sequential(self.resblocks, 3, x)


# class TemporalVisionTransformer(nn.Module):
#     def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, T = 8, temporal_modeling_type = None, use_checkpoint = False, num_experts=0, expert_insert_layers=[], record_routing=False, routing_type='patch-level'):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.output_dim = output_dim
#         self.temporal_modeling_type = temporal_modeling_type
#         self.T = T
#         self.use_checkpoint = use_checkpoint
#         self.record_routing = record_routing
#         self.routing_type = routing_type

#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

#         scale = width ** -0.5
#         self.class_embedding = nn.Parameter(scale * torch.randn(width))
#         self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
#         self.ln_pre = LayerNorm(width)

#         self.transformer = TSTransformer(width, layers, heads, use_checkpoint=self.use_checkpoint, T=self.T, temporal_modeling_type=self.temporal_modeling_type, num_experts=num_experts, expert_insert_layers=expert_insert_layers, record_routing=record_routing, routing_type=routing_type)

#         self.ln_post = LayerNorm(width)
#         self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

#         self.tuned_model = False
#         self.layers = layers

#     def construct_scar_components(self):
#         self.dca = DualCrossAttention()
#         self.tempx_blocks = nn.Sequential(*[SCAR_TempX(T=self.T) for _ in range(self.layers)])
#         self.tuned_model = True
        
#     def forward(self, x):
#         x, [maskf, mask] = x
#         x = self.conv1(x) #[b*T, 768, 14, 14]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # [b*T, 768, 196]
#         x = x.permute(0, 2, 1).contiguous()  # [b*T, 196, 768]

#         # TSTransformer
#         cls_token = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device) #[b*T, 1, 768]
#         x = torch.cat([cls_token, x], dim=1)  # [b*T, 197, 768]
#         x = x + self.positional_embedding.to(x.dtype) #[b*T, 197, 768]
#         x = self.ln_pre(x) #[b*T, 197, 768]

#         x = x.permute(1, 0, 2)  # [197, b*T, 768]
#         ts_x, prelims = self.transformer(x) # [197, b*T, 768], [12, 197, b*T, 768]
#         ts_x = ts_x.permute(1, 0, 2)  # [b*T, 197, 768]

#         if self.tuned_model:
#             x_tmpx = torch.zeros((ts_x.shape[0], ts_x.shape[-1]), requires_grad=True).to("cuda" if torch.cuda.is_available() else "mps")
#             for i in range(len(prelims)):
#                 x_tmpx = x_tmpx + self.tempx_blocks[i](prelims[i])
#             x_tmpx = x_tmpx / len(prelims)
            
#         x_attn = self.ln_post(ts_x[:, 0, :]) #[b*T, 197, 768] -> taking CLS token -> [b*T, 768]
#         if self.tuned_model:
#             x = (x_attn+x_tmpx)/2
#             # x = x_attn + 0.5*x_tmpx

#             # x_attn_r = x_attn.reshape(x_attn.shape[0]//self.T, self.T, x_attn.shape[-1]) #[b, T, 768]
#             # x_tmpx = x_tmpx.reshape(x_tmpx.shape[0]//self.T, self.T, x_tmpx.shape[-1])
#             # x_dca = self.dca(x_attn_r, x_tmpx).reshape(x_tmpx.shape[0]*self.T, x_tmpx.shape[-1]) #[b*T, 768]

#             # x = x_attn + 0.5*x_dca
#         else:
#             x = x_attn

        

#         if self.proj is not None:
#             x = x @ self.proj #[b*T, num_classes]
        
#         return x

# TYPE
class TemporalVisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, T = 8, temporal_modeling_type = None, use_checkpoint = False, num_experts=0, expert_insert_layers=[], record_routing=False, routing_type='patch-level', vil=True, lstm=False):
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
        self.include_vil = vil
        self.include_lstm = lstm

    def construct_scar_components(self):
        if self.include_vil:
            self.vil = torch.hub.load("nx-ai/vision-lstm", "vil2-base")
            self.vil.patch_embed = torch.nn.Identity()
            self.vil.pos_embed = torch.nn.Identity()
            self.vil.head = torch.nn.Linear(1536, 768)

        if self.include_lstm:
            self.lstm = SCAR_LSTM(768, 512, 2)
        
        self.tuned_model = True

        print("ViL:", self.include_vil, "LSTM:", self.include_lstm)
        
    def forward(self, x):
        x, [maskf, mask] = x
        x = self.conv1(x) #[b*T, 768, 14, 14]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [b*T, 768, 196]
        x = x.permute(0, 2, 1).contiguous()  # [b*T, 196, 768]

        # ViL encoder
        if self.include_vil and self.tuned_model:
            vil_x = self.vil(x)

        # TSTransformer
        cls_token = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device) #[b*T, 1, 768]
        x = torch.cat([cls_token, x], dim=1)  # [b*T, 197, 768]
        x = x + self.positional_embedding.to(x.dtype) #[b*T, 197, 768]
        x = self.ln_pre(x) #[b*T, 197, 768]

        x = x.permute(1, 0, 2)  # [197, b*T, 768]
        x = self.transformer(x) # [197, b*T, 768]
        ts_x = x.permute(1, 0, 2)  # [b*T, 197, 768]

        # # SCAR LSTM
        if self.include_lstm:
            lstm_x = ts_x.reshape(ts_x.shape[0]//self.T, self.T, ts_x.shape[1], ts_x.shape[-1])
            lstm_x = self.lstm(lstm_x)
            lstm_x = lstm_x.reshape(lstm_x.shape[0]*lstm_x.shape[1], -1)

        ts_x = self.ln_post(ts_x[:, 0, :]) #[b*T, 197, 768] -> taking CLS token -> [b*T, 768]

        if self.include_vil and self.include_lstm:
            x = (ts_x + vil_x + lstm_x)/3
        elif self.include_vil and self.tuned_model:
            x = (ts_x + vil_x)/2
        elif self.include_lstm:
            x = (ts_x + lstm_x)/2
        else:
            x = ts_x

        if self.proj is not None:
            x = x @ self.proj #[b*T, num_classes]
        
        return [x, ts_x]

