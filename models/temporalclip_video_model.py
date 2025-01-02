import os
import numpy as np
import json
import warnings
import random
from typing import Tuple, Union
from pprint import pprint

import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from clip.model import CLIP,LayerNorm,Transformer
from clip.clip import _MODELS, _download, tokenize
from models.customize_visiontransformer import TemporalVisionTransformer

class TemporalClipVideo(nn.Module):
    def __init__(self, cfg):
        super(TemporalClipVideo, self).__init__()
        
        self.device = cfg.DEVICE
        self.cfg = cfg
        self._construct_network(cfg)

        self.model.eval()

        for k, v in self.model.named_parameters():
            v.requires_grad = True

        for k, v in self.model.named_parameters():
            v.requires_grad = True

        self.text_dict = self.text_prompt(os.path.join(cfg.DATA.INDEX_LABEL_MAPPING_FILE))
        if not cfg.TEST.OPENSET: self.text_dict = self.text_prompt(os.path.join(cfg.DATA.INDEX_LABEL_MAPPING_FILE))

        self.prompt_type_num = len(self.text_dict)
        self.cls_num = self.text_dict[0].shape[0]
        self.tune_head = cfg.TUNE_HEAD
        self.text_prompting = cfg.MODEL.TEXT_PROMPT
        self.context_length = cfg.MODEL.CONTEXT_LENGTH
        self.record_routing = cfg.MODEL.RECORD_ROUTING
        self.keep_raw_model = cfg.MODEL.KEEP_RAW_MODEL
        self.ensemble_pred = cfg.MODEL.ENSEMBLE_PRED
        self.distillation = cfg.MODEL.RAW_MODEL_DISTILLATION

        self.projector_v = nn.Sequential(
            nn.Linear(self.model.embed_dim, self.model.embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.model.embed_dim, self.model.embed_dim, bias=False)
        )
        nn.init.zeros_(self.projector_v[2].weight)
        nn.init.kaiming_normal_(self.projector_v[0].weight)

        self.projector_t = nn.Sequential(
            nn.Linear(self.model.embed_dim, self.model.embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.model.embed_dim, self.model.embed_dim, bias=False)
        )
        nn.init.zeros_(self.projector_t[2].weight)
        nn.init.kaiming_normal_(self.projector_t[0].weight)

        if self.distillation and (not self.keep_raw_model):
            print("Distillation is not supported if raw model is not kept")
            exit()

        if (self.keep_raw_model and self.ensemble_pred) and self.record_routing:
            print("ensemble pred cannot not exist together with record-routing")
            exit()
        
        if self.text_prompting:
            self.prompt_num = int(cfg.MODEL.PROMPT_NUM)
            embedding_dim = self.model.ln_final.weight.shape[0]
            
            self.prompt_embed = torch.nn.Parameter(
                torch.rand(int(self.prompt_num), embedding_dim), 
                requires_grad=True
            )
            torch.nn.init.normal_(self.prompt_embed, std=0.01)
            
            id2cls = {}
            for idx, cls in  json.load(open(cfg.DATA.INDEX_LABEL_MAPPING_FILE, 'r')).items():
                id2cls[int(idx)] = cls
            self.classnames = [id2cls[i] for i in range(len(id2cls))]
            prompts = [" ".join(["X"] * self.prompt_num) + " " + name + "." for name in self.classnames]
            tokenized_prompts = torch.cat([tokenize(p, context_length=self.context_length) for p in prompts])
            self.tokenized_prompts = tokenized_prompts
            
            for _, param in self.model.transformer.named_parameters():
                param.requires_grad = False
        else:
            self.dynamic_classifier = self.achieve_csf_matrix(self.text_dict, self.model)

        self.alpha = 0.1
        
        self.lr_factor = {
            "message": cfg.MODEL.FINETUNE_FACTOR,
            "stadapt": cfg.MODEL.ADAPT_FINETUNE_FACTOR,
            "mlp": cfg.MODEL.MLP_FINETUNE_FACTOR,
            "experts": cfg.MODEL.EXPERT_FINETUNE_FACTOR,
            "routing": cfg.MODEL.ROUTING_FINETUNE_FACTOR,
        } 

    def _construct_network(self, cfg):
        context_length = cfg.MODEL.CONTEXT_LENGTH

        if cfg.MODEL.ARCH == 'vitb16':
            self.model, self.preprocess = load(
                "ViT-B/16", 
                jit=False, 
                T=cfg.DATA.NUM_FRAMES, 
                temporal_modeling_type=cfg.MODEL.TEMPORAL_MODELING_TYPE,
                use_checkpoint=cfg.MODEL.USE_CHECKPOINT,
                context_length=context_length,
                num_experts=cfg.MODEL.NUM_EXPERTS, 
                expert_insert_layers=cfg.MODEL.EXPERT_INSERT_LAYERS,
                record_routing=cfg.MODEL.RECORD_ROUTING,
                routing_type=cfg.MODEL.ROUTING_TYPE
            )
            if cfg.MODEL.KEEP_RAW_MODEL:   
                self.raw_model, self.preprocess = load(
                    "ViT-B/16", 
                    jit=False, 
                    T=cfg.DATA.NUM_FRAMES, 
                    temporal_modeling_type=None,
                    use_checkpoint=cfg.MODEL.USE_CHECKPOINT, 
                    context_length=context_length,
                    num_experts=cfg.MODEL.NUM_EXPERTS,
                    expert_insert_layers=cfg.MODEL.EXPERT_INSERT_LAYERS,
                    record_routing=cfg.MODEL.RECORD_ROUTING, 
                    routing_type=cfg.MODEL.ROUTING_TYPE
                )
                for _, p in self.raw_model.named_parameters():
                    p.requires_grad = False
        else:
            raise NotImplementedError

    def update_state(self):
        self.dynamic_classifier = self.achieve_csf_matrix(self.text_dict, self.model)

    def forward(self, x=None, update=False): # x: [bz, channel, clip_len, h, w]
        x = x[0]
        if len(x.shape) == 4:
            x = x.unsqueeze(2) # image input
        
        if self.keep_raw_model:
            self.raw_model.eval()

        bz, channel_dim, clip_len, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(bz*clip_len, channel_dim, h, w)
        
        img_encode = self.model.encode_image(x) # [bz, feat_size]

        if isinstance(img_encode, list):
            img_encode, _ = img_encode

        if self.training: # text_dict  {id: [400, feat_size]},
            img_encode = img_encode / img_encode.norm(dim=-1, keepdim=True) #normalization
            
            # logit_scale: scale the similarity scores between two modalities; helps to adjust the temperature in the softmax operation
            if self.text_prompting:
                text_embedding = torch.cat(
                    (
                        self.token_prefix, 
                        self.prompt_embed.unsqueeze(0).expand(len(self.classnames), -1, -1), 
                        self.token_suffix
                    ), 1
                )
                norm_head = self.model.prompt_encode_text(text_embedding, self.tokenized_prompts,)
                norm_head /= norm_head.norm(dim=-1, keepdim=True)
                pred = self.model.logit_scale.exp() * img_encode @ norm_head.T
            else:
                text_dict = self.text_prompt(os.path.join(self.cfg.DATA.INDEX_LABEL_MAPPING_FILE))
                dynamic_classifier_new = self.achieve_csf_matrix(text_dict, self.model, trainable=True)
                pred = self.model.logit_scale.exp() * img_encode @ dynamic_classifier_new.T

            pred = pred.reshape(bz, clip_len, -1).mean(1)
            
            # residual distillation
            if self.keep_raw_model and (self.ensemble_pred or self.distillation):
                with torch.no_grad():
                    raw_img_encode = self.raw_model.encode_image(x)[0]
                    raw_img_encode = raw_img_encode / raw_img_encode.norm(dim=-1, keepdim=True)

                dynamic_classifier_raw = self.achieve_csf_matrix(text_dict, self.raw_model, trainable=False)
                
                img_encode = img_encode + self.alpha * self.projector_v(img_encode)
                dynamic_classifier_new = dynamic_classifier_new + self.alpha * self.projector_t(dynamic_classifier_new)

                return [pred, img_encode, dynamic_classifier_new], [None, raw_img_encode, dynamic_classifier_raw]

            return pred
        
        else: # dynamic_clf: [type_num * cls_num, feat_size]
            img_encode /= img_encode.norm(dim=-1, keepdim=True)

            if self.text_prompting:
                text_embedding = torch.cat(
                    (
                        self.token_prefix, 
                        self.prompt_embed.unsqueeze(0).expand(len(self.classnames), -1, -1), 
                        self.token_suffix
                    ), 1
                )
                
                norm_head = self.model.prompt_encode_text(text_embedding, self.tokenized_prompts,)
                norm_head /= norm_head.norm(dim=-1, keepdim=True)
                pred = self.model.logit_scale.exp() * img_encode @ norm_head.T
            else:
                text_dict = self.text_prompt(os.path.join(self.cfg.DATA.INDEX_LABEL_MAPPING_FILE))
                dynamic_classifier_new = self.achieve_csf_matrix(text_dict, self.model, trainable=False)
                pred = self.model.logit_scale.exp() * img_encode @ dynamic_classifier_new.T

            pred = pred.reshape(bz, clip_len, -1).mean(1)
                        
            if self.keep_raw_model and (self.ensemble_pred or self.distillation):
                return [pred, None], [None, None]
            
            return pred
    
    def text_prompt(self, data_file):
        text_aug = [
            f'a photo of {{}}.',
            f'a photo of a person {{}}.',
            f'a photo of a person using {{}}.',
            f'a photo of a person doing {{}}.',
            f'a photo of a person during {{}}.',
            f'a photo of a person performing {{}}.',
            f'a photo of a person practicing {{}}.',
            f'a video of {{}}.',
            f'a video of a person {{}}.',
            f'a video of a person using {{}}.',
            f'a video of a person doing {{}}.',
            f'a video of a person during {{}}.',
            f'a video of a person performing {{}}.',
            f'a video of a person practicing {{}}.',
            f'a example of {{}}.',
            f'a example of a person {{}}.',
            f'a example of a person using {{}}.',
            f'a example of a person doing {{}}.',
            f'a example of a person during {{}}.',
            f'a example of a person performing {{}}.',
            f'a example of a person practicing {{}}.',
            f'a demonstration of {{}}.',
            f'a demonstration of a person {{}}.',
            f'a demonstration of a person using {{}}.',
            f'a demonstration of a person doing {{}}.',
            f'a demonstration of a person during {{}}.',
            f'a demonstration of a person performing {{}}.',
            f'a demonstration of a person practicing {{}}.',
            f'{{}}'
        ]
        text_dict = {}
        
        id2cls = {}
        temp_mapping = json.load(open(data_file, 'r'))
        for key in temp_mapping:
            id2cls[int(key)] = temp_mapping[key]

        cls_num = len(id2cls)
        if self.training:
            index = random.randint(0, len(text_aug)-2)
            text_aug = [text_aug[index], text_aug[-1]]

        for idx, txt in enumerate(text_aug):
            if idx == len(text_aug)-1:
                text_dict[idx] = torch.cat([tokenize(txt.format(id2cls[id])) for id in range(cls_num)])
            else:
                text_dict[idx] = torch.cat([tokenize(txt.format(id2cls[id].split(':')[0]) + ' ' + id2cls[id]) for id in range(cls_num)])

        return text_dict
        
    def achieve_csf_matrix(self, text_dict, model, trainable=False):
        if not trainable:
            with torch.no_grad():
                csf_matrix_list = [model.encode_text(text_dict[i].to(self.device)).detach() for i in range(len(text_dict))]
                for csf_matrix in csf_matrix_list:
                    csf_matrix = csf_matrix / csf_matrix.norm(dim=-1, keepdim=True)
        else:
            csf_matrix_list = [model.encode_text(text_dict[i].to(self.device)) for i in range(len(text_dict))]
            for csf_matrix in csf_matrix_list:
                csf_matrix = csf_matrix / csf_matrix.norm(dim=-1, keepdim=True)
        
        csf_matrix = torch.stack(csf_matrix_list, 0).mean(0)
        csf_matrix = csf_matrix / csf_matrix.norm(dim=-1, keepdim=True)
        
        return csf_matrix

class WCLIP(CLIP):
    def __init__(self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,

        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,

        # video
        T=8,
        temporal_modeling_type=None,

        # other
        use_checkpoint=False,
        num_experts=0,
        expert_insert_layers=[],
        record_routing=False,
        routing_type = 'patch-level'
    ):
        super().__init__(
                embed_dim,
                image_resolution, vision_layers, vision_width, vision_patch_size, #vision
                context_length, vocab_size, transformer_width, transformer_heads, transformer_layers #text
            )
        
        self.vision_width = vision_width
        vision_heads = vision_width // 64
        self.visual = TemporalVisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            T=T,
            temporal_modeling_type=temporal_modeling_type,
            use_checkpoint=use_checkpoint,
            num_experts=num_experts,
            expert_insert_layers=expert_insert_layers,
            record_routing = record_routing,
            routing_type = routing_type,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(max(self.context_length, 77), transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.embed_dim = embed_dim
        self.initialize_parameters()
        self.temporal_modeling_type = temporal_modeling_type
        
    # ignore. copy from videoX
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}
    
    def encode_image(self, image, maeout=None):
        if maeout is not None:
            maskf = maeout[0]
            mask = maeout[1]
        else:
            maskf, mask = None, None
        return self.visual([image.type(self.dtype), [maskf, mask]])

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def prompt_encode_text(self, prompts, tokenized_prompts,):
        prompts = prompts.type(self.dtype)
        x = prompts + self.positional_embedding.type(self.dtype)[:self.context_length, :]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x 

def build_model(
        state_dict: dict, 
        T=8, 
        temporal_modeling_type=None, 
        use_checkpoint=False,
        context_length=None, 
        num_experts=0, 
        expert_insert_layers=[], 
        record_routing=False, 
        routing_type='patch-level'
    ):

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    
    else:
        raise NotImplementedError
    
    embed_dim = state_dict["text_projection"].shape[1]
    if context_length:
        context_length = context_length
    else:
        context_length = state_dict["positional_embedding"].shape[0]

    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64

    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    model = WCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size, #vision
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, #text
        T=T, temporal_modeling_type=temporal_modeling_type, #video
        use_checkpoint=use_checkpoint, num_experts=num_experts, #other
        expert_insert_layers=expert_insert_layers,
        record_routing=record_routing,
        routing_type=routing_type,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    
    if num_experts > 0:
        for key in list(state_dict.keys()):
            if 'mlp' in key and key.startswith('visual'):
                for expert_id in range(num_experts):
                    if 'c_fc' in key or 'gelu' in key:
                        new_key = key.replace('mlp', 'experts_head.%d'%expert_id)
                    else:
                        new_key = key.replace('mlp', 'experts_tail.%d'%expert_id)
                    state_dict[new_key] = state_dict[key]
    
    return model.eval()

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def load(name: str, 
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "mps",
        jit:bool = False, 
        download_root: str = None, 
        T=8, 
        temporal_modeling_type=None, 
        use_checkpoint=False, 
        context_length = 77, 
        num_experts=0, 
        expert_insert_layers=[], 
        record_routing=False, 
        routing_type='patch-level'
    ):
    
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found.")

    try:
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")
     
    model = build_model(
        state_dict or model.state_dict(), 
        T=T, 
        temporal_modeling_type=temporal_modeling_type, 
        use_checkpoint=use_checkpoint, 
        context_length = context_length,
        num_experts=num_experts, 
        expert_insert_layers=expert_insert_layers,
        record_routing=record_routing, 
        routing_type=routing_type
    ).to(device)

    return model, _transform(model.visual.input_resolution)