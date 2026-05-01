import os
import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

warnings.filterwarnings('ignore')


class VLMConfig(MiniMindConfig):
    model_type = "minichat-cv"

    def __init__(
            self,
            image_special_token: str = '@',
            image_ids: Optional[List] = None,
            max_seq_len: Optional[int] = None,
            alignment_type: str = 'cross_attn',
            cross_attn_layers: Optional[Union[str, List[int], Tuple[int, ...]]] = None,
            cross_attn_every: int = 2,
            cross_attn_start_layer: Optional[int] = None,
            cross_attn_gate_init: float = 0.1,
            vision_hidden_size: int = 768,
            **kwargs,
    ):
        self.image_special_token = image_special_token
        super().__init__(
            alignment_type=alignment_type,
            cross_attn_layers=cross_attn_layers,
            cross_attn_every=cross_attn_every,
            cross_attn_start_layer=cross_attn_start_layer,
            cross_attn_gate_init=cross_attn_gate_init,
            **kwargs
        )
        self.image_ids = image_ids if image_ids is not None else [34]
        self.max_seq_len = max_seq_len if max_seq_len is not None else self.max_position_embeddings
        self.vision_hidden_size = vision_hidden_size

class VisionProj(nn.Module):
    def __init__(self, ve_hidden_size=768, hidden_size=512):
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

    def forward(self, image_encoders):
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj


# 继承
class MiniMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model_path="./model/vision_model/clip-vit-base-patch16"):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        self.vision_proj = VisionProj(
            ve_hidden_size=getattr(params, 'vision_hidden_size', 768),
            hidden_size=params.hidden_size
        )

    @staticmethod
    def resolve_vision_model_path(model_path: str):
        if not model_path:
            return None

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        candidates = [model_path]
        if not os.path.isabs(model_path):
            candidates.extend([
                os.path.abspath(model_path),
                os.path.abspath(os.path.join(repo_root, model_path)),
            ])
            stripped_path = model_path
            while stripped_path.startswith('../'):
                stripped_path = stripped_path[3:]
            candidates.append(os.path.abspath(os.path.join(repo_root, stripped_path)))

        seen = set()
        checked = []
        for candidate in candidates:
            candidate = os.path.abspath(candidate)
            if candidate in seen:
                continue
            seen.add(candidate)
            checked.append(candidate)
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, 'config.json')):
                return candidate

        checked_paths = '\n  - '.join(checked)
        raise FileNotFoundError(
            "CLIP vision model path is invalid. Please set --clip_path to a local "
            "clip-vit-base-patch16 directory containing config.json and preprocessor_config.json.\n"
            f"Checked:\n  - {checked_paths}"
        )

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not model_path:
            return None, None
        model_path = MiniMindVLM.resolve_vision_model_path(model_path)
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if processor is None:
            raise RuntimeError(
                "CLIP processor is not loaded. Check --clip_path and make sure it points "
                "to a local clip-vit-base-patch16 directory."
            )
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        with torch.no_grad():
            outputs = vision_model.vision_model(pixel_values=image_tensors)
        img_embedding = outputs.last_hidden_state[:, 1:, :]
        return img_embedding

    @property
    def alignment_type(self):
        return getattr(self.params, 'alignment_type', 'cross_attn')

    def encode_vision(self, pixel_values, vision_attention_mask=None):
        if pixel_values is None:
            return None, vision_attention_mask
        if self.vision_encoder is None:
            raise RuntimeError("vision_encoder is not loaded, but pixel_values were provided.")

        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)
        if pixel_values.dim() == 6 and pixel_values.size(2) == 1:
            pixel_values = pixel_values.squeeze(2)
        if pixel_values.dim() != 5:
            raise ValueError(f"pixel_values should have shape [B,N,C,H,W], got {tuple(pixel_values.shape)}")

        bs, num_images, channels, image_h, image_w = pixel_values.shape
        flat_pixels = pixel_values.reshape(bs * num_images, channels, image_h, image_w)
        with torch.no_grad():
            outputs = self.vision_encoder.vision_model(pixel_values=flat_pixels)
        patch_states = outputs.last_hidden_state[:, 1:, :]
        patch_count = patch_states.size(1)
        patch_states = patch_states.reshape(bs, num_images, patch_count, patch_states.size(-1))

        if vision_attention_mask is None:
            vision_attention_mask = torch.ones(
                bs, num_images, patch_count,
                dtype=torch.bool,
                device=patch_states.device
            )
        else:
            vision_attention_mask = vision_attention_mask.to(device=patch_states.device)
            if vision_attention_mask.dim() == 2:
                vision_attention_mask = vision_attention_mask[:, :, None].expand(bs, num_images, patch_count)
            elif vision_attention_mask.dim() == 3:
                if vision_attention_mask.size(1) == num_images and vision_attention_mask.size(2) == 1:
                    vision_attention_mask = vision_attention_mask.expand(bs, num_images, patch_count)
                elif vision_attention_mask.size(1) == num_images * patch_count:
                    vision_attention_mask = vision_attention_mask.reshape(bs, num_images, patch_count)
            if vision_attention_mask.shape != (bs, num_images, patch_count):
                raise ValueError(
                    f"vision_attention_mask should broadcast to {(bs, num_images, patch_count)}, "
                    f"got {tuple(vision_attention_mask.shape)}"
                )
            vision_attention_mask = vision_attention_mask > 0

        vision_states = patch_states.reshape(bs, num_images * patch_count, patch_states.size(-1))
        vision_attention_mask = vision_attention_mask.reshape(bs, num_images * patch_count)
        vision_states = self.vision_proj(vision_states)
        vision_states = vision_states * vision_attention_mask[:, :, None].to(dtype=vision_states.dtype)
        return vision_states, vision_attention_mask

    def _prepend_vision_tokens(self, input_ids, labels, attention_mask, past_key_values, vision_states, vision_attention_mask):
        start_pos = 0
        if past_key_values is not None and not hasattr(past_key_values, 'layers') and past_key_values[0] is not None:
            start_pos = past_key_values[0][0].shape[1]
        if start_pos != 0 or vision_states is None:
            return None, labels, attention_mask

        text_embeds = self.model.embed_tokens(input_ids)
        if vision_attention_mask is None:
            vision_attention_mask = torch.ones(
                vision_states.shape[:2],
                dtype=torch.bool,
                device=vision_states.device
            )
        inputs_embeds = torch.cat([vision_states.to(dtype=text_embeds.dtype), text_embeds], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask = torch.cat([vision_attention_mask.to(dtype=attention_mask.dtype), attention_mask], dim=1)
        if labels is not None:
            vision_labels = labels.new_full((labels.size(0), vision_states.size(1)), -100)
            labels = torch.cat([vision_labels, labels], dim=1)
        return inputs_embeds, labels, attention_mask

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                vision_attention_mask: Optional[torch.Tensor] = None,
                vision_states: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        if vision_states is None and pixel_values is not None:
            vision_states, vision_attention_mask = self.encode_vision(pixel_values, vision_attention_mask)

        inputs_embeds = None
        model_vision_states = vision_states
        model_vision_attention_mask = vision_attention_mask
        if self.alignment_type == 'token':
            inputs_embeds, labels, attention_mask = self._prepend_vision_tokens(
                input_ids, labels, attention_mask, past_key_values, vision_states, vision_attention_mask
            )
            model_vision_states = None
            model_vision_attention_mask = None

        hidden_states, presents, aux_loss = self.model(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            vision_states=model_vision_states,
            vision_attention_mask=model_vision_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)
        return output

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            pixel_values=None,
            vision_states=None,
            vision_attention_mask=None,
            **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if self.alignment_type == 'token' and vision_attention_mask is not None and attention_mask is not None:
                past_len = past_key_values[0][0].shape[1]
                expected_len = past_len + input_ids.shape[1]
                if attention_mask.shape[1] + vision_attention_mask.shape[1] == expected_len:
                    attention_mask = torch.cat([
                        vision_attention_mask.to(dtype=attention_mask.dtype, device=attention_mask.device),
                        attention_mask
                    ], dim=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "pixel_values": pixel_values,
            "vision_states": vision_states,
            "vision_attention_mask": vision_attention_mask,
        }

    def generate(self, *args, pixel_values=None, vision_attention_mask=None, vision_states=None, **kwargs):
        if vision_states is None and pixel_values is not None:
            vision_states, vision_attention_mask = self.encode_vision(pixel_values, vision_attention_mask)
            pixel_values = None
        return super().generate(
            *args,
            pixel_values=pixel_values,
            vision_states=vision_states,
            vision_attention_mask=vision_attention_mask,
            **kwargs
        )
