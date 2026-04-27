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
            **kwargs,
    ):
        self.image_special_token = image_special_token
        super().__init__(**kwargs)
        self.image_ids = image_ids if image_ids is not None else [34]
        self.max_seq_len = max_seq_len if max_seq_len is not None else self.max_position_embeddings

class VisionProj(nn.Module):
    def __init__(self, ve_hidden_size=768, hidden_size=512):
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_hidden_size, self.hidden_size)
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
        self.vision_proj = VisionProj(hidden_size=params.hidden_size)

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        with torch.no_grad():
            outputs = vision_model.vision_model(pixel_values=image_tensors)
        img_embedding = outputs.last_hidden_state[:, 1:, :]
        return img_embedding

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        vision_states = None
        if pixel_values is not None and start_pos == 0:
            if self.vision_encoder is None:
                raise RuntimeError("vision_encoder is not loaded, but pixel_values were provided.")
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            bs, num, c, im_h, im_w = pixel_values.shape
            vision_tensors = torch.stack([
                MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                for i in range(num)
            ], dim=1)
            vision_tensors = vision_tensors.reshape(bs, num * vision_tensors.size(2), vision_tensors.size(3))
            vision_states = self.vision_proj(vision_tensors)

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                vision_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)
        return output
