import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionGQA(nn.Module):
    def __init__(self, hidden_size, num_heads=8, num_key_value_heads=2, gate_init=0.1):
        super().__init__()
        assert num_heads % num_key_value_heads == 0
        self.head_dim = hidden_size // num_heads
        self.n_local_heads = num_heads
        self.n_local_kv_heads = num_key_value_heads
        self.n_rep = num_heads // num_key_value_heads
        self.kv_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, self.n_local_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_local_heads * self.head_dim, hidden_size, bias=False)
        gate_init = min(max(float(gate_init), 1e-4), 1 - 1e-4)
        self.gate_alpha = nn.Parameter(torch.tensor(gate_init).logit().view(1))

    def repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if self.n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:,:, None, :, :].expand(batch, num_key_value_heads, self.n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * self.n_rep, slen, head_dim)

    def forward(self, hidden_states, vision_states, vision_attention_mask=None):
        bsz, seq_len, _ = hidden_states.shape
        vision_seq_len = vision_states.size(1)

        vision_states = self.kv_norm(vision_states)
        xq = self.q_proj(hidden_states).view(bsz, seq_len, self.n_local_heads, self.head_dim).transpose(1,2)
        xk = self.k_proj(vision_states).view(bsz, vision_seq_len, self.n_local_kv_heads, self.head_dim).transpose(1,2)
        xv = self.v_proj(vision_states).view(bsz, vision_seq_len, self.n_local_kv_heads, self.head_dim).transpose(1,2) 

        xk, xv = (
            self.repeat_kv(xk),
            self.repeat_kv(xv)
        )
        scores = torch.matmul(xq, xk.transpose(2,3)) / (self.head_dim ** 0.5)
        valid_vision_tokens = None
        if vision_attention_mask is not None:
            vision_attention_mask = vision_attention_mask.to(device=scores.device)
            if vision_attention_mask.dtype != torch.bool:
                vision_attention_mask = vision_attention_mask > 0
            valid_vision_tokens = vision_attention_mask[:, None, None, :]
            scores = scores.masked_fill(~valid_vision_tokens, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)
        if valid_vision_tokens is not None:
            attn_weights = attn_weights.masked_fill(~valid_vision_tokens, 0.0)
        attn_output = torch.matmul(attn_weights, xv)

        attn_output = attn_output.transpose(1,2).contiguous().view(bsz,seq_len,-1)
        out = self.o_proj(attn_output)
        return out * torch.sigmoid(self.gate_alpha).to(dtype=out.dtype)
