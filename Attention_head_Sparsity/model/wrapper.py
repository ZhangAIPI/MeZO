import itertools as I
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)


from data import CacheDataset
import math

logger = logging.getLogger(__name__)


class PrunableLlamaAttention(torch.nn.Module):
    def __init__(
        self,
        model,
        r=None,
    ):
        super().__init__()
        assert isinstance(model, LlamaAttention)
        self.model = model

        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.model.q_proj(hidden_states)
        key_states = self.model.k_proj(hidden_states)
        value_states = self.model.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.model.num_heads, self.model.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.model.num_key_value_heads, self.model.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.model.num_key_value_heads, self.model.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.model.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.model.layer_idx
            )
        cos, sin = self.model.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.model.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.model.num_key_value_groups)
        value_states = repeat_kv(value_states, self.model.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.model.head_dim)

        if attn_weights.size() != (bsz, self.model.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.model.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.model.attention_dropout, training=self.model.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (
            bsz,
            self.model.num_heads,
            q_len,
            self.model.head_dim,
        ):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.model.num_heads, q_len, self.model.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.model.hidden_size)

        self.cache_X = attn_output.clone()
        attn_output = self.model.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        self.cache_Z = attn_output.clone()

        return attn_output, attn_weights, past_key_value

    @torch.no_grad()
    def enumerate(self):
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        loss_history = dict()

        with torch.inference_mode():
            for dropped in I.combinations(
                range(self.model.num_experts), self.model.num_experts - self.r
            ):
                self.experts_to_drop = dropped
                loss = 0

                for hidden_states, final_hidden_states in zip(
                    self.cache_space.Xs, self.cache_space.Zs
                ):
                    hidden_states = hidden_states.to(
                        device=self.model.gate.weight.data.device, non_blocking=True
                    )
                    final_hidden_states = final_hidden_states.to(
                        dtype=torch.float64,
                        device=self.model.gate.weight.data.device,
                        non_blocking=True,
                    )

                    final_hidden_states_e, _ = self.forward(hidden_states.unsqueeze(0))
                    loss += torch.norm(
                        final_hidden_states
                        - final_hidden_states_e.squeeze(0).to(torch.float64)
                    ).item()
                loss_history[dropped] = loss

        self.experts_to_drop = min(loss_history, key=loss_history.get)
        return loss_history

    @torch.no_grad()
    def prune(self):
        assert self.experts_to_drop is not None
        assert len(self.experts_to_drop) == self.model.num_experts - self.r
        del self.cache_space
        self.cache_X = False
        self.cache_Z = False

        experts_to_reserve = sorted(
            set(range(self.model.num_experts)) - set(self.experts_to_drop)
        )

        gate_new = torch.nn.Linear(
            in_features=self.model.gate.in_features,
            out_features=self.r,
            bias=False,
            device="cpu",
            dtype=torch.bfloat16,
        )
        gate_new.weight.data = self.model.gate.weight.data[list(experts_to_reserve)]
        self.model.gate = gate_new

        self.model.experts = torch.nn.ModuleList(
            [self.model.experts[i] for i in experts_to_reserve]
        )
        self.model.num_experts = self.r
