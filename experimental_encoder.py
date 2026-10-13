"""Isolated exact-attention pilot; NOT a production exporter or context approval.

Original Transformers EmbeddingGemma2 architecture (Apache-2.0, HuggingFace),
with FixedEncoder operation order. Query tiling bounds attention intermediates,
not global attention FLOPs. PLE is deliberately unchanged and can still dominate
memory. Original absolute RoPE frequencies are retained without context scaling.
16k -> 32k -> 64k -> 128k require separate quality/resource/hardware gates;
constructor support and tiny-model parity do not prove 128k or authorize rollout.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


EXPERIMENTAL_SIZES = tuple(2**power for power in range(9, 18))


def rotate(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class TiledEncoder(nn.Module):
    """Fixed-length pilot, query tiles 256 (default) or 512, no square mask.

    Full layers attend to every K/V. Local layers attend to absolute distance
    <=512, including both endpoints. Empty local neighborhoods produce zero
    attention, rather than an arbitrary distribution over padding. Masked keys
    cannot affect valid rows. Inputs are embeddings and a binary padding mask.
    Modules/weights are shared with the supplied text model; no model config,
    tokenizer, production artifact, service or decision head is modified.
    """

    def __init__(self, text_model, size, query_tile=256):
        super().__init__()
        if size not in EXPERIMENTAL_SIZES:
            raise ValueError("Experimental fixed shapes: powers of two from 512 to 131072")
        if query_tile not in (256, 512):
            raise ValueError("Experimental query tiles: 256 or 512")
        if text_model.config.sliding_window != 512:
            raise ValueError("This pilot preserves the original sliding window of 512")
        if text_model.config.hidden_size_per_layer_input != 512:
            raise ValueError("This pilot preserves the original 512-wide PLE")
        self.size = size
        self.query_tile = query_tile
        self.layers = text_model.layers
        self.ple = text_model.ple
        self.norm = text_model.norm
        self.projection = text_model.embedding_projection
        self.types = tuple(text_model.config.layer_types)
        self.register_buffer("positions", torch.arange(size))
        self.rope = nn.ParameterDict()
        for typ in sorted(set(self.types)):
            inv = getattr(text_model.rotary_emb, typ + "_inv_freq")
            angle = torch.arange(size).float()[:, None] * inv[None, :]
            angle = torch.cat((angle, angle), dim=-1)
            factor = getattr(text_model.rotary_emb, typ + "_attention_scaling")
            self.rope[typ + "_cos"] = nn.Parameter(angle.cos() * factor, requires_grad=False)
            self.rope[typ + "_sin"] = nn.Parameter(angle.sin() * factor, requires_grad=False)

    def forward(self, embeddings, attention_mask):
        x = embeddings
        # Keep the original all-layer PLE projection for the initial pilot.
        p = self.ple.per_layer_model_projection(embeddings) * self.ple.per_layer_model_projection_scale
        parts = p.split(512, dim=-1)
        bias = (1 - attention_mask.float()).unsqueeze(1) * -10000.0
        for i, layer in enumerate(self.layers):
            per = self.ple.per_layer_projection_norm(parts[i])
            h = layer.input_layernorm(x)
            a = layer.self_attn
            d = a.head_dim
            qs = a.q_proj(h).split(d, dim=-1)
            ks = a.k_proj(h).split(d, dim=-1)
            vs = a.v_proj(h).split(d, dim=-1)
            cos = self.rope[self.types[i] + "_cos"]
            sin = self.rope[self.types[i] + "_sin"]
            local = self.types[i] == "sliding_attention"
            heads = []
            for j, q in enumerate(qs):
                k = a.k_norm(ks[j // a.num_key_value_groups])
                v = a.v_norm(vs[j // a.num_key_value_groups])
                q = a.q_norm(q)
                q = q * cos + rotate(q) * sin
                k = k * cos + rotate(k) * sin
                tiles = []
                for s in range(0, self.size, self.query_tile):
                    e = min(self.size, s + self.query_tile)
                    lo = max(0, s - 512) if local else 0
                    hi = min(self.size, e + 512) if local else self.size
                    layer_bias = bias[:, :, lo:hi]
                    allowed = attention_mask[:, None, lo:hi] != 0
                    if local:
                        distance = (self.positions[s:e, None] - self.positions[None, lo:hi]).abs()
                        in_window = distance <= 512
                        layer_bias = layer_bias + (~in_window).float().unsqueeze(0) * -10000.0
                        allowed = allowed & in_window.unsqueeze(0)
                    # EmbeddingGemma2 scaling is 1.0: NO 1/sqrt(head_dim).
                    scores = q[:, s:e] @ k[:, lo:hi].transpose(-1, -2) + layer_bias
                    # Finite empty rows are made harmless before softmax. The
                    # explicit key mask also prevents padded states from leaking
                    # into valid rows in subsequent global attention layers.
                    scores = torch.where(allowed.any(dim=-1, keepdim=True), scores, torch.zeros_like(scores))
                    probs = F.softmax(scores, dim=-1) * allowed.to(scores.dtype)
                    tiles.append(probs @ v[:, lo:hi])
                heads.append(torch.cat(tiles, dim=1))
            attended = a.o_proj(torch.cat(heads, dim=-1))
            x = x + layer.post_attention_layernorm(attended)
            x = x + layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(x)))
            x = layer.ple_block(x, per) * layer.layer_scalar
        x = self.norm(x)
        mask = attention_mask.float().unsqueeze(-1)
        pool_weights = mask / mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (x * pool_weights).sum(dim=1)
        out = self.projection(pooled)
        scaled = out / out.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        return scaled / torch.sqrt((scaled * scaled).sum(dim=-1, keepdim=True)).clamp(min=1e-4)
