"""Independent Transformers parity, padding isolation and allocation contracts.

Run on the mini's existing environment: uv run pytest -q tests/test_experimental_encoder.py
Tiny-model parity is not trained long-context quality or Core ML hardware proof.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel

from experimental_encoder import TiledEncoder


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(37)
    config = EmbeddingGemma2TextConfig(
        hidden_size=512, hidden_size_per_layer_input=512,
        intermediate_size=32, embedding_dim=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=20, sliding_window=512,
        layer_types=["sliding_attention", "full_attention"],
        per_layer_config={"1": {"head_dim": 32, "num_key_value_heads": 1}},
    )
    config._attn_implementation = "eager"
    return EmbeddingGemma2TextModel(config).eval()


def mask_for(case):
    mask = torch.zeros(1, 1024, dtype=torch.int32)
    if isinstance(case, int):
        mask[:, :case] = 1
    elif case == "holes":
        mask[:] = 1
        # Holes on query seams and inclusive/exclusive local window endpoints.
        mask[:, [0, 1, 255, 256, 257, 511, 512, 513, 767, 768, 1023]] = 0
    elif case == "sparse_empty_tail":
        mask[:, [0, 255, 256, 511]] = 1
    else:
        raise AssertionError(case)
    return mask


def capture_attention(model):
    values = []
    handles = []
    for layer in model.layers:
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
            lambda module, args: values.append(args[0].detach().clone())
        ))
    return values, handles


@pytest.mark.parametrize("query_tile", [256, 512])
@pytest.mark.parametrize("case", [64, 513, 514, 1024, "holes", "sparse_empty_tail"])
def test_tiled_encoder_matches_independent_transformers_at_seams_and_windows(tiny_model, query_tile, case):
    # Protects scaling=1, GQA grouping, per-layer head sizes, RoPE positions,
    # inclusive window edges, global K/V, PLE order and mean/projection/L2.
    generator = torch.Generator().manual_seed(71)
    inputs = torch.randn(1, 1024, 512, generator=generator)
    mask = mask_for(case)
    wrapper = TiledEncoder(tiny_model, 1024, query_tile=query_tile).eval()
    reference_heads, handles = capture_attention(tiny_model)
    try:
        with torch.inference_mode():
            tokens = tiny_model(inputs_embeds=inputs, attention_mask=mask).last_hidden_state
            expected = torch.nn.functional.normalize(
                (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True), dim=-1
            )
    finally:
        for handle in handles:
            handle.remove()
    actual_heads, handles = capture_attention(tiny_model)
    try:
        with torch.inference_mode():
            actual = wrapper(inputs, mask)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    valid = mask.bool()
    for actual_head, reference_head in zip(actual_heads, reference_heads, strict=True):
        assert torch.isfinite(actual_head).all()
        # Token-level comparison prevents pooling from hiding seam/window errors.
        torch.testing.assert_close(actual_head[valid], reference_head[valid], atol=1e-5, rtol=1e-4)
    if case == 64:
        # Local query576 is 513 away from the last valid key63. Empty padded
        # rows must be safe zero attention, not softmax over padded keys.
        assert torch.count_nonzero(actual_heads[0][:, 576:]) == 0
        perturbed = inputs.clone()
        perturbed[:, 64:] = torch.randn(1, 960, 512, generator=generator) * 10
        with torch.inference_mode():
            isolated = wrapper(perturbed, mask)
        torch.testing.assert_close(isolated, actual, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("query_tile", [256, 512])
def test_entirely_masked_input_has_finite_zero_embedding(tiny_model, query_tile):
    wrapper = TiledEncoder(tiny_model, 1024, query_tile=query_tile).eval()
    with torch.inference_mode():
        actual = wrapper(torch.ones(1, 1024, 512), torch.zeros(1, 1024, dtype=torch.int32))
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) == 0


class AllocationCeiling(TorchDispatchMode):
    """Observe *temporary* allocations too, not just surviving buffers."""
    def __init__(self, max_elements):
        super().__init__()
        self.max_elements = max_elements

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        def check(value):
            if isinstance(value, torch.Tensor):
                assert value.numel() <= self.max_elements, (
                    f"Constructor allocated {tuple(value.shape)} exceeding linear budget"
                )
            elif isinstance(value, (tuple, list)):
                for item in value:
                    check(item)
        check(result)
        return result


@pytest.mark.parametrize("size", [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
def test_experimental_constructor_has_linear_allocation_budget(size):
    # This catches a transient NxN distance tensor even if never registered.
    with torch.device("meta"), AllocationCeiling(size * 32):
        model = SimpleNamespace(
            layers=nn.ModuleList(), ple=nn.Identity(), norm=nn.Identity(),
            embedding_projection=nn.Identity(),
            config=SimpleNamespace(
                layer_types=["sliding_attention", "full_attention"],
                sliding_window=512, hidden_size_per_layer_input=512,
            ),
            rotary_emb=SimpleNamespace(
                sliding_attention_inv_freq=torch.ones(8),
                sliding_attention_attention_scaling=1.0,
                full_attention_inv_freq=torch.ones(16),
                full_attention_attention_scaling=1.0,
            ),
        )
        wrapper = TiledEncoder(model, size)
    tensors = list(wrapper.buffers()) + list(wrapper.parameters())
    assert sum(tensor.numel() for tensor in tensors) <= size * 97


@pytest.mark.parametrize("size,query_tile", [(256, 256), (513, 256), (262144, 256), (1024, 128)])
def test_experimental_shape_and_tile_rejected_before_model_access(size, query_tile):
    with pytest.raises(ValueError, match="Experimental"):
        TiledEncoder(None, size, query_tile=query_tile)
