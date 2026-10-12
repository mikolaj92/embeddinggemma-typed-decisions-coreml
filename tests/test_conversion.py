"""Stable normalization matches independent FP32 RMSNorm, including overflow inputs."""
import torch
from types import SimpleNamespace
from convert import StableNorm, FixedEncoder

def test_stable_norm_large_zero_and_small_values_match_fp32():
    x=torch.tensor([[0.,0.,0.,0.],[1e-7,-2e-7,0.,1e-7],[2000.,-1000.,1500.,0.]],dtype=torch.float32)
    norm=StableNorm(SimpleNamespace(eps=1e-6,with_scale=True,weight=torch.nn.Parameter(torch.tensor([1.,2.,.5,1.]))))
    expected=x / torch.sqrt((x*x).mean(-1,keepdim=True)+1e-6) * norm.weight
    actual=norm(x.half())
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(),expected,atol=.005,rtol=.005)

def test_fixed_encoder_long_context_matches_transformers_at_window_boundaries():
    from transformers import EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel
    torch.manual_seed(23)
    config=EmbeddingGemma2TextConfig(hidden_size=512,intermediate_size=32,num_hidden_layers=2,num_attention_heads=1,num_key_value_heads=1,head_dim=32,vocab_size=20,layer_types=["sliding_attention","full_attention"],per_layer_config={"1":{"head_dim":32,"num_key_value_heads":1}},sliding_window=512)
    model=EmbeddingGemma2TextModel(config).eval()
    wrapper=FixedEncoder(model,1024).eval()
    inputs=torch.randn(1,1024,512)
    for valid in [64,513,514,1024]:
        mask=torch.zeros(1,1024,dtype=torch.int32); mask[:,:valid]=1
        with torch.inference_mode():
            tokens=model(inputs_embeds=inputs,attention_mask=mask).last_hidden_state
            pooled=(tokens*mask.unsqueeze(-1)).sum(1)/valid
            expected=torch.nn.functional.normalize(pooled,dim=-1)
            actual=wrapper(inputs,mask)
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)
