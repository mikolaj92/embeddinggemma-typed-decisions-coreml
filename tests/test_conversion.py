"""Stable normalization matches independent FP32 RMSNorm, including overflow inputs."""
import torch
from types import SimpleNamespace
from convert import StableNorm

def test_stable_norm_large_zero_and_small_values_match_fp32():
    x=torch.tensor([[0.,0.,0.,0.],[1e-7,-2e-7,0.,1e-7],[2000.,-1000.,1500.,0.]],dtype=torch.float32)
    norm=StableNorm(SimpleNamespace(eps=1e-6,with_scale=True,weight=torch.nn.Parameter(torch.tensor([1.,2.,.5,1.]))))
    expected=x / torch.sqrt((x*x).mean(-1,keepdim=True)+1e-6) * norm.weight
    actual=norm(x.half())
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(),expected,atol=.005,rtol=.005)
