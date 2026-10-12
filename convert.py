"""Fixed-shape Core ML adaptation of Transformers EmbeddingGemma2 (Apache-2.0).
Original trained model/head: mrm8488. No training or new model architecture.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer
from runtime import load_upstream

SOURCE = "mrm8488/embeddinggemma-2-ft-typed-decision"
REVISION = "09fc4497a95529ec66842caac291e21c62490fab"

class StableNorm(nn.Module):
    """Algebraic max-abs rescaling avoids FP16 square overflow."""
    def __init__(self, original):
        super().__init__()
        self.eps = original.eps
        self.weight = original.weight if original.with_scale else None
    def forward(self, x):
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        y = x / scale
        z = y * torch.rsqrt((y*y).mean(dim=-1, keepdim=True) + (self.eps / scale) / scale)
        return z if self.weight is None else z * self.weight

def stabilize(module):
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2RMSNorm
    for name, child in list(module.named_children()):
        if isinstance(child, EmbeddingGemma2RMSNorm):
            setattr(module, name, StableNorm(child))
        else:
            stabilize(child)

def rotate(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)

class FixedEncoder(nn.Module):
    def __init__(self, text_model, size):
        super().__init__()
        if size > 512:
            raise ValueError("This adaptation only supports <=512, where all local windows cover the sequence")
        self.size = size
        self.layers = text_model.layers
        self.ple = text_model.ple
        self.norm = text_model.norm
        self.projection = text_model.embedding_projection
        cfg = text_model.config
        self.types = cfg.layer_types
        self.rope = nn.ParameterDict()
        for typ in sorted(set(self.types)):
            inv = getattr(text_model.rotary_emb, typ + "_inv_freq")
            angle = torch.arange(size).float()[:,None] * inv[None,:]
            angle = torch.cat((angle,angle),dim=-1)
            factor = getattr(text_model.rotary_emb, typ + "_attention_scaling")
            self.rope[typ+"_cos"] = nn.Parameter(angle.cos()*factor,requires_grad=False)
            self.rope[typ+"_sin"] = nn.Parameter(angle.sin()*factor,requires_grad=False)
    def forward(self, embeddings, attention_mask):
        x = embeddings
        # Projection-only PLE; split first to avoid rank-4 24-layer intermediate.
        p = self.ple.per_layer_model_projection(embeddings) * self.ple.per_layer_model_projection_scale
        parts = p.split(512,dim=-1)
        bias = (1-attention_mask.float()).reshape(1,1,self.size) * -10000.0
        for i, layer in enumerate(self.layers):
            per = self.ple.per_layer_projection_norm(parts[i])
            h = layer.input_layernorm(x)
            a = layer.self_attn
            d = a.head_dim
            qs = a.q_proj(h).split(d,dim=-1)
            ks = a.k_proj(h).split(d,dim=-1)
            vs = a.v_proj(h).split(d,dim=-1)
            cos = self.rope[self.types[i]+"_cos"]
            sin = self.rope[self.types[i]+"_sin"]
            heads = []
            for j,q in enumerate(qs):
                k = a.k_norm(ks[j//a.num_key_value_groups])
                v = a.v_norm(vs[j//a.num_key_value_groups])
                q = a.q_norm(q)
                q = q*cos + rotate(q)*sin
                k = k*cos + rotate(k)*sin
                scores = q @ k.transpose(-1,-2) + bias
                probs = F.softmax(scores,dim=-1)
                heads.append(probs @ v)
            attended = a.o_proj(torch.cat(heads,dim=-1))
            x = x + layer.post_attention_layernorm(attended)
            x = x + layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(x)))
            x = layer.ple_block(x,per) * layer.layer_scalar
        x = self.norm(x)
        mask = attention_mask.float().unsqueeze(-1)
        pool_weights = mask / mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (x*pool_weights).sum(dim=1)
        out = self.projection(pooled)
        scaled = out / out.abs().amax(dim=-1,keepdim=True).clamp(min=1e-4)
        return scaled / torch.sqrt((scaled*scaled).sum(dim=-1,keepdim=True)).clamp(min=1e-4)

def reference(directory):
    upstream = load_upstream(directory)
    return upstream.GemmaDecisionModel.from_pretrained(directory,local_files_only=True,dtype=torch.float32).eval()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts",type=Path,default=Path("artifacts"))
    parser.add_argument("--size",type=int,default=512)
    args=parser.parse_args()
    torch.set_num_threads(4)
    root=args.artifacts
    model=reference(root/"upstream")
    tokenizer=AutoTokenizer.from_pretrained(root/"upstream",local_files_only=True, trust_remote_code=False)
    tokenizer.save_pretrained(root/"tokenizer")
    np.save(root/"embeddings.npy",model.get_input_embeddings().weight.detach().numpy())
    save_file({k:v.contiguous() for k,v in model.state_dict().items() if k.startswith("head.") or k=="temperatures"},root/"head.safetensors")
    text=model.encoder.language_model
    wrapper=FixedEncoder(text,args.size).eval()
    batch=tokenizer("task: classification | query: This is a conversion sanity check.",padding="max_length",max_length=args.size,return_tensors="pt")
    embeds=model.get_input_embeddings()(batch["input_ids"]).detach()
    mask=batch["attention_mask"].to(torch.int32)
    with torch.no_grad():
        expected=model.embed(batch["input_ids"],batch["attention_mask"])
        actual=wrapper(embeds,mask)
    cosine=F.cosine_similarity(expected,actual).item()
    error=(expected-actual).abs().max().item()
    print(json.dumps({"wrapper_cosine":cosine,"wrapper_max_abs_error":error}),flush=True)
    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all() or not np.isfinite([cosine,error]).all() or cosine<0.99999 or error>1e-4:
        raise RuntimeError("FP32 wrapper failed parity; refusing export")
    stabilize(wrapper)
    with torch.no_grad():
        stable=wrapper(embeds,mask)
    if not torch.isfinite(stable).all() or (stable-expected).abs().max().item()>1e-4:
        raise RuntimeError("Stable normalization failed FP32 parity")
    print("Tracing fixed encoder",flush=True)
    traced=torch.jit.trace(wrapper,(embeds,mask),check_trace=False)
    import coremltools as ct
    print("Converting Core ML",flush=True)
    converted=ct.convert(traced,inputs=[ct.TensorType(name="embeddings",shape=embeds.shape,dtype=np.float32),ct.TensorType(name="attention_mask",shape=mask.shape,dtype=np.int32)],outputs=[ct.TensorType(name="sentence_embedding",dtype=np.float32)],minimum_deployment_target=ct.target.macOS15,compute_precision=ct.precision.FLOAT16,compute_units=ct.ComputeUnit.CPU_AND_NE,skip_model_load=True)
    converted.short_description="Core ML adaptation of mrm8488 EmbeddingGemma 2 typed decisions encoder (no training)."
    converted.author="Model: mrm8488; base: Google; adaptation: mikolaj92"
    converted.license="Apache-2.0"
    converted.save(str(root/"Encoder.mlpackage"))
    (root/"manifest.json").write_text(json.dumps({"source":SOURCE,"source_revision":REVISION,"max_tokens":args.size,"buckets":[args.size],"compute_units":"CPU_AND_NE","ane_placement_verified":False},indent=2)+"\n")
    print("Export complete. Neural Engine placement and numerical parity still require validation.",flush=True)
if __name__=="__main__": main()
