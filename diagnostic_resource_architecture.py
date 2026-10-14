import json,torch
from pathlib import Path
from convert import reference
torch.set_num_threads(2)
m=reference(Path("artifacts-8192/upstream")).cpu().float().eval()
storages={}
for t in list(m.parameters())+list(m.buffers()):
 storages[t.untyped_storage().data_ptr()]=t.untyped_storage().nbytes()
rows=[]
for i,l in enumerate(m.encoder.language_model.layers):
 a=l.self_attn
 rows.append({"index":i,"type":m.encoder.language_model.config.layer_types[i],"q_width":a.q_proj.out_features,"kv_width":a.k_proj.out_features,"head_dim":a.head_dim,"groups":a.num_key_value_groups,"mlp_width":l.mlp.up_proj.out_features})
print(json.dumps({"unique_weight_buffer_bytes":sum(storages.values()),"layers":rows,"native_limit":m.config.max_seq_length,"positional_limit":m.encoder.language_model.config.max_position_embeddings},indent=2))
