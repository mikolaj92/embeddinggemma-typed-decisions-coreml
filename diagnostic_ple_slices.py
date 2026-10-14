import json,time,resource
from pathlib import Path
import torch
from convert import reference
torch.set_num_threads(2)
m=reference(Path("artifacts-8192/upstream")).cpu().float().eval()
for p in m.parameters():p.requires_grad_(False)
ple=m.encoder.language_model.ple
rows=[]
with torch.inference_mode():
 for n in (512,1024):
  ids=(torch.arange(n).reshape(1,n)%1000)+10
  e0=m.get_input_embeddings()(ids)
  gold=ple(e0)
  errors=[]
  for i in range(24):
   pieces=[]
   for s in range(0,n,256):
    z=torch.nn.functional.linear(e0[:,s:s+256],ple.per_layer_model_projection.weight[i*512:(i+1)*512],None)
    pieces.append(ple.per_layer_projection_norm(z*ple.per_layer_model_projection_scale))
   actual=torch.cat(pieces,dim=1)
   error=float((actual-gold[:,:,i]).abs().max());errors.append(error)
   assert torch.isfinite(actual).all() and error<=1e-5,(n,i,error)
  rows.append({"tokens":n,"layer_errors":errors,"max_abs":max(errors),"passed":True})
report={"experiment":"original_weight_sliced_PLE","scope":"projection output rows and token tiles only; not attention/quality approval","cases":rows,"process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
print(json.dumps(report,indent=2))
