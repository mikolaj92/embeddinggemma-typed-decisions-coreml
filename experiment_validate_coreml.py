"""Validate an explicitly named isolated package/input; never deploys artifacts."""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package",type=Path,required=True)
    p.add_argument("--inputs",type=Path,required=True,help="npz embeddings, attention_mask and gold")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    package=a.package.resolve(); inputs=a.inputs.resolve(); output=a.output.resolve()
    if output.is_relative_to(package) or output==inputs or output.exists():
        p.error("Output must be new and separate from package and inputs")
    import numpy as np
    import coremltools as ct
    arrays=np.load(inputs,allow_pickle=False)
    buffers={name:arrays[name].copy() for name in ("embeddings","attention_mask")}
    gold=arrays["reference_embedding"] if "reference_embedding" in arrays else arrays["gold"]
    if buffers["embeddings"].ndim!=3 or buffers["embeddings"].shape[0]!=1:
        raise ValueError("Expected batch-one rank-three encoder inputs")
    if buffers["attention_mask"].shape!=buffers["embeddings"].shape[:2]:
        raise ValueError("Mask shape mismatch")
    if gold.shape!=(1,768) or not np.issubdtype(gold.dtype,np.floating) or not np.isfinite(gold).all():
        raise ValueError("Reference must be finite floating (1,768); no broadcasting permitted")
    if not np.linalg.norm(gold)>0:raise ValueError("Zero reference norm")
    report={"experimental_only":True,"quality_approved":False,"production_changed":False,"compute_units":"CPU_AND_NE","package":str(package),"input_shape":list(buffers["embeddings"].shape),"consumed_tokens":int(buffers["attention_mask"].sum()),"passed":False}
    def save():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_peak_rss_bytes"]=int(rss if sys.platform=="darwin" else rss*1024)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    report["stage"]="loading"
    save()
    try:
        start=time.perf_counter()
        model=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_AND_NE)
        report["load_seconds"]=time.perf_counter()-start
        report["stage"]="first_prediction"
        save()
        actual=model.predict(buffers)["sentence_embedding"]
        if actual.shape!=gold.shape or not np.issubdtype(actual.dtype,np.floating):
            raise RuntimeError("Actual shape/dtype mismatch; no broadcasting permitted")
        finite=bool(np.isfinite(actual).all())
        if not finite: raise RuntimeError("Non-finite embedding")
        norm=float(np.linalg.norm(actual)); gold_norm=float(np.linalg.norm(gold))
        if norm==0 or gold_norm==0: raise RuntimeError("Zero embedding norm")
        cosine=float((actual*gold).sum()/(norm*gold_norm))
        times=[]
        for _ in range(3):
            start=time.perf_counter(); current=model.predict(buffers)["sentence_embedding"]; times.append(time.perf_counter()-start)
            if current.shape!=gold.shape or not np.isfinite(current).all():raise RuntimeError("Invalid warm embedding")
            warm_norm=float(np.linalg.norm(current))
            if warm_norm<=0 or float((current*gold).sum()/(warm_norm*gold_norm))<.998:raise RuntimeError("Warm parity failed")
        report.update(cosine=cosine,max_abs_error=float(np.abs(actual-gold).max()),embedding_norm=norm,warm_seconds=times,warm_median_seconds=float(np.median(times)),passed=cosine>=.998)
        report["stage"]="complete"
        save(); print(json.dumps(report),flush=True)
        if not report["passed"]: raise RuntimeError("Experimental numerical gate failed")
    except Exception as exc:
        report.update(passed=False,error=f"{type(exc).__name__}: {exc}");save();raise

if __name__=="__main__":main()
