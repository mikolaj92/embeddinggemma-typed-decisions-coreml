"""Inspect Core ML's anticipated per-operation device plan, not a hardware trace."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--artifacts",type=Path,default=Path("artifacts")); args=ap.parse_args()
    model=ct.models.MLModel(str(args.artifacts/"Encoder.mlpackage"),compute_units=ct.ComputeUnit.CPU_AND_NE)
    plan=MLComputePlan.load_from_path(model.get_compiled_model_path(),compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts=Counter(); ops=[]; constants=0
    def walk(block):
        nonlocal constants
        for op in block.operations:
            if op.operator_name == "const":
                constants += 1
                continue
            usage=plan.get_compute_device_usage_for_mlprogram_operation(op)
            device=type(usage.preferred_compute_device).__name__ if usage else "unknown"
            counts[device]+=1
            if "NeuralEngine" not in device: ops.append({"operator":op.operator_name,"device":device})
            for nested in op.blocks: walk(nested)
    for function in plan.model_structure.program.functions.values(): walk(function.block)
    report={"host":platform.node(),"compute_units":"CPU_AND_NE","measurement":"Core ML anticipated compute plan (not execution trace)","preferred_devices":dict(counts),"non_ane_operations":ops,"total_operations":sum(counts.values()),"constants_excluded":constants}
    print(json.dumps(report,indent=2),flush=True)
    (args.artifacts/"placement.json").write_text(json.dumps(report,indent=2)+"\n")
if __name__=="__main__": main()
