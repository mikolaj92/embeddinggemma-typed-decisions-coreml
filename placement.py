"""Inspect Core ML's anticipated per-operation device plan, not a hardware trace."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
from time import perf_counter

from integrity import manifest_packages, model_sha256, package_sha256


def inspect_package(package):
    """Compile/load a selected package and inspect its anticipated device plan."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    started = perf_counter()
    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled_path = model.get_compiled_model_path()
    compile_seconds = perf_counter() - started
    plan = MLComputePlan.load_from_path(compiled_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = Counter()
    ops = []
    constants = 0

    def walk(block):
        nonlocal constants
        for op in block.operations:
            if op.operator_name == "const":
                constants += 1
            else:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                preferred = usage.preferred_compute_device if usage else None
                device = type(preferred).__name__ if preferred is not None else "unknown"
                counts[device] += 1
                if "NeuralEngine" not in device:
                    ops.append({"operator": op.operator_name, "device": device})
            for nested in op.blocks:
                walk(nested)

    for function in plan.model_structure.program.functions.values():
        walk(function.block)
    return {
        "package_sha256": model_sha256(package),
        "preferred_devices": dict(sorted(counts.items())),
        "non_ane_operations": ops,
        "total_operations": sum(counts.values()),
        "constants_excluded": constants,
        "compile_seconds": compile_seconds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = ap.parse_args()
    root = args.artifacts.resolve()
    models = {}
    counts = Counter()
    ops = []
    for bucket, package in manifest_packages(root):
        result = inspect_package(package)
        result["bucket"] = bucket
        result["package"] = package.relative_to(root).as_posix()
        models[str(bucket) if bucket is not None else "default"] = result
        counts.update(result["preferred_devices"])
        ops.extend(dict(op, bucket=bucket) for op in result["non_ane_operations"])
    report = {
        "host": platform.node(),
        "compute_units": "CPU_AND_NE",
        "measurement": "Core ML anticipated compute plan (not execution trace)",
        "ane_placement_verified": False,
        "compile_time_measurement": "MLModel load/compile wall time; excludes compute-plan inspection",
        "encoder_sha256": package_sha256(root),
        "models": models,
        "preferred_devices": dict(sorted(counts.items())),
        "non_ane_operations": ops,
        "total_operations": sum(counts.values()),
        "constants_excluded": sum(item["constants_excluded"] for item in models.values()),
        "compile_seconds": sum(item["compile_seconds"] for item in models.values()),
    }
    encoded = json.dumps(report, indent=2)
    print(encoded, flush=True)
    (root / "placement.json").write_text(encoded + "\n")


if __name__ == "__main__":
    main()
