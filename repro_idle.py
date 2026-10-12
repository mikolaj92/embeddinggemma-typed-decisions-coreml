"""Hardware lifetime regression: mixed buckets with idle periods between predictions."""
import argparse
import time
from pathlib import Path
from runtime import DecisionRuntime

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--artifacts",type=Path,default=Path("artifacts-long")); ap.add_argument("--cycles",type=int,default=6); args=ap.parse_args()
    runtime=DecisionRuntime(args.artifacts)
    for i in range(args.cycles):
        for text in ["short text", " document"*700, " review"*1400]:
            runtime.encode([text])
        print(f"cycle={i} predictions_complete; idle15s",flush=True)
        time.sleep(15)
    print("idle lifecycle regression passed",flush=True)
if __name__=="__main__": main()
