"""Download the audited source/checkpoint, pinned to one commit."""
from huggingface_hub import snapshot_download
from convert import SOURCE, REVISION
if __name__ == "__main__":
    snapshot_download(SOURCE, revision=REVISION, local_dir="artifacts/upstream", allow_patterns=["*.json", "*.py", "*.safetensors", "*.jinja", "README.md"])
