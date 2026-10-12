"""Bind validation evidence to the exact manifest-selected encoder packages."""
import hashlib
import json
from pathlib import Path


def manifest_packages(root):
    """Return (bucket, package path) pairs in numeric bucket order.

    A legacy artifact directory without a manifest uses the single Encoder
    package and a None bucket. Only a single bucket may omit its models entry.
    """
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    buckets = manifest.get("buckets", [None])
    if (not isinstance(buckets, list) or not buckets
            or (buckets != [None] and any(type(n) is not int or n <= 0 for n in buckets))
            or len(set(buckets)) != len(buckets)):
        raise ValueError("invalid manifest buckets")
    models = manifest.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("manifest models must be a mapping")
    if buckets == [None] and models:
        raise ValueError("models mapping requires manifest buckets")
    selected = []
    for bucket in sorted(buckets):
        filename = models.get(str(bucket))
        if filename is None:
            if len(buckets) != 1 or str(bucket) in models:
                raise ValueError(f"missing model package for bucket {bucket}")
            filename = "Encoder.mlpackage"
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"invalid model package for bucket {bucket}")
        package = (root / filename).resolve()
        if not package.is_relative_to(root):
            raise ValueError("model path escapes artifact directory")
        if not package.is_dir():
            raise FileNotFoundError(f"model package does not exist: {package}")
        selected.append((bucket, package))
    return selected


def model_sha256(package):
    """Hash one package using the original relative-filename/content algorithm."""
    package = Path(package)
    if not package.is_dir():
        raise FileNotFoundError(f"model package does not exist: {package}")
    paths = sorted(p for p in package.rglob("*") if p.is_file())
    if not paths:
        raise ValueError(f"model package is empty: {package}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def package_sha256(root):
    """Hash exactly the selected packages, preserving the single-package digest.

    Multi-bucket hashes also bind bucket-to-package assignments and relative
    package names; unselected packages and mapping insertion order do not matter.
    """
    root = Path(root).resolve()
    packages = manifest_packages(root)
    if len(packages) == 1:
        return model_sha256(packages[0][1])
    digest = hashlib.sha256(b"manifest-model-packages-v1\0")
    for bucket, package in packages:
        record = [bucket, package.relative_to(root).as_posix(), model_sha256(package)]
        digest.update(json.dumps(record, separators=(",", ":"), ensure_ascii=True).encode())
        digest.update(b"\0")
    return digest.hexdigest()
