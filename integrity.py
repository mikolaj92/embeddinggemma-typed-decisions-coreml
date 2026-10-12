"""Bind validation evidence to the exact encoder package and source revision."""
import hashlib
from pathlib import Path

def package_sha256(root):
    package=Path(root)/"Encoder.mlpackage"
    digest=hashlib.sha256()
    for path in sorted(p for p in package.rglob("*") if p.is_file()):
        digest.update(path.relative_to(package).as_posix().encode()+b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
