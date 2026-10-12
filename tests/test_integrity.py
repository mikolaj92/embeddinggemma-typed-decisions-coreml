"""Validation hashes bind selected packages and bucket assignments, not unused files."""
import json
import pytest
from integrity import package_sha256

def test_multi_bucket_integrity_detects_selected_mutation_and_bad_mapping(tmp_path):
    for name in ["a.mlpackage","b.mlpackage","unused.mlpackage"]:
        package=tmp_path/name; package.mkdir(); (package/"weights").write_bytes(name.encode())
    manifest=tmp_path/"manifest.json"
    spec={"buckets":[512,1024],"models":{"512":"a.mlpackage","1024":"b.mlpackage"}}
    manifest.write_text(json.dumps(spec))
    original=package_sha256(tmp_path)
    (tmp_path/"unused.mlpackage/weights").write_bytes(b"new")
    assert package_sha256(tmp_path)==original
    spec["buckets"]=[1024,512]; manifest.write_text(json.dumps(spec))
    assert package_sha256(tmp_path)==original
    (tmp_path/"b.mlpackage/weights").write_bytes(b"new")
    assert package_sha256(tmp_path)!=original
    spec["models"].pop("1024"); manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match="missing model"):
        package_sha256(tmp_path)
