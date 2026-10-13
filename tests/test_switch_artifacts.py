"""Rollout refuses incomplete parity evidence before touching the live runner."""
import json
import sys
from pathlib import Path

import pytest

import switch_artifacts


@pytest.fixture
def switch_reports(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    (artifacts / "upstream").mkdir(parents=True)
    manifest = {"buckets": [512, 1024, 2048, 4096, 8192], "max_tokens": 8192}
    (artifacts / "manifest.json").write_text(json.dumps(manifest))
    (artifacts / "upstream/config.json").write_text(json.dumps({"max_seq_length": 8192}))
    digest = "a" * 64
    (artifacts / "validation.json").write_text(json.dumps({"passed": True, "encoder_sha256": digest}))
    report = {
        **manifest,
        "passed": True,
        "encoder_sha256": digest,
        "boundary_embeddings": [
            {"tokens": bucket, "reference_consumed_tokens": bucket, "passed": True}
            for bucket in manifest["buckets"]
        ],
        "overlength_tokens": 8193,
        "overlength_rejected": True,
        "varied_embeddings": [],
        "long_decisions": [],
    }
    for bucket in [4096, 8192]:
        for position in ["beginning", "middle", "end"]:
            metadata = {"target_bucket": bucket, "evidence_position": position, "passed": True}
            report["varied_embeddings"].append({
                **metadata, "rendered_state_tokens": bucket - 64,
                "reference_consumed_tokens": bucket - 64,
            })
            report["long_decisions"].extend(
                {**metadata, "question": question}
                for question in ["action", "risk", "needs_review"]
            )
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(sys, "argv", ["switch_artifacts.py", str(artifacts)])
    monkeypatch.setattr(switch_artifacts, "package_sha256", lambda root: digest)

    def forbidden_subprocess(*args, **kwargs):
        pytest.fail("Parity gate must finish before invoking launchctl or verification")

    monkeypatch.setattr(switch_artifacts.subprocess, "run", forbidden_subprocess)
    return artifacts, report, home


@pytest.mark.parametrize("defect, error", [
    ("stale", "Reference no-truncation evidence missing"),
    ("truncated_boundary", "Reference no-truncation evidence missing"),
    ("missing_varied", "Varied-state embedding coverage missing"),
    ("truncated_varied", "Varied-state embedding coverage missing"),
    ("missing_counts", "Varied-state embedding coverage missing"),
    ("missing_question", "Varied-state typed decision coverage missing"),
])
def test_switch_rejects_incomplete_evidence_before_runner_access(switch_reports, defect, error):
    artifacts, report, home = switch_reports
    if defect == "stale":
        for row in report["boundary_embeddings"]:
            row.pop("reference_consumed_tokens")
        report.pop("varied_embeddings")
        report.pop("long_decisions")
    elif defect == "truncated_boundary":
        report["boundary_embeddings"][-1]["reference_consumed_tokens"] = 2048
    elif defect == "missing_varied":
        report["varied_embeddings"].pop()
    elif defect == "truncated_varied":
        report["varied_embeddings"][-1]["reference_consumed_tokens"] = 2048
    elif defect == "missing_counts":
        report["varied_embeddings"][-1].pop("rendered_state_tokens")
        report["varied_embeddings"][-1].pop("reference_consumed_tokens")
    elif defect == "missing_question":
        report["long_decisions"].pop()
    (artifacts / "validation-long.json").write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match=error):
        switch_artifacts.main()
    assert list(home.iterdir()) == []


def test_switch_accepts_complete_evidence_up_to_runner_access(switch_reports):
    artifacts, report, home = switch_reports
    (artifacts / "validation-long.json").write_text(json.dumps(report))
    with pytest.raises(FileNotFoundError) as exc:
        switch_artifacts.main()
    assert Path(exc.value.filename) == home / ".config/typed-decisions/run.sh"
    assert list(home.iterdir()) == []
