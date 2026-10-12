"""Behavioral owners: wire validation, encoder artifact boundary, typed decisions.

CoreML hardware alone is substituted; tokenizer and pinned upstream head are real.
"""

import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from runtime import (
    DecisionRequest,
    DecisionRuntime,
    InferenceError,
    RequestError,
    load_upstream,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    for name in ("modeling_gemmadecision.py", "configuration_gemmadecision.py"):
        source = ROOT / "artifacts" / "upstream" / name
        if not source.exists():
            source = Path("/tmp") / ("eg2td-" + name)
        if not source.exists():
            pytest.fail("pinned upstream custom source unavailable")
        shutil.copyfile(source, upstream / name)
    config = {
        "embed_dim": 4,
        "head_width": 8,
        "head_dropout": 0.0,
        "prompt": "task ",
        "pooling_include_prompt": True,
    }
    (upstream / "config.json").write_text(json.dumps(config))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"source_revision": "a" * 40, "max_tokens": 8, "buckets": [8]})
    )
    vocabulary = {
        word: i
        for i, word in enumerate(
            ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "task", "short", "long"]
        )
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 2), ("[EOS]", 3)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    table = np.arange(7 * 512, dtype=np.float32).reshape(7, 512) / 1024
    np.save(tmp_path / "embeddings.npy", table)
    upstream_module = load_upstream(upstream)
    torch.manual_seed(12)
    head = upstream_module.GemmaDecisionHead(4, 8, 0.0, 3).eval()
    weights = {"head." + key: value for key, value in head.state_dict().items()}
    weights["temperatures"] = torch.tensor([0.5, 1.5, 2.0])
    save_file(weights, tmp_path / "head.safetensors")

    class Encoder:
        def __init__(self, path, compute_units):
            assert compute_units == "cpu_and_ne"
            self.calls = []
            self.input_ids = []
            self.output = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
            self.active = 0
            self.max_active = 0
            self.delay = 0
            self.outputs = []
            self.distinct = False

        def predict(self, inputs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(self.delay)
                self.input_ids.append(tuple(id(inputs[k]) for k in ["embeddings","attention_mask"]))
                self.calls.append({k:v.copy() for k,v in inputs.items()})
                output = self.output.copy()
                if self.distinct:
                    index = len(self.calls)
                    output = np.sin(np.arange(4, dtype=np.float32) + index).reshape(
                        1, 4
                    )
                self.outputs.append(output)
                return {"sentence_embedding": output}
            finally:
                self.active -= 1

    monkeypatch.setitem(
        sys.modules,
        "coremltools",
        SimpleNamespace(
            models=SimpleNamespace(MLModel=Encoder),
            ComputeUnit=SimpleNamespace(CPU_AND_NE="cpu_and_ne"),
        ),
    )
    return DecisionRuntime(tmp_path)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "unknown", "criteria": ["a"]},
        {"type": "choice", "criteria": []},
        {"type": "choice", "criteria": ["a", "a"]},
        {"type": "choice", "criteria": [1]},
        {"type": "score", "criteria": {"1": "a", "01": "b"}},
        {"type": "score", "criteria": {"--1": "a", "2": "b"}},
        {"type": "score", "criteria": {"²": "a", "2": "b"}},
        {"type": "noul", "criteria": ["yes", "no"]},
        {"type": "noul", "unexpected": True},
        {"type": "choice", "criteria": [str(i) for i in range(65)]},
    ],
)
def test_schema_rejects_ambiguous_or_unbounded_questions(question):
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate({"state": {}, "questions": {"q": question}})


@pytest.mark.parametrize(
    "state",
    [float("nan"), {"x": float("inf")}, '{"x": NaN}', "{bad json}", "x" * 262145],
)
def test_state_invalid_or_oversized_is_rejected(state):
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(
            {"state": state, "questions": {"q": {"type": "noul"}}}
        )


def test_rendering_is_training_text_contract(runtime):
    upstream = runtime.upstream
    assert (
        upstream.render_state(
            "audit", {"constraint_violations": 1, "nested": {"is_ready": False}}
        )
        == "workflow: audit\nconstraint violations: 1\nnested · is ready: False"
    )
    spec = upstream.parse_question(
        "risk", {"type": "score", "criteria": {"5": "high", "1": "low"}}
    )
    assert spec.options == ("1", "5")
    assert upstream.render_question(spec) == "score question: risk"
    assert (
        upstream.render_option(spec, 1) == "question: risk | answer: level 5 of 1: high"
    )
    noul = upstream.parse_question("safe", {"type": "noul"})
    assert upstream.render_option(noul, 0) == "question: safe | answer: no"


def test_encoder_preserves_specials_padding_scale_and_rejects_overflow(runtime):
    model = runtime.models[8]
    runtime.encode(["short"])
    actual = model.calls[0]
    expected_ids = [2, 4, 5, 3, 0, 0, 0, 0]
    np.testing.assert_allclose(
        actual["embeddings"],
        runtime.table[np.array([expected_ids])] * np.float32(np.sqrt(512)),
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(actual["attention_mask"], [[1, 1, 1, 1, 0, 0, 0, 0]])
    assert actual["attention_mask"].dtype == np.int32
    runtime.encode(["short short short short short"])
    assert len(model.calls) == 2  # exactly eight tokens including task+BOS+EOS
    assert model.input_ids[0] == model.input_ids[1]  # native resets never own the last Python array reference
    with pytest.raises(RequestError, match="max_tokens"):
        runtime.encode(["short", "short short short short short short"])
    assert len(model.calls) == 2  # fail before any partial prediction


def test_encoder_selects_smallest_bucket_and_rejects_2049(runtime):
    encoder_type=type(runtime.models[8])
    runtime.max_tokens=2048
    runtime.buckets=[512,1024,2048]
    runtime.models={n:encoder_type(str(n),"cpu_and_ne") for n in runtime.buckets}
    runtime._input_buffers={n:{"embeddings":np.zeros((1,n,512),dtype=np.float32),"attention_mask":np.zeros((1,n),dtype=np.int32)} for n in runtime.buckets}
    for length in [512,513,1024,1025,2048]:
        runtime.encode(["short "*(length-3)])
        bucket=next(n for n in runtime.buckets if length<=n)
        assert runtime.models[bucket].calls[-1]["attention_mask"].shape==(1,bucket)
        assert runtime.models[bucket].calls[-1]["attention_mask"].sum()==length
    assert [len(runtime.models[n].calls) for n in runtime.buckets]==[1,2,2]
    with pytest.raises(RequestError,match="max_tokens"):
        runtime.encode(["short "*2046])
    assert [len(runtime.models[n].calls) for n in runtime.buckets]==[1,2,2]


def test_typed_decisions_use_upstream_head_and_calibration(runtime):
    runtime.max_tokens = 128
    runtime.buckets = [128]
    runtime.models[128] = runtime.models.pop(8)
    runtime._input_buffers[128]={"embeddings":np.zeros((1,128,512),dtype=np.float32),"attention_mask":np.zeros((1,128),dtype=np.int32)}
    questions = {
        "pick": {"type": "choice", "criteria": ["a", "b", "c"]},
        "risk": {"type": "score", "criteria": {"5": "high", "1": "low"}},
        "safe": {"type": "noul"},
    }
    runtime.models[128].distinct = True
    answers = runtime.decide({}, questions)
    encoded = torch.from_numpy(np.concatenate(runtime.models[128].outputs))
    with torch.inference_mode():
        for index, (qid, labels) in enumerate(
            [
                ("pick", ["a", "b", "c"]),
                ("risk", ["1", "5"]),
                ("safe", ["false", "true"]),
            ]
        ):
            option_rows = [[4, 5, 6], [7, 8], [9, 10]][index]
            logits = runtime.head(
                encoded[0:1],
                encoded[index + 1 : index + 2],
                encoded[option_rows].unsqueeze(0),
                torch.tensor([index]),
                torch.ones(1, len(labels), dtype=torch.bool),
            )
            expected = (logits / runtime.temperatures[index]).softmax(-1)[0]
            assert list(answers[qid]["probabilities"]) == labels
            np.testing.assert_allclose(
                list(answers[qid]["probabilities"].values()),
                expected.numpy(),
                rtol=1e-6,
            )
            assert answers[qid]["label"] == labels[int(expected.argmax())]
    assert isinstance(answers["safe"]["value"], bool)
    assert isinstance(answers["risk"]["value"], int)
    risk = answers["risk"]["probabilities"]
    assert answers["risk"]["expected_level"] == pytest.approx(risk["1"] + 5 * risk["5"])
    assert answers["pick"]["expected_level"] is None


def test_inference_rejects_nonfinite_and_serializes_hardware(runtime):
    model = runtime.models[8]
    model.output[0, 0] = np.nan
    with pytest.raises(InferenceError, match="non-finite"):
        runtime.encode(["short"])
    model.output[:] = 0
    with pytest.raises(InferenceError, match="zero"):
        runtime.encode(["short"])
    model.output[:] = 0.1
    model.delay = 0.01
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: runtime.encode(["short"]), range(8)))
    assert model.max_active == 1


def test_http_auth_size_validation_health_and_safe_failure(monkeypatch):
    from fastapi.testclient import TestClient

    import server

    monkeypatch.delenv("DECISIONS_API_KEY", raising=False)
    with TestClient(server.app) as client:
        assert client.get("/health").json() == {"status": "ok", "ready": False}
        assert client.post("/v1/decisions", content="not json").status_code == 503
        server.app.state.api_key = "test-secret"
        assert client.post("/v1/decisions", content="not json").status_code == 401
        auth = {"Authorization": "Bearer test-secret"}
        assert (
            client.post("/v1/decisions", headers=auth, content="x" * 262145).status_code
            == 413
        )
        assert (
            client.post(
                "/v1/decisions",
                headers=auth,
                content=iter([b"x" * 140000, b"x" * 140000]),
            ).status_code
            == 413
        )
        assert (
            client.post(
                "/v1/decisions", headers=auth, json={"state": {}, "questions": {}}
            ).status_code
            == 422
        )

        class BrokenRuntime:
            def decide(self, *args):
                raise InferenceError("private model path and state")

        server.app.state.runtime = BrokenRuntime()
        response = client.post(
            "/v1/decisions",
            headers=auth,
            json={"state": {}, "questions": {"q": {"type": "noul"}}},
        )
        assert response.status_code == 503
        assert response.json() == {"detail": "inference unavailable"}
        assert "test-secret" not in client.get("/health").text
