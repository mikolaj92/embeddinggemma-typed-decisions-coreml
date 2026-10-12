"""CoreML encoder + pinned upstream CPU decision head. No torch encoder is built."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

MAX_BODY_BYTES = 262_144
MAX_QUESTIONS = 32
MAX_OPTIONS = 64
MAX_TOTAL_OPTIONS = 256


class RequestError(ValueError):
    """Invalid or oversized decision input."""


class InferenceError(RuntimeError):
    """Invalid numerical output; never return partial decisions."""


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["choice", "score", "noul"] = "choice"
    instructions: str = Field(default="", max_length=8192)
    criteria: list[str] | dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_options(self):
        if self.type == "noul":
            if self.criteria is not None:
                raise ValueError("noul has fixed false/true options; omit criteria")
        else:
            if not self.criteria or len(self.criteria) > MAX_OPTIONS:
                raise ValueError(f"criteria must contain 1..{MAX_OPTIONS} options")
            labels = list(self.criteria)
            if any(not label or len(label) > 8192 for label in labels):
                raise ValueError("option labels must be nonempty and bounded")
            if self.type == "choice" and len(set(labels)) != len(labels):
                raise ValueError("choice labels must be unique")
            if isinstance(self.criteria, dict):
                if any(len(v) > 8192 for v in self.criteria.values()):
                    raise ValueError("option descriptions are too long")
                if self.type == "score" and all(
                    k.lstrip("-").isdigit() for k in labels
                ):
                    if any(len(k) > 12 or not re.fullmatch(r"-?[0-9]+", k) for k in labels) or len(
                        {int(k) for k in labels}
                    ) != len(labels):
                        raise ValueError("score levels must be bounded and unique")
        return self


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    state: JsonValue
    questions: dict[str, Question]
    workflow: str = Field(default="", max_length=1024)

    @model_validator(mode="after")
    def validate_size(self):
        if not 1 <= len(self.questions) <= MAX_QUESTIONS:
            raise ValueError(f"provide 1..{MAX_QUESTIONS} questions")
        if any(not qid or len(qid) > 256 for qid in self.questions):
            raise ValueError("question ids must be nonempty and bounded")
        if (
            sum(
                2 if q.type == "noul" else len(q.criteria)
                for q in self.questions.values()
            )
            > MAX_TOTAL_OPTIONS
        ):
            raise ValueError("too many total options")
        state = self.state
        if isinstance(state, str) and (not state.strip() or state.strip()[:1] in "{["):
            try:
                state = json.loads(state)
            except ValueError as exc:
                raise ValueError("state looks like JSON but is not valid JSON") from exc

        def walk(value, depth=0):
            if depth > 32:
                raise ValueError("state nesting exceeds 32")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("state numbers must be finite")
            if isinstance(value, dict):
                for key, child in value.items():
                    if not isinstance(key, str):
                        raise ValueError("state object keys must be strings")
                    walk(child, depth + 1)
            elif isinstance(value, list):
                for child in value:
                    walk(child, depth + 1)

        walk(state)
        if len(self.model_dump_json().encode("utf-8")) > MAX_BODY_BYTES:
            raise ValueError("request exceeds size limit")
        return self


def load_upstream(directory: Path) -> ModuleType:
    """Import local, revision-pinned custom source with its relative config import."""
    directory = directory.resolve()
    package_name = (
        "_decision_upstream_" + hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    )
    module_name = package_name + ".modeling_gemmadecision"
    if module_name in sys.modules:
        return sys.modules[module_name]
    package = ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        module_name, directory / "modeling_gemmadecision.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("missing upstream modeling source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        sys.modules.pop(package_name, None)
        raise
    return module


def _finite(value, label: str):
    if not np.isfinite(value).all():
        raise InferenceError(f"non-finite {label}")


class DecisionRuntime:
    def __init__(self, artifacts: str | Path):
        import coremltools as ct
        from safetensors import safe_open
        from transformers import AutoTokenizer

        self.artifacts = Path(artifacts).resolve()
        self.manifest = json.loads((self.artifacts / "manifest.json").read_text())
        revision = self.manifest.get("source_revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("manifest source_revision must be a pinned commit SHA")
        self.max_tokens = self.manifest.get("max_tokens")
        self.buckets = self.manifest.get("buckets")
        if (
            type(self.max_tokens) is not int
            or self.max_tokens <= 0
            or not isinstance(self.buckets, list)
            or not self.buckets
            or any(type(n) is not int or n <= 0 for n in self.buckets)
            or self.buckets != sorted(set(self.buckets))
            or self.max_tokens > self.buckets[-1]
        ):
            raise ValueError("invalid manifest token limits/buckets")
        self.config = json.loads(
            (self.artifacts / "upstream" / "config.json").read_text()
        )
        if self.config.get("question_types", ["choice", "score", "noul"]) != [
            "choice",
            "score",
            "noul",
        ]:
            raise ValueError("unsupported question type ordering")
        if self.config.get("pooling_include_prompt", True) is not True:
            raise ValueError("encoder export must include prompt pooling")
        self.upstream = load_upstream(self.artifacts / "upstream")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.artifacts / "tokenizer", local_files_only=True, trust_remote_code=False
        )
        self.table = np.load(
            self.artifacts / "embeddings.npy", mmap_mode="r", allow_pickle=False
        )
        if (
            self.table.dtype != np.float32
            or self.table.ndim != 2
            or self.table.shape[1] != 512
        ):
            raise ValueError("embeddings.npy must be unscaled FP32 [vocab,512]")
        self.head = (
            self.upstream.GemmaDecisionHead(
                self.config["embed_dim"],
                self.config.get("head_width", 256),
                self.config.get("head_dropout", 0.1),
                3,
            )
            .cpu()
            .float()
            .eval()
        )
        with safe_open(
            self.artifacts / "head.safetensors", framework="pt", device="cpu"
        ) as weights:
            head_weights = {
                k.removeprefix("head."): weights.get_tensor(k).float()
                for k in weights.keys()
                if k.startswith("head.")
            }
            self.temperatures = weights.get_tensor("temperatures").float()
        self.head.load_state_dict(head_weights, strict=True)
        for weight in list(self.head.parameters()) + [self.temperatures]:
            _finite(weight.detach().numpy(), "head weights")
        if self.temperatures.shape != (3,) or not (self.temperatures > 0).all():
            raise ValueError("temperatures must be three positive finite values")
        files = self.manifest.get("models", {})
        if len(self.buckets) > 1 and not files:
            raise ValueError("multiple buckets require models mapping")
        self.models = {}
        # coremltools9 retains py::array inside an Objective-C MLMultiArray.
        # Core ML can release that wrapper later on a native reset queue without
        # Python GIL. Keep these arrays owned by Python for the runtime lifetime
        # so delayed native decref is not the final object deallocation.
        self._input_buffers = {}
        for bucket in self.buckets:
            filename = files.get(str(bucket), "Encoder.mlpackage")
            path = (self.artifacts / filename).resolve()
            if not path.is_relative_to(self.artifacts):
                raise ValueError("model path escapes artifact directory")
            self.models[bucket] = ct.models.MLModel(
                str(path), compute_units=ct.ComputeUnit.CPU_AND_NE
            )
            self._input_buffers[bucket] = {
                "embeddings": np.zeros((1, bucket, 512), dtype=np.float32),
                "attention_mask": np.zeros((1, bucket), dtype=np.int32),
            }
        self._lock = threading.Lock()

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Reject oversize before prediction; preserve AutoTokenizer special tokens."""
        with self._lock:
            return self._encode(texts)

    def _encode(self, texts: list[str]) -> torch.Tensor:
        if not texts or any(not isinstance(text, str) for text in texts):
            raise RequestError("provide nonempty text list")
        if (
            len(texts) > 1 + MAX_QUESTIONS + MAX_TOTAL_OPTIONS
            or sum(len(text.encode("utf-8")) for text in texts) > MAX_BODY_BYTES * 2
        ):
            raise RequestError("encoded text batch exceeds size limits")
        prompt = self.config.get("prompt", "") or ""
        tokenized = self.tokenizer(
            [prompt + text for text in texts],
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        sequences = tokenized["input_ids"]
        if any(len(ids) > self.max_tokens for ids in sequences):
            raise RequestError(f"rendered text exceeds max_tokens={self.max_tokens}")
        result = []
        for ids in sequences:
            bucket = next(n for n in self.buckets if len(ids) <= n)
            padded = self.tokenizer.pad(
                {"input_ids": [ids]},
                padding="max_length",
                max_length=bucket,
                return_tensors="np",
            )
            indices = padded["input_ids"]
            if indices.min() < 0 or indices.max() >= self.table.shape[0]:
                raise InferenceError("token id is outside embedding table")
            embeddings = np.asarray(self.table[indices], dtype=np.float32) * np.float32(
                math.sqrt(512)
            )
            _finite(embeddings, "token embeddings")
            mask = np.asarray(padded["attention_mask"], dtype=np.int32)
            buffers = self._input_buffers[bucket]
            np.copyto(buffers["embeddings"], embeddings)
            np.copyto(buffers["attention_mask"], mask)
            output = self.models[bucket].predict(buffers)
            embedding = np.asarray(output["sentence_embedding"], dtype=np.float32)
            if embedding.shape != (1, self.config["embed_dim"]):
                raise InferenceError("unexpected encoder output shape")
            _finite(embedding, "sentence embedding")
            if np.linalg.norm(embedding) < 1e-6:
                raise InferenceError("zero sentence embedding")
            result.append(torch.from_numpy(embedding.copy())[0])
        return torch.stack(result)

    def decide(self, state: JsonValue, questions: dict, workflow: str = "") -> dict:
        request = DecisionRequest.model_validate(
            {"state": state, "questions": questions, "workflow": workflow}
        )
        with self._lock, torch.inference_mode():
            specs = [
                self.upstream.parse_question(qid, q.model_dump(exclude_none=True))
                for qid, q in request.questions.items()
            ]
            bank = {}

            def text_id(text):
                return bank.setdefault(text, len(bank))

            state_id = text_id(
                self.upstream.render_state(request.workflow, request.state)
            )
            question_ids = [
                text_id(self.upstream.render_question(spec)) for spec in specs
            ]
            option_ids = [
                [
                    text_id(self.upstream.render_option(spec, i))
                    for i in range(len(spec.options))
                ]
                for spec in specs
            ]
            embeddings = self._encode(list(bank))
            width = max(map(len, option_ids))
            indices = torch.zeros((len(specs), width), dtype=torch.long)
            mask = torch.zeros_like(indices, dtype=torch.bool)
            for row, ids in enumerate(option_ids):
                indices[row, : len(ids)] = torch.tensor(ids)
                mask[row, : len(ids)] = True
            kinds = torch.tensor([spec.kind_id for spec in specs])
            logits = self.head(
                embeddings[state_id].expand(len(specs), -1),
                embeddings[question_ids],
                embeddings[indices],
                kinds,
                mask,
            ).float()
            _finite(logits.numpy(), "head logits")
            probabilities = (
                (logits / self.temperatures[kinds, None])
                .masked_fill(~mask, -1e4)
                .softmax(-1)
            )
            _finite(probabilities.numpy(), "probabilities")
            answers = {}
            for spec, row in zip(specs, probabilities):
                p = row[: len(spec.options)]
                winner = int(p.argmax())
                label = spec.options[winner]
                value = (
                    label == "true"
                    if spec.kind == "noul"
                    else int(label)
                    if spec.kind == "score"
                    else label
                )
                expected = (
                    sum(int(o) * float(pi) for o, pi in zip(spec.options, p))
                    if spec.kind == "score"
                    else None
                )
                answers[spec.qid] = self.upstream.TypedAnswer(
                    spec.kind,
                    label,
                    value,
                    float(p[winner]),
                    dict(zip(spec.options, p.tolist())),
                    expected,
                ).to_dict()
            return answers
