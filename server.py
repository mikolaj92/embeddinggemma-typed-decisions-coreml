"""Local authenticated decision API; run with python server.py."""

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse

from runtime import MAX_BODY_BYTES, DecisionRequest, DecisionRuntime, RequestError


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    kind: Literal["choice", "score", "noul"]
    label: str
    value: str | int | bool
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    expected_level: float | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.api_key = os.environ.get("DECISIONS_API_KEY", "")
    app.state.runtime = None
    if app.state.api_key:
        artifacts = os.environ.get(
            "DECISIONS_ARTIFACTS", str(Path(__file__).parent / "artifacts")
        )
        app.state.runtime = DecisionRuntime(artifacts)
        from integrity import package_sha256
        app.state.encoder_sha256 = package_sha256(artifacts)
    yield
    app.state.runtime = None
    app.state.api_key = ""


app = FastAPI(
    title="Typed Decisions",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


class DecisionBoundary:
    """Authenticate before parsing; cap streamed bodies as well as Content-Length."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/v1/decisions":
            return await self.app(scope, receive, send)
        key = getattr(app.state, "api_key", "")
        headers = dict(scope.get("headers", []))
        if not key:
            return await JSONResponse(
                {"detail": "service unavailable"}, status_code=503
            )(scope, receive, send)
        authorization = headers.get(b"authorization", b"")
        if not hmac.compare_digest(authorization, ("Bearer " + key).encode()):
            return await JSONResponse(
                {"detail": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )(scope, receive, send)
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse(
                {"detail": "invalid content length"}, status_code=400
            )(scope, receive, send)
        if length < 0 or length > MAX_BODY_BYTES:
            return await JSONResponse({"detail": "request too large"}, status_code=413)(
                scope, receive, send
            )
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                return await JSONResponse(
                    {"detail": "request too large"}, status_code=413
                )(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        sent = False

        async def bounded_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


app.add_middleware(DecisionBoundary)


@app.get("/health")
def health():
    runtime = getattr(app.state, "runtime", None)
    result = {"status": "ok", "ready": runtime is not None and bool(getattr(app.state, "api_key", ""))}
    if runtime is not None and hasattr(runtime, "buckets"):
        result.update(max_tokens=runtime.max_tokens, buckets=runtime.buckets)
        result["encoder_sha256"] = getattr(app.state, "encoder_sha256", None)
    return result


@app.post("/v1/decisions", response_model=dict[str, Answer])
def decisions(payload: DecisionRequest, request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="service unavailable")
    try:
        answers = runtime.decide(
            payload.state,
            {
                qid: q.model_dump(exclude_none=True)
                for qid, q in payload.questions.items()
            },
            payload.workflow,
        )
        return {qid: Answer.model_validate(answer) for qid, answer in answers.items()}
    except RequestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Never expose filesystem paths, model internals, or request state.
        raise HTTPException(status_code=503, detail="inference unavailable") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="127.0.0.1", port=int(os.environ.get("DECISIONS_PORT", "8000"))
    )
