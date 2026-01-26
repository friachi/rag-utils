# main.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, cast

from fastapi import FastAPI, HTTPException, File, UploadFile, Body, Request, Query
from starlette.concurrency import run_in_threadpool
from flashrank import RerankRequest
from transformers import pipeline
from transformers.pipelines.base import Pipeline
import base64

import os
from contextlib import asynccontextmanager

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers.pipelines.base import Pipeline
from transformers import pipeline

# reranker
from utils.reranker import (
    PAIRWISE_MAX_MAX_LENGTH,
    LISTWISE_MAX_MAX_LENGTH,
    LISTWISE_MAX_PASSAGES,
    MODEL_REGISTRY,
    ModelFamily,
    RerankIn,
    RerankOut,
    RankedPassage,
    resolve_model,
    get_ranker
)

# converter
from utils.converter import (
    ConvertJsonRequest,
    ConvertResponse,
    ConvertOptions,
    decode_bytes_to_text,
    convert_to_markdown,
    detect_nature
)

# classifier

from utils.classifier import (
    DEFAULT_MODEL_ID,
    ClassifyRequest,
    ClassifyResponse
)

HF_CACHE_DIR = os.getenv("HF_HUB_CACHE", "/opt/hf/hub")
DEFAULT_MODEL_ID = os.getenv("HF_MODEL_ID", DEFAULT_MODEL_ID)  # keep your utils default, but allow env override

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Force offline + load only from baked files
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        DEFAULT_MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        local_files_only=True,
    )

    clf: Pipeline = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU
    )

    app.state.zero_shot_clf = clf
    yield

# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(title="RagUtils", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


###############################
########## Reranker ###########
###############################

@app.get("/models")
async def models() -> Dict[str, Any]:
    """Expose model aliases with metadata."""
    return {
        k.value: {
            "model_name": v["model_name"],
            "family": v["family"].value,
            "max_max_length": (
                PAIRWISE_MAX_MAX_LENGTH if v["family"] == ModelFamily.pairwise else LISTWISE_MAX_MAX_LENGTH
            ),
            "listwise_max_passages": (LISTWISE_MAX_PASSAGES if v["family"] == ModelFamily.listwise else None),
        }
        for k, v in MODEL_REGISTRY.items()
    }


@app.post("/rerank", response_model=RerankOut)
async def rerank(payload: RerankIn) -> RerankOut:
    try:
        actual_model, family = resolve_model(payload.model)
        ranker = get_ranker(actual_model, payload.max_length)  # type: ignore # payload.max_length is now always int

        rr = RerankRequest(
            query=payload.query,
            passages=[p.model_dump() for p in payload.passages],
        )

        # FlashRank is sync + CPU-bound; offload to threadpool.
        results_raw = await run_in_threadpool(ranker.rerank, rr)

        # Normalize results to always include score (listwise may omit it)
        normalized: List[Dict[str, Any]] = []
        n = len(results_raw) if results_raw else 0

        for i, r in enumerate(results_raw or []):
            rr = dict(r)  # ensure mutable copy
            if "score" not in rr or rr["score"] is None:
                # Fallback: assign a synthetic score based on rank (higher is better)
                # Top item gets ~1.0, then slightly decreasing.
                rr["score"] = 1.0 - (i / max(n, 1)) * 1e-3
            normalized.append(rr)

        return RerankOut(
            model=actual_model,
            family=family,
            max_length=payload.max_length, # type: ignore
            query=payload.query,
            results=[RankedPassage.model_validate(r) for r in normalized],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rerank failed: {e}") from e

###############################
########## Converter ##########
###############################

from fastapi import Query

import base64
from typing import Optional

from fastapi import Body, File, HTTPException, Request, UploadFile
from pydantic import ValidationError

@app.post("/convert", response_model=ConvertResponse)
async def convert_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    max_example_length: Optional[int] = None,
    max_enum_values: Optional[int] = None,
) -> ConvertResponse:
    raw: bytes
    filename: Optional[str] = None

    content_type = (request.headers.get("content-type") or "").lower()

    # --- JSON path (application/json) ---
    if content_type.startswith("application/json"):
        try:
            payload = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid json body: {e}")

        try:
            json_body = ConvertJsonRequest.model_validate(payload)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"invalid request body: {e}")

        filename = json_body.filename
        try:
            raw = base64.b64decode(json_body.content_base64, validate=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid base64 content: {e}")

    # --- multipart path (file upload) ---
    else:
        if file is None:
            raise HTTPException(
                status_code=400,
                detail=f"provide either multipart field 'file' or json body with 'content_base64' (content-type was: {content_type})",
            )
        try:
            raw = await file.read()
            filename = file.filename
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"failed to read upload: {e}")

    # decode bytes -> text (utf8 first, then fallbacks)
    try:
        text, encoding_used = decode_bytes_to_text(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to decode bytes to text: {e}")

    # detect
    try:
        detection = detect_nature(text)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=f"unsupported or unrecognized file type: {e}")

    # convert with overrides
    try:
        md = convert_to_markdown(
            text,
            filename=filename,
            options=ConvertOptions(
                source_name=filename,
                max_example_length=max_example_length,
                max_enum_values=max_enum_values,
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"conversion failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"conversion crashed: {e}")

    return ConvertResponse(detected_type=detection.nature.value, markdown=md)

###############################
######### Classifier ##########
###############################

@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest, request: Request) -> ClassifyResponse:
    clf: Pipeline | None = getattr(request.app.state, "zero_shot_clf", None)
    if clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # The pipeline returns a dict like:
        # {"sequence": str, "labels": [str], "scores": [float]}
        result = await run_in_threadpool(
            clf,
            req.text,
            req.labels,
            multi_label=req.multi_label,
            hypothesis_template=req.hypothesis_template,
        )
        result_dict = cast(Dict[str, Any], result)  # makes Pylance happy
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    labels_any = result_dict.get("labels")
    scores_any = result_dict.get("scores")
    sequence = str(result_dict.get("sequence", req.text))

    if not isinstance(labels_any, list) or not isinstance(scores_any, list):
        raise HTTPException(status_code=500, detail=f"Unexpected model output: {type(result_dict)}")
    if len(labels_any) != len(scores_any):
        raise HTTPException(status_code=500, detail="Unexpected model output lengths differ.")

    labels = [str(x) for x in labels_any]
    scores = [float(x) for x in scores_any]

    if req.top_k is not None and req.top_k < len(labels):
        labels = labels[: req.top_k]
        scores = scores[: req.top_k]

    return ClassifyResponse(
        model_id=DEFAULT_MODEL_ID,
        labels=labels,
        scores=scores
        # sequence=sequence,
    )



# Run with:
#   uvicorn main:app --host 0.0.0.0 --port 8181
