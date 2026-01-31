from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, cast

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from transformers import pipeline
from transformers.pipelines.base import Pipeline

# ---- HF / Transformers offline + cache configuration ----
HF_ROOT = os.getenv("HF_ROOT", "/opt/hf")  
HF_ROOT = os.getenv("HF_ROOT", "/Users/fahed/.cache/huggingface")  # override in macOS dev, and make sure to set this env var to local dev cache path. !!!!! REMOVE before building container.

os.environ.setdefault("HF_HOME", HF_ROOT)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_ROOT, "transformers"))

# force offline in runtime containers; in dev you can override these too if needed
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEFAULT_MODEL_ID = os.getenv("HF_MODEL_ID", "facebook/bart-large-mnli")


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1)
    labels: List[str] = Field(..., min_length=2)
    multi_label: bool = False
    hypothesis_template: str = "This example is about {}."
    top_k: Optional[int] = Field(None, ge=1)

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, v: List[str]) -> List[str]:
        cleaned = [s.strip() for s in v if s and s.strip()]
        seen: set[str] = set()
        deduped: List[str] = []
        for s in cleaned:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        if len(deduped) < 2:
            raise ValueError("Provide at least 2 non-empty labels.")
        return deduped


class ClassifyResponse(BaseModel):
    model_id: str
    labels: List[str]
    scores: List[float]
    # sequence: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the zero-shot pipeline once, reuse for all requests. make it explicit so you never accidentally hit the network (i.e rely on cache for models)
    """
    try:
        clf: Pipeline = pipeline(
            "zero-shot-classification",
            model=DEFAULT_MODEL_ID,
            device=-1,  # CPU
            model_kwargs={"local_files_only": True},
            tokenizer_kwargs={"local_files_only": True},
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load zero-shot pipeline for {DEFAULT_MODEL_ID}: {e}") from e

    app.state.zero_shot_clf = clf
    yield
    # No explicit cleanup needed
