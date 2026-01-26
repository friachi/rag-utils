from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import os
from pydantic import BaseModel, Field, ConfigDict, model_validator
from flashrank import Ranker
from fastapi import HTTPException

# -----------------------------
# Configuration
# -----------------------------

DEFAULT_MAX_LENGTH = int(os.getenv("FLASHRANK_MAX_LENGTH", "128"))

# Family-specific caps (pairwise models are typically 512-token max; listwise can be larger)
PAIRWISE_MAX_MAX_LENGTH = int(os.getenv("FLASHRANK_PAIRWISE_MAX_MAX_LENGTH", "512"))
LISTWISE_MAX_MAX_LENGTH = int(os.getenv("FLASHRANK_LISTWISE_MAX_MAX_LENGTH", "8192"))

# Guardrails for listwise throughput / safety
LISTWISE_MAX_PASSAGES = int(os.getenv("FLASHRANK_LISTWISE_MAX_PASSAGES", "50"))

CACHE_DIR = os.getenv("FLASHRANK_CACHE_DIR", "/opt/flashrank_models")

class ModelFamily(str, Enum):
    pairwise = "pairwise"
    listwise = "listwise"


# Renamed sizes + added listwise options
class ModelId(str, Enum):
    pairwise_nano = "pairwise_nano"
    pairwise_small = "pairwise_small"
    pairwise_medium = "pairwise_medium"
    pairwise_large = "pairwise_large"

    listwise_medium = "listwise_medium"


# Model registry: alias -> metadata (resolved FlashRank model + family)
MODEL_REGISTRY: Dict[ModelId, Dict[str, Any]] = {
    # Pairwise / cross-encoder / ONNX-ish models
    ModelId.pairwise_nano: {
        "model_name": "ms-marco-TinyBERT-L-2-v2",   # ~4MB
        "family": ModelFamily.pairwise,
    },
    ModelId.pairwise_small: {
        "model_name": "ms-marco-MiniLM-L-12-v2",    # best cross-encoder reranker
        "family": ModelFamily.pairwise,
    },
    ModelId.pairwise_medium: {
        "model_name": "ms-marco-MultiBERT-L-12",    # multilingual
        "family": ModelFamily.pairwise,
    },
    ModelId.pairwise_large: {
        "model_name": "rank-T5-flan",               # best non cross-encoder reranker
        "family": ModelFamily.pairwise,
    },

    # Listwise / LLM-based rerankers (require flashrank[listwise] + large model assets)
    ModelId.listwise_medium: {
        "model_name": "rank_zephyr_7b_v1_full",
        "family": ModelFamily.listwise,
    },
}

# -----------------------------
# Pydantic models
# -----------------------------

JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, Dict[str, Any], List[Any]]


class Passage(BaseModel):
    """
    FlashRank expects passages like:
      {"id": <int/str>, "text": <str>, "meta": <optional dict>}
    """
    model_config = ConfigDict(extra="allow")

    id: Union[int, str] = Field(..., description="Passage identifier (DB id or index).")
    text: str = Field(..., min_length=1, description="Passage text.")
    meta: Optional[Dict[str, JSONValue]] = Field(
        default=None, description="Optional metadata blob."
    )


class RerankIn(BaseModel):
    query: str = Field(
        ...,
        description="Search query used to rerank passages.",
        examples=["what is uvicorn"],
    )

    model: ModelId = Field(
        ...,
        description="Model alias (includes model family).",
        examples=["pairwise_nano"],
    )

    # Users may set it; if omitted, we use DEFAULT_MAX_LENGTH (and clamp per-family).
    max_length: Optional[int] = Field(
        default=None,
        description="Max token length (query + passage) used by FlashRank tokenizer/truncation. "
                    "If omitted, defaults to server DEFAULT_MAX_LENGTH (then clamped per model family).",
        examples=[128, 256, 512, 1024],
    )

    passages: List[Passage] = Field(
        ...,
        description="Candidate passages to rerank.",
        min_length=1,
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is the capital of France?",
                "model": "pairwise_nano",
                "max_length": 128,
                "passages": [
                    {
                        "id": 1,
                        "text": "France is a country in Western Europe known for its rich history, cuisine, and cultural influence. The country has several major cities including Marseille, Lyon, and Paris."
                    },
                    {
                        "id": 2,
                        "text": "Paris is the capital and most populous city of France. It is the political, economic, and cultural center of the country."
                    },
                    {
                        "id": 3,
                        "text": "Lyon is the gastronomic capital and a major city in France known for its culinary traditions and central location."
                    }
                ]
            }
        }
    )

    @model_validator(mode="after")
    def _validate_family_limits(self) -> "RerankIn":
        meta = MODEL_REGISTRY.get(self.model)
        if not meta:
            # Defensive (Enum should prevent this)
            raise ValueError(f"Unknown model: {self.model}")

        family: ModelFamily = meta["family"]

        # Default max_length if user didn't set it
        max_len = self.max_length if self.max_length is not None else DEFAULT_MAX_LENGTH

        # Clamp per family
        if family == ModelFamily.pairwise:
            cap = PAIRWISE_MAX_MAX_LENGTH
        else:
            cap = LISTWISE_MAX_MAX_LENGTH

            # Extra listwise guardrail: limit number of passages
            if len(self.passages) > LISTWISE_MAX_PASSAGES:
                raise ValueError(
                    f"Too many passages for listwise rerank: {len(self.passages)} (max {LISTWISE_MAX_PASSAGES})"
                )

        # Validate bounds
        if max_len < 1:
            raise ValueError("max_length must be >= 1")
        if max_len > cap:
            raise ValueError(f"max_length too large for {family.value} model (max {cap})")

        # Store back the effective value
        self.max_length = max_len
        return self


class RankedPassage(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Union[int, str]
    text: str
    meta: Optional[Dict[str, JSONValue]] = None
    score: Optional[float] = Field( 
        default=None,
        description="Relevance score returned by FlashRank (may be absent for listwise models).",
    )



class RerankOut(BaseModel):
    model: str = Field(..., description="Resolved full FlashRank model name.")
    family: ModelFamily = Field(..., description="Reranker family (pairwise or listwise).")
    max_length: int = Field(..., description="Effective max_length used for this request.")
    query: str
    results: List[RankedPassage]


# -----------------------------
# Ranker cache (one per actual model + max_length + cache_dir)
# -----------------------------

_ranker_cache: Dict[Tuple[str, int, str], Ranker] = {}

def get_ranker(actual_model_name: str, max_length: int) -> Ranker:
    key = (actual_model_name, max_length, CACHE_DIR)
    kwargs = {"model_name": actual_model_name, "max_length": max_length, "cache_dir": CACHE_DIR}
    ranker = Ranker(**kwargs)
    if ranker is None:
        kwargs = {"model_name": actual_model_name, "max_length": max_length}
        if CACHE_DIR:
            kwargs["cache_dir"] = CACHE_DIR
        try:
            ranker = Ranker(**kwargs)
        except Exception as e:
            # Helpful error in case listwise deps/models aren't installed
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize ranker for model '{actual_model_name}' "
                       f"(max_length={max_length}). Error: {e}"
            ) from e
        _ranker_cache[key] = ranker
    return ranker


def resolve_model(model_id: ModelId) -> Tuple[str, ModelFamily]:
    meta = MODEL_REGISTRY.get(model_id)
    if not meta:
        # Defensive (Enum should prevent this)
        raise HTTPException(status_code=422, detail=f"Unknown model id: {model_id}")
    return meta["model_name"], meta["family"]