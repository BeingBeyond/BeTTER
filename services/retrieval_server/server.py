"""
Retrieval server — FastAPI application.

Runs in the DuoduoCLIP Python environment (separate from the simulation stack).
Loads config from configs/retrieval/default.yaml relative to the BeTTER root,
or from the path given by the RETRIEVAL_CONFIG env var.

Start:
    cd <repo-root>
    RETRIEVAL_CONFIG=configs/retrieval/default.yaml \
        uvicorn services.retrieval_server.server:app --host 0.0.0.0 --port 8001

Endpoints:
    POST /search        – text → UIDs
    POST /search_batch  – [texts] → [[UIDs]]
    POST /download      – UIDs → {uid: path}
    GET  /health        – liveness check
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_BETTER_ROOT = os.path.abspath(os.path.join(_SERVER_DIR, "..", ".."))

# Allow local import: from retriever import Retriever
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

# Allow shared package import: from src.retrieval.filters import ...
if _BETTER_ROOT not in sys.path:
    sys.path.insert(0, _BETTER_ROOT)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("retrieval_server")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.environ.get(
    "RETRIEVAL_CONFIG",
    os.path.join(os.path.dirname(_SERVER_DIR), "..", "configs", "retrieval", "default.yaml"),
)
_CONFIG_PATH = os.path.abspath(_CONFIG_PATH)
logger.info("Loading config from %s", _CONFIG_PATH)

with open(_CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

_retriever_cfg = _cfg["retriever"]

# ---------------------------------------------------------------------------
# Retriever — loaded once at startup
# ---------------------------------------------------------------------------
_duoduoclip_root = _retriever_cfg["duoduoclip_root"]

from retriever import Retriever  # noqa: E402

logger.info("Initializing Retriever…")
retriever = Retriever(
    model_checkpoint=_retriever_cfg["model_checkpoint"],
    embeddings_path=_retriever_cfg["embeddings_path"],
    metadata_root=_retriever_cfg["metadata_root"],
    duoduoclip_root=_duoduoclip_root,
    default_download_dir=_retriever_cfg.get("default_download_dir", "/share/tmp/downloads"),
)
logger.info("Retriever ready")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BeTTER Retrieval Server",
    description="Text-to-3D-asset retrieval backed by DuoduoCLIP + Objaverse.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    prompt: str = Field(..., description="Natural-language object description.")
    top_k: int = Field(10, ge=1, le=100, description="Max results to return.")
    offset: int = Field(0, ge=0, description="Deterministic pagination offset.")
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Serialized FilterChain (from FilterChain.model_dump()).",
    )
    exclude_uids: List[str] = Field(
        default_factory=list,
        description="UIDs to omit from ranked results before pagination is applied.",
    )


class SearchResponse(BaseModel):
    uids: List[str]


class SearchBatchRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0, description="Deterministic pagination offset.")
    filters: Optional[Dict[str, Any]] = None
    exclude_uids: List[str] = Field(default_factory=list)


class SearchBatchResponse(BaseModel):
    results: List[List[str]]


class DownloadRequest(BaseModel):
    uids: List[str] = Field(..., min_length=1)
    download_dir: Optional[str] = Field(
        None,
        description="Server-side directory for GLB files. "
                    "Defaults to the value set in config.",
    )
    timeout_seconds: float = Field(120.0, gt=0, le=3600, description="Per-UID download timeout.")
    max_workers: int = Field(4, ge=1, le=32, description="Parallel worker count.")


class DownloadResponse(BaseModel):
    paths: Dict[str, str]
    timed_out_uids: List[str] = Field(default_factory=list)
    failed_uids: List[str] = Field(default_factory=list)
    partial: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness probe — returns 200 when the server is up and the retriever is ready."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """
    Retrieve Objaverse UIDs matching a single text prompt.

    The optional ``filters`` field accepts a serialized ``FilterChain``
    (use ``FilterChain.model_dump()`` on the client side).
    """
    try:
        results = retriever.search(
            prompts=[req.prompt],
            top_k=req.top_k,
            offset=req.offset,
            filter_chain_data=req.filters,
            exclude_uids=req.exclude_uids,
        )
        return SearchResponse(uids=results[0])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search_batch", response_model=SearchBatchResponse)
def search_batch(req: SearchBatchRequest) -> SearchBatchResponse:
    """
    Retrieve UIDs for multiple prompts in a single call.

    All prompts share the same FilterChain and ``top_k`` setting.
    """
    try:
        results = retriever.search(
            prompts=req.prompts,
            top_k=req.top_k,
            offset=req.offset,
            filter_chain_data=req.filters,
            exclude_uids=req.exclude_uids,
        )
        return SearchBatchResponse(results=results)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Batch search failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download", response_model=DownloadResponse)
def download(req: DownloadRequest) -> DownloadResponse:
    """
    Download GLB files for the given UIDs.

    Returns a mapping of UID → absolute file path on the server host.
    The caller is responsible for ensuring the path is accessible (e.g. shared
    filesystem or the same machine).
    """
    try:
        download_result = retriever.download(
            uids=req.uids,
            download_dir=req.download_dir,
            timeout_seconds=req.timeout_seconds,
            max_workers=req.max_workers,
        )
        return DownloadResponse(**download_result)
    except Exception as e:
        logger.exception("Download failed")
        raise HTTPException(status_code=500, detail=str(e))
