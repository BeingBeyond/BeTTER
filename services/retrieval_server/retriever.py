"""
Core Retriever for Objaverse asset search.

Runs inside the retrieval server process (DuoduoCLIP environment).
All heavy resources — metadata, FAISS index, DuoduoCLIP model — are loaded
once at startup and shared across requests.

Filter application is driven by the FilterChain deserialized from each
incoming request.  Custom callable filters can additionally be pre-registered
server-side via ``register_custom_filter``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Dict, List, Optional

import faiss
import h5py
import numpy as np
import objaverse
import torch
import torch.nn.functional as F

from services.retrieval_server.duoduoclip_loader import load_duoduoclip_class
from src.retrieval.filters import CustomFilter, FilterChain

logger = logging.getLogger(__name__)


class Retriever:
    """
    Text-to-shape retriever backed by DuoduoCLIP and Objaverse metadata.

    Initialization loads all metadata sources into memory and builds a FAISS
    index over pre-computed shape embeddings.  This is intentionally done once
    at server startup — it is expensive but amortised over all requests.

    Custom server-side filters can be registered with ``register_custom_filter``
    and referenced by name from ``CustomFilter`` specs sent by the client.

    Args:
        model_checkpoint: Absolute path to the DuoduoCLIP .ckpt file.
        embeddings_path:  Directory containing ``shape_emb_objaverse.h5`` and
                          ``shape_emb_objaverse_model_to_idx.json``.
        metadata_root:    Root directory for Objaverse metadata JSON files.
        duoduoclip_root:  Root directory of the DuoduoCLIP source tree.
        default_download_dir: Fallback download directory when requests omit one.
    """

    def __init__(
        self,
        model_checkpoint: str,
        embeddings_path: str,
        metadata_root: str,
        duoduoclip_root: str,
        default_download_dir: str = "/share/tmp/downloads",
    ):
        self.default_download_dir = default_download_dir
        os.makedirs(default_download_dir, exist_ok=True)

        # Server-side custom filter registry
        self._custom_filters: Dict[str, Callable[[str, Dict[str, Any]], bool]] = {}

        # ----------------------------------------------------------------
        # Metadata (stays in memory for the lifetime of the server)
        # ----------------------------------------------------------------
        logger.info("Loading metadata from %s", metadata_root)

        # Layer 1 – LVIS: category → [uids]
        lvis_path = os.path.join(metadata_root, "lvis/lvis-annotations.json")
        with open(lvis_path, encoding="utf-8") as f:
            self._lvis: Dict[str, List[str]] = json.load(f)
        # Reverse index: uid → [categories]
        self._uid_to_lvis: Dict[str, List[str]] = {}
        for cat, uids in self._lvis.items():
            for uid in uids:
                self._uid_to_lvis.setdefault(uid, []).append(cat)
        logger.info("LVIS: %d categories, %d UIDs indexed", len(self._lvis), len(self._uid_to_lvis))

        # Layer 2 – Step-1X-3D quality set
        step1x_path = os.path.join(metadata_root, "step1x-3d/high_quality_objaverse_1.0.json")
        with open(step1x_path, encoding="utf-8") as f:
            step1x_data = json.load(f)
        self._step1x: set = (
            set(step1x_data) if isinstance(step1x_data, list) else set(step1x_data.keys())
        )
        logger.info("Step-1X-3D: %d high-quality UIDs", len(self._step1x))

        # Layer 3 – Official metadata (license)
        official_path = os.path.join(metadata_root, "official/objaverse_meta_merged_simple.json")
        with open(official_path, encoding="utf-8") as f:
            self._official: Dict[str, Dict] = json.load(f)
        logger.info("Official metadata: %d entries", len(self._official))

        # Layer 4 – Objaverse++ quality annotations
        plusplus_path = os.path.join(metadata_root, "objaverse++/annotated_800k.json")
        with open(plusplus_path, encoding="utf-8") as f:
            plusplus_list = json.load(f)
        self._plusplus: Dict[str, Dict[str, Any]] = {}
        for item in plusplus_list:
            uid = item.get("UID")
            if not uid:
                continue
            normalized = {
                k: (True if v == "true" else False if v == "false" else v)
                for k, v in item.items()
                if k != "UID"
            }
            self._plusplus[uid] = normalized
        logger.info("Objaverse++: %d annotated objects", len(self._plusplus))

        # Pali merged annotations (type / material / caption)
        pali_path = os.path.join(metadata_root, "pali_merged/pali_merged_annotations.json")
        with open(pali_path, encoding="utf-8") as f:
            self._pali: Dict[str, Dict] = json.load(f)
        logger.info("Pali annotations: %d entries", len(self._pali))

        # ----------------------------------------------------------------
        # Shape embeddings + FAISS index
        # ----------------------------------------------------------------
        logger.info("Loading shape embeddings from %s", embeddings_path)
        h5_path = os.path.join(embeddings_path, "shape_emb_objaverse.h5")
        with h5py.File(h5_path, "r") as f:
            raw = f["shape_feat"][:].astype(np.float32)

        emb_t = torch.from_numpy(raw)
        emb_t = F.normalize(emb_t, dim=1)
        self._embeddings = emb_t.numpy()

        logger.info("Building FAISS index (%d vectors, dim=%d)…", *self._embeddings.shape)
        self._index = faiss.IndexFlatIP(self._embeddings.shape[1])
        self._index.add(self._embeddings)

        idx_path = os.path.join(embeddings_path, "shape_emb_objaverse_model_to_idx.json")
        with open(idx_path) as f:
            uid_to_idx = json.load(f)
        self._idx_to_uid: Dict[int, str] = {v: k for k, v in uid_to_idx.items()}
        logger.info("FAISS index ready: %d entries", len(self._idx_to_uid))

        # ----------------------------------------------------------------
        # DuoduoCLIP model
        # ----------------------------------------------------------------
        logger.info("Loading DuoduoCLIP from %s", duoduoclip_root)
        DuoduoCLIP = load_duoduoclip_class(duoduoclip_root)

        self._duoduoclip = DuoduoCLIP.load_from_checkpoint(model_checkpoint)
        self._duoduoclip.eval()
        if torch.cuda.is_available():
            self._duoduoclip.cuda()
            logger.info("DuoduoCLIP loaded on GPU")
        else:
            logger.warning("DuoduoCLIP running on CPU — expect slow inference")

        logger.info("Retriever ready")

    # ------------------------------------------------------------------
    # Custom filter registry (server-side only)
    # ------------------------------------------------------------------

    def register_custom_filter(
        self,
        name: str,
        fn: Callable[[str, Dict[str, Any]], bool],
    ) -> None:
        """
        Register a named callable for use with ``CustomFilter``.

        The callable receives ``(uid, unified_metadata)`` and returns bool.

        Example:
            retriever.register_custom_filter(
                "high_density",
                lambda uid, meta: meta.get("plusplus", {}).get("density") == "high",
            )
        """
        self._custom_filters[name] = fn
        logger.info("Registered custom filter: %s", name)

    def unregister_custom_filter(self, name: str) -> None:
        self._custom_filters.pop(name, None)
        logger.info("Unregistered custom filter: %s", name)

    # ------------------------------------------------------------------
    # Metadata access
    # ------------------------------------------------------------------

    def get_unified_metadata(self, uid: str) -> Dict[str, Any]:
        """
        Return a unified metadata dict for a single UID.

        Schema:
            {
                "lvis_categories": List[str],
                "is_step1x_quality": bool,
                "license": str | None,
                "plusplus": dict,
                "pali": dict,
            }
        """
        return {
            "lvis_categories": self._uid_to_lvis.get(uid, []),
            "is_step1x_quality": uid in self._step1x,
            "license": self._official.get(uid, {}).get("license"),
            "plusplus": self._plusplus.get(uid, {}),
            "pali": self._pali.get(uid, {}),
        }

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def _encode_text(self, prompts: List[str]) -> np.ndarray:
        tokens = self._duoduoclip.tokenizer(prompts)
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            feat = self._duoduoclip.duoduoclip.encode_text(tokens)
            feat = F.normalize(feat, dim=1)
        return feat.cpu().numpy().astype(np.float32)

    def search(
        self,
        prompts: List[str],
        top_k: int,
        offset: int = 0,
        filter_chain_data: Optional[Dict] = None,
        exclude_uids: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """
        Search for objects matching the given text prompts.

        Args:
            prompts:           List of text prompts.
            top_k:             Number of results per prompt after filtering.
            offset:            Pagination offset for deterministic incremental fetch.
            filter_chain_data: Serialized FilterChain dict from the client,
                               or None for no filtering.

        Returns:
            List of UID lists, one per prompt.
        """
        chain = FilterChain.model_validate(filter_chain_data) if filter_chain_data else FilterChain()
        excluded_uid_set = {uid for uid in (exclude_uids or []) if uid}

        # Inject callables for CustomFilters
        for i, f in enumerate(chain.filters):
            if isinstance(f, CustomFilter):
                fn = self._custom_filters.get(f.name)
                if fn is None:
                    raise ValueError(
                        f"CustomFilter '{f.name}' is not registered on the server. "
                        f"Available: {list(self._custom_filters)}"
                    )
                chain.filters[i] = f.inject_callable(fn)

        has_filters = not chain.is_empty()
        wanted = top_k + max(0, offset)
        search_k = wanted * 5 if has_filters else wanted

        text_emb = self._encode_text(prompts)
        _, indices = self._index.search(text_emb, search_k)

        results = []
        for i, prompt in enumerate(prompts):
            uids = [self._idx_to_uid[idx] for idx in indices[i] if idx in self._idx_to_uid]
            if excluded_uid_set:
                uids = [uid for uid in uids if uid not in excluded_uid_set]

            if has_filters:
                filtered = []
                for uid in uids:
                    meta = self.get_unified_metadata(uid)
                    if chain.apply_all(uid, meta):
                        filtered.append(uid)
                    if len(filtered) >= wanted:
                        break
                uids = filtered
            else:
                uids = uids[:wanted]

            # Deterministic pagination window
            start = max(0, offset)
            end = start + top_k
            uids = uids[start:end]

            logger.info("prompt %r → %d results", prompt, len(uids))
            results.append(uids)

        return results

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(
        self,
        uids: List[str],
        download_dir: Optional[str] = None,
        timeout_seconds: float = 120.0,
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        """
        Download GLB files via objaverse with timeout-aware partial return.

        Args:
            uids:             UIDs to download.
            download_dir:     Local directory to save files (falls back to default).
            timeout_seconds:  Per-UID timeout budget in seconds.
            max_workers:      Parallel worker count for per-UID download tasks.

        Returns:
            {
                "paths": Dict[uid, path],
                "timed_out_uids": List[str],
                "failed_uids": List[str],
                "partial": bool,
            }
        """
        target_dir = download_dir or self.default_download_dir
        os.makedirs(target_dir, exist_ok=True)

        logger.info(
            "Downloading %d objects to %s (timeout=%.1fs, workers=%d)",
            len(uids),
            target_dir,
            timeout_seconds,
            max_workers,
        )

        final_paths: Dict[str, str] = {}
        timed_out_uids: List[str] = []
        failed_uids: List[str] = []

        def _download_one(uid: str) -> Optional[str]:
            result = objaverse.load_objects(uids=[uid], download_processes=1)
            return result.get(uid)

        executor = ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(uids) or 1)))
        try:
            future_to_uid = {executor.submit(_download_one, uid): uid for uid in uids}
            for future, uid in future_to_uid.items():
                try:
                    src_path = future.result(timeout=timeout_seconds)
                except FuturesTimeoutError:
                    timed_out_uids.append(uid)
                    logger.warning("Download timeout for uid=%s", uid)
                    future.cancel()
                    continue
                except Exception as e:
                    failed_uids.append(uid)
                    logger.warning("Download failed for uid=%s error=%s", uid, e)
                    continue

                if not src_path or not os.path.exists(src_path):
                    failed_uids.append(uid)
                    logger.warning("Missing downloaded source file for uid=%s", uid)
                    continue

                dst_path = os.path.join(target_dir, f"{uid}.glb")
                if os.path.abspath(src_path) != os.path.abspath(dst_path):
                    shutil.copy2(src_path, dst_path)
                final_paths[uid] = dst_path
        finally:
            # Do not block the request on straggler tasks once partial results
            # are already available.
            executor.shutdown(wait=False, cancel_futures=True)

        partial = len(final_paths) < len(uids)
        logger.info(
            "Download finished: requested=%d success=%d timeout=%d failed=%d",
            len(uids),
            len(final_paths),
            len(timed_out_uids),
            len(failed_uids),
        )

        return {
            "paths": final_paths,
            "timed_out_uids": timed_out_uids,
            "failed_uids": failed_uids,
            "partial": partial,
        }
