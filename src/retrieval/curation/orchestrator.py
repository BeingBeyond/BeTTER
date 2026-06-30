"""Assets curation orchestrator.

This module provides a clean, script-friendly pipeline that connects:
1) text retrieval (UID search)
2) UID download (GLB paths)
3) physics preprocessing adapter

It is intentionally independent from Isaac Sim UI concerns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from src.retrieval.client import RetrieverClient
else:
    RetrieverClient = Any

from .contracts import (
    AssetCandidate,
    CurationRequest,
    CurationResult,
    PhysicsPreprocessInput,
    PreparedAsset,
)
from .physics_adapter import PhysicsPreprocessingAdapter

logger = logging.getLogger(__name__)


class AssetsCurationOrchestrator:
    """Orchestrates retrieval + preprocessing for assets curation."""

    def __init__(
        self,
        retriever_client: RetrieverClient,
        physics_adapter: PhysicsPreprocessingAdapter,
    ):
        self.retriever_client = retriever_client
        self.physics_adapter = physics_adapter

    def curate(self, req: CurationRequest) -> CurationResult:
        """Run a full curation pass and return structured results."""
        output_dir = Path(req.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        errors: List[str] = []

        # 1) Retrieve UIDs
        retrieved_uids = self._retrieve_uids(req)

        # 2) Blacklist filtering
        filtered_uids, skipped_blacklisted = self._filter_blacklist(
            retrieved_uids,
            req.blacklist_uids,
        )

        # 3) Download GLBs
        downloaded_paths = self._download_paths(req, filtered_uids)

        # 4) Build candidates
        candidates = self._build_candidates(filtered_uids, downloaded_paths)

        # 5) Preprocess candidates (up to max_prepare)
        prepared_assets = self._prepare_candidates(req, candidates, errors)

        return CurationResult(
            request=req,
            retrieved_uids=retrieved_uids,
            considered_candidates=candidates,
            prepared_assets=prepared_assets,
            skipped_blacklisted=skipped_blacklisted,
            errors=errors,
        )

    def _retrieve_uids(self, req: CurationRequest) -> List[str]:
        return self.retriever_client.search(
            prompt=req.retrieval.prompt,
            top_k=req.retrieval.top_k,
            offset=req.retrieval.offset,
            filters=req.retrieval.filters,
        )

    @staticmethod
    def _filter_blacklist(uids: List[str], blacklist: set[str]) -> tuple[List[str], List[str]]:
        if not blacklist:
            return uids, []
        filtered: List[str] = []
        skipped: List[str] = []
        for uid in uids:
            if uid in blacklist:
                skipped.append(uid)
            else:
                filtered.append(uid)
        return filtered, skipped

    def _download_paths(
        self,
        req: CurationRequest,
        uids: List[str],
    ) -> Dict[str, str]:
        if not uids:
            return {}
        return self.retriever_client.download(
            uids=uids,
            download_dir=req.retrieval.download_dir,
            timeout_seconds=req.retrieval.download_timeout_seconds,
            max_workers=req.retrieval.download_workers,
        )

    @staticmethod
    def _build_candidates(uids: List[str], downloaded_paths: Dict[str, str]) -> List[AssetCandidate]:
        candidates: List[AssetCandidate] = []
        for idx, uid in enumerate(uids):
            path_str = downloaded_paths.get(uid)
            candidates.append(
                AssetCandidate(
                    uid=uid,
                    rank=idx,
                    glb_path=Path(path_str) if path_str else None,
                )
            )
        return candidates

    def _prepare_candidates(
        self,
        req: CurationRequest,
        candidates: List[AssetCandidate],
        errors: List[str],
    ) -> List[PreparedAsset]:
        prepared: List[PreparedAsset] = []

        for cand in candidates:
            if len(prepared) >= req.max_prepare:
                break

            if cand.glb_path is None:
                errors.append(f"missing downloaded path for uid={cand.uid}")
                continue

            preprocess_input = PhysicsPreprocessInput(
                uid=cand.uid,
                glb_path=cand.glb_path,
                output_dir=Path(req.output_dir),
                session_id=req.session_id,
                physics_type=req.physics.physics_type,
                scale_range=req.physics.scale_range,
                mass_range=req.physics.mass_range,
                materials=list(req.physics.materials),
                use_coacd=req.physics.use_coacd,
                coacd_threshold=req.physics.coacd_threshold,
                convert_timeout_seconds=req.physics.convert_timeout_seconds,
                temp_usd_dir=req.physics.temp_usd_dir,
                cleanup_temp=req.physics.cleanup_temp,
                keep_temp_on_failure=req.physics.keep_temp_on_failure,
                bake_transforms_to_mesh=req.physics.bake_transforms_to_mesh,
                normal_mode=req.physics.normal_mode,
                enable_icp_alignment=req.physics.enable_icp_alignment,
                canonical_usd_path=req.physics.canonical_usd_path,
                canonical_mesh_prim_path=req.physics.canonical_mesh_prim_path,
                icp_max_iterations=req.physics.icp_max_iterations,
                icp_tolerance=req.physics.icp_tolerance,
                icp_sample_points=req.physics.icp_sample_points,
                icp_max_corr_distance=req.physics.icp_max_corr_distance,
                icp_seed=req.physics.icp_seed,
                icp_rmse_threshold=req.physics.icp_rmse_threshold,
                metadata={
                    "rank": cand.rank,
                    "retrieval_metadata": cand.retrieval_metadata,
                },
            )

            prepared_asset = self.physics_adapter.preprocess(preprocess_input)
            prepared.append(prepared_asset)

        return prepared


def curate_assets(
    req: CurationRequest,
    retriever_client: RetrieverClient,
    physics_adapter: PhysicsPreprocessingAdapter,
) -> CurationResult:
    """Convenience wrapper for one-off curation calls."""
    orchestrator = AssetsCurationOrchestrator(
        retriever_client=retriever_client,
        physics_adapter=physics_adapter,
    )
    return orchestrator.curate(req)
