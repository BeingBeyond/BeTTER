"""Physics preprocessing adapters for curation pipeline.

This layer isolates the orchestrator from concrete simulation/physics tooling.
V1 provides:
- NoOpPhysicsAdapter: passthrough adapter for environments without Isaac Sim.
- CallablePhysicsAdapter: wrapper for externally provided conversion/preprocess
  callable (e.g., bridge to LoHoBench converter or future BeTTER implementation).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Protocol

from .contracts import PhysicsPreprocessInput, PreparedAsset


class PhysicsPreprocessingAdapter(Protocol):
    """Protocol for physics preprocessing adapters."""

    name: str

    def preprocess(self, req: PhysicsPreprocessInput) -> PreparedAsset:
        """Preprocess an input GLB and return prepared asset metadata."""


class NoOpPhysicsAdapter:
    """Passthrough adapter that performs no preprocessing."""

    name = "noop"

    def preprocess(self, req: PhysicsPreprocessInput) -> PreparedAsset:
        return PreparedAsset(
            uid=req.uid,
            source_glb_path=req.glb_path,
            prepared_asset_path=req.glb_path,
            physics_applied=False,
            adapter_name=self.name,
        )


class CallablePhysicsAdapter:
    """
    Adapter wrapping an external preprocessing callable.

    The wrapped callable must accept ``PhysicsPreprocessInput`` and return a
    ``PreparedAsset``. If the returned ``PreparedAsset.adapter_name`` is empty,
    this adapter name is injected.
    """

    def __init__(
        self,
        preprocess_fn: Callable[[PhysicsPreprocessInput], PreparedAsset],
        name: str = "callable",
    ):
        self._preprocess_fn = preprocess_fn
        self.name = name

    def preprocess(self, req: PhysicsPreprocessInput) -> PreparedAsset:
        result = self._preprocess_fn(req)
        if not isinstance(result, PreparedAsset):
            raise TypeError(
                "CallablePhysicsAdapter preprocess_fn must return PreparedAsset"
            )
        if result.adapter_name:
            return result
        return replace(result, adapter_name=self.name)


class IsaacSimPhysicsAdapter:
    """Full-fidelity Isaac preprocessing adapter (LoHoBench-equivalent steps)."""

    name = "isaac_sim_full"

    def preprocess(self, req: PhysicsPreprocessInput) -> PreparedAsset:
        prepared_usd_path = req.output_dir / f"{req.uid}.usd"
        temp_usd_dir = req.temp_usd_dir if req.temp_usd_dir is not None else Path("/share/tmp")

        from .isaac_pipeline import preprocess_asset

        pipeline_result = preprocess_asset(
            uid=req.uid,
            glb_path=req.glb_path,
            output_usd_path=prepared_usd_path,
            scale_range=req.scale_range,
            mass_range=req.mass_range,
            physics_type=req.physics_type,
            materials=req.materials,
            use_coacd=req.use_coacd,
            coacd_threshold=req.coacd_threshold,
            convert_timeout_seconds=req.convert_timeout_seconds,
            temp_usd_dir=temp_usd_dir,
            cleanup_temp=req.cleanup_temp,
            keep_temp_on_failure=req.keep_temp_on_failure,
            bake_transforms_to_mesh=req.bake_transforms_to_mesh,
            normal_mode=req.normal_mode,
            enable_icp_alignment=req.enable_icp_alignment,
            canonical_usd_path=req.canonical_usd_path,
            canonical_mesh_prim_path=req.canonical_mesh_prim_path,
            icp_max_iterations=req.icp_max_iterations,
            icp_tolerance=req.icp_tolerance,
            icp_sample_points=req.icp_sample_points,
            icp_max_corr_distance=req.icp_max_corr_distance,
            icp_seed=req.icp_seed,
            icp_rmse_threshold=req.icp_rmse_threshold,
        )

        metadata = {
            **req.metadata,
            **pipeline_result,
        }

        return PreparedAsset(
            uid=req.uid,
            source_glb_path=req.glb_path,
            prepared_asset_path=Path(pipeline_result["prepared_usd_path"]),
            physics_applied=bool(pipeline_result.get("physics_applied", False)),
            adapter_name=self.name,
            metadata=metadata,
        )
