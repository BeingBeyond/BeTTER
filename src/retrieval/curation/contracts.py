"""Contracts for the assets curation pipeline.

These are in-process data contracts used by the orchestrator and adapters.
They intentionally use dataclasses (not Pydantic) to match BeTTER's existing
internal contract style in scene registries/managers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Tuple

if TYPE_CHECKING:
    from src.retrieval.filters import FilterChain
else:
    FilterChain = Any


@dataclass(frozen=True)
class RetrievalInput:
    """Input contract for retrieval and download stages."""

    prompt: str
    top_k: int = 10
    offset: int = 0
    download_dir: str = "/share/tmp/downloads"
    download_timeout_seconds: float = 120.0
    download_workers: int = 4
    filters: Optional[FilterChain] = None


@dataclass(frozen=True)
class PhysicsPreparationConfig:
    """Configuration contract for physics preprocessing behavior."""

    physics_type: str = "rigid"
    scale_range: Tuple[float, float] = (0.1, 0.3)
    mass_range: Tuple[float, float] = (0.1, 1.0)
    materials: List[str] = field(default_factory=list)
    use_coacd: bool = True
    coacd_threshold: float = 0.05
    convert_timeout_seconds: float = 180.0
    temp_usd_dir: Optional[Path] = Path("/share/tmp")
    cleanup_temp: bool = True
    keep_temp_on_failure: bool = False
    bake_transforms_to_mesh: bool = True
    normal_mode: str = "preserve"
    enable_icp_alignment: bool = False
    canonical_usd_path: Optional[Path] = None
    canonical_mesh_prim_path: Optional[str] = None
    icp_max_iterations: int = 40
    icp_tolerance: float = 1e-5
    icp_sample_points: int = 5000
    icp_max_corr_distance: float = 0.05
    icp_seed: int = 0
    icp_rmse_threshold: float = 0.02


@dataclass(frozen=True)
class PhysicsPreprocessInput:
    """Input contract for physics preprocessing adapter."""

    uid: str
    glb_path: Path
    output_dir: Path
    session_id: Optional[str] = None
    physics_type: str = "rigid"
    scale_range: Tuple[float, float] = (0.1, 0.3)
    mass_range: Tuple[float, float] = (0.1, 1.0)
    materials: List[str] = field(default_factory=list)
    use_coacd: bool = True
    coacd_threshold: float = 0.05
    convert_timeout_seconds: float = 180.0
    temp_usd_dir: Optional[Path] = Path("/share/tmp")
    cleanup_temp: bool = True
    keep_temp_on_failure: bool = False
    bake_transforms_to_mesh: bool = True
    normal_mode: str = "preserve"
    enable_icp_alignment: bool = False
    canonical_usd_path: Optional[Path] = None
    canonical_mesh_prim_path: Optional[str] = None
    icp_max_iterations: int = 40
    icp_tolerance: float = 1e-5
    icp_sample_points: int = 5000
    icp_max_corr_distance: float = 0.05
    icp_seed: int = 0
    icp_rmse_threshold: float = 0.02
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssetCandidate:
    """A retrieved candidate asset before preprocessing."""

    uid: str
    rank: int
    glb_path: Optional[Path] = None
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedAsset:
    """A candidate asset after preprocessing."""

    uid: str
    source_glb_path: Path
    prepared_asset_path: Path
    physics_applied: bool
    adapter_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EditorSessionRequest:
    """Request contract for opening a preview or editable asset session."""

    uid: str
    source_asset_path: Path
    session_dir: Path
    session_id: Optional[str] = None
    mode: Literal["preview", "editable"] = "editable"
    temp_usd_dir: Optional[Path] = Path("/share/tmp")
    convert_timeout_seconds: float = 180.0
    root_prim_name: str = "Object"
    flatten_for_edit: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EditorSessionAsset:
    """Editor-facing asset session metadata."""

    uid: str
    session_id: str
    mode: Literal["preview", "editable"]
    source_asset_path: Path
    stage_path: Path
    root_prim_path: str
    mesh_prim_path: str
    local_mesh_owned: bool
    temp_usd_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fallback_mode(self) -> str:
        return str(self.metadata.get("fallback_mode") or "none")


@dataclass(frozen=True)
class EditorValidationResult:
    """Validation contract for publish-time asset editor invariants."""

    session_id: str
    stage_path: Path
    root_prim_path: str
    mesh_count: int
    local_mesh_count: int
    xform_op_count: int
    passes_local_mesh_ownership: bool
    passes_no_xform_ops: bool
    passes_default_prim: bool
    is_publish_ready: bool
    issues: Tuple[str, ...] = ()
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurationRequest:
    """Top-level request contract for a curation run."""

    retrieval: RetrievalInput
    output_dir: Path
    max_prepare: int = 3
    blacklist_uids: Set[str] = field(default_factory=set)
    session_id: Optional[str] = None
    physics: PhysicsPreparationConfig = field(default_factory=PhysicsPreparationConfig)


@dataclass(frozen=True)
class CurationResult:
    """Top-level result contract for a curation run."""

    request: CurationRequest
    retrieved_uids: List[str]
    considered_candidates: List[AssetCandidate]
    prepared_assets: List[PreparedAsset]
    skipped_blacklisted: List[str]
    errors: List[str]

    def is_success(self) -> bool:
        """Return True when at least one asset is prepared and no errors occurred."""
        return len(self.prepared_assets) > 0 and len(self.errors) == 0
