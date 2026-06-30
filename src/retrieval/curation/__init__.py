"""Assets curation pipeline API (v1)."""

from .contracts import (
    AssetCandidate,
    CurationRequest,
    CurationResult,
    EditorSessionAsset,
    EditorSessionRequest,
    EditorValidationResult,
    PhysicsPreparationConfig,
    PhysicsPreprocessInput,
    PreparedAsset,
    RetrievalInput,
)
from .isaac_pipeline import (
    apply_icp_to_editor_session,
    bake_editor_session_to_geometry,
    get_editor_session_validation,
    normalize_editor_session,
    open_editor_session,
    publish_editor_session_asset,
    save_editor_session,
    solve_icp_for_editor_session,
    validate_editor_session_asset,
    validate_editor_stage,
)
from .orchestrator import AssetsCurationOrchestrator, curate_assets
from .physics_adapter import (
    CallablePhysicsAdapter,
    IsaacSimPhysicsAdapter,
    NoOpPhysicsAdapter,
    PhysicsPreprocessingAdapter,
)

__all__ = [
    "RetrievalInput",
    "PhysicsPreprocessInput",
    "PhysicsPreparationConfig",
    "AssetCandidate",
    "PreparedAsset",
    "EditorSessionRequest",
    "EditorSessionAsset",
    "EditorValidationResult",
    "CurationRequest",
    "CurationResult",
    "open_editor_session",
    "validate_editor_session_asset",
    "validate_editor_stage",
    "normalize_editor_session",
    "solve_icp_for_editor_session",
    "apply_icp_to_editor_session",
    "bake_editor_session_to_geometry",
    "publish_editor_session_asset",
    "save_editor_session",
    "get_editor_session_validation",
    "PhysicsPreprocessingAdapter",
    "NoOpPhysicsAdapter",
    "CallablePhysicsAdapter",
    "IsaacSimPhysicsAdapter",
    "AssetsCurationOrchestrator",
    "curate_assets",
]
