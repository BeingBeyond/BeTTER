"""
BeTTER retrieval module.

Provides a lightweight HTTP client and typed filter system for retrieving
Objaverse 3D assets by text prompt.  The heavy lifting (DuoduoCLIP inference,
FAISS search, metadata loading) runs in a separate server process under
services/retrieval_server/ to avoid dependency conflicts with the simulation
stack.

Typical usage:
    from src.retrieval import RetrieverClient
    from src.retrieval.filters import (
        Step1xQualityFilter,
        LicenseFilter,
        ObjAversePlusPlusFilter,
        LvisFilter,
        CustomFilter,
        FilterChain,
    )

    client = RetrieverClient("http://192.168.20.173:8001")
    client.register_filter(Step1xQualityFilter())
    client.register_filter(LicenseFilter(allowed=["by", "by-sa", "cc0"]))
    client.register_filter(ObjAversePlusPlusFilter(conditions={
        "is_scene": False, "is_multi_object": False,
    }))

    uids  = client.search("a red coffee mug", top_k=5)
    paths = client.download(uids, download_dir="/tmp/glbs")
"""

try:
    from .client import RetrieverClient
except ModuleNotFoundError:  # optional in minimal test/runtime envs
    RetrieverClient = None

from .curation import (
    AssetCandidate,
    AssetsCurationOrchestrator,
    CallablePhysicsAdapter,
    CurationRequest,
    CurationResult,
    NoOpPhysicsAdapter,
    PhysicsPreprocessInput,
    PhysicsPreprocessingAdapter,
    PreparedAsset,
    RetrievalInput,
    curate_assets,
)

try:
    from .filters import (
        AnyFilter,
        BaseFilter,
        CustomFilter,
        FilterChain,
        LicenseFilter,
        LvisFilter,
        ObjAversePlusPlusFilter,
        Step1xQualityFilter,
    )
except ModuleNotFoundError:  # optional in minimal envs
    AnyFilter = None
    BaseFilter = None
    CustomFilter = None
    FilterChain = None
    LicenseFilter = None
    LvisFilter = None
    ObjAversePlusPlusFilter = None
    Step1xQualityFilter = None

__all__ = [
    "RetrieverClient",
    # Filters
    "BaseFilter",
    "LvisFilter",
    "Step1xQualityFilter",
    "LicenseFilter",
    "ObjAversePlusPlusFilter",
    "CustomFilter",
    "AnyFilter",
    "FilterChain",
    # Curation
    "RetrievalInput",
    "PhysicsPreprocessInput",
    "AssetCandidate",
    "PreparedAsset",
    "CurationRequest",
    "CurationResult",
    "PhysicsPreprocessingAdapter",
    "NoOpPhysicsAdapter",
    "CallablePhysicsAdapter",
    "AssetsCurationOrchestrator",
    "curate_assets",
]
