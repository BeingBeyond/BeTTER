from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .schema import AssetCandidateSpec, ResolvedAssetSpec, TaskSpec


@dataclass(frozen=True)
class AssetRegistryResolution:
    source_uid: str
    asset_key: str
    asset_path: Path
    registry_usd: str
    registry_meta: str | None = None

    def to_spec(self) -> ResolvedAssetSpec:
        return ResolvedAssetSpec(
            source_uid=self.source_uid,
            asset_key=self.asset_key,
            asset_path=str(self.asset_path),
            registry_usd=self.registry_usd,
            registry_meta=self.registry_meta,
        )


class AssetRegistryIndex:
    """Resolve task asset candidates to concrete registry USD paths."""

    def __init__(self, registry_root: str | Path) -> None:
        self.registry_root = Path(registry_root).expanduser().resolve()

    def resolve_candidate(
        self,
        task_spec: TaskSpec,
        instance_id: str,
        candidate: AssetCandidateSpec,
    ) -> AssetRegistryResolution:
        if candidate.registry_usd:
            usd_path = self._resolve_registry_ref(candidate.registry_usd)
            if not usd_path.exists():
                raise FileNotFoundError(
                    f"registry_usd for {task_spec.task_id}/{instance_id} does not exist: {usd_path}"
                )
            return self._build_resolution(candidate, usd_path, registry_meta=candidate.registry_meta)

        task_registry_dir = self.registry_root / task_spec.template_type / task_spec.task_id

        if candidate.asset_key:
            exact_path = task_registry_dir / f"{candidate.asset_key}.usd"
            if exact_path.exists():
                return self._build_resolution(candidate, exact_path, registry_meta=candidate.registry_meta)

        matches = sorted(task_registry_dir.glob(f"{candidate.source_uid}__*.usd"))
        if len(matches) == 1:
            return self._build_resolution(candidate, matches[0], registry_meta=candidate.registry_meta)
        if len(matches) > 1:
            pretty = ", ".join(str(path) for path in matches[:5])
            suffix = "" if len(matches) <= 5 else f", ... ({len(matches)} total)"
            raise ValueError(
                f"Ambiguous source_uid for {task_spec.task_id}/{instance_id}: "
                f"{candidate.source_uid} matched {len(matches)} registry USDs: {pretty}{suffix}. "
                "Provide registry_usd or asset_key in asset_bindings.yaml."
            )

        raise FileNotFoundError(
            f"No registry USD found for {task_spec.task_id}/{instance_id} "
            f"source_uid={candidate.source_uid} under {task_registry_dir}"
        )

    def _resolve_registry_ref(self, registry_ref: str) -> Path:
        path = Path(registry_ref).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.registry_root / path).resolve()

    def _build_resolution(
        self,
        candidate: AssetCandidateSpec,
        usd_path: Path,
        registry_meta: str | None,
    ) -> AssetRegistryResolution:
        asset_key = candidate.asset_key or usd_path.stem
        registry_usd = self._relative_registry_ref(usd_path)
        resolved_meta = self._resolve_meta_ref(usd_path, registry_meta)
        return AssetRegistryResolution(
            source_uid=candidate.source_uid,
            asset_key=asset_key,
            asset_path=usd_path.resolve(),
            registry_usd=registry_usd,
            registry_meta=resolved_meta,
        )

    def _relative_registry_ref(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(self.registry_root))
        except ValueError:
            return str(resolved)

    def _resolve_meta_ref(self, usd_path: Path, registry_meta: str | None) -> str | None:
        if registry_meta:
            meta_path = self._resolve_registry_ref(registry_meta)
            if meta_path.exists():
                return self._relative_registry_ref(meta_path)
            return registry_meta

        inferred = usd_path.with_suffix(".meta.json")
        if inferred.exists():
            return self._relative_registry_ref(inferred)
        return None
