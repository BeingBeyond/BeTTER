from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .light_state import apply_light_state, collect_light_state, scale_light_intensities
from .manager_v2 import ResolvedBackgroundVariant


@dataclass
class BackgroundScene:
    resolved_variant: ResolvedBackgroundVariant
    prim_path: str = "/World/Background"
    loaded: bool = False
    active: bool = False
    baseline_lights: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def load_into_stage(self, stage: Any, activate_on_load: bool = True) -> None:
        if not self.loaded:
            prim = stage.GetPrimAtPath(self.prim_path)
            if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
                prim = stage.DefinePrim(self.prim_path, "Xform")

            payloads = prim.GetPayloads()
            payloads.ClearPayloads()
            payloads.AddPayload(str(self.resolved_variant.root_layer_path))
            self.loaded = True

        if activate_on_load:
            self.activate(stage)
        else:
            self.deactivate(stage)

    def activate(self, stage: Any) -> None:
        if not self.loaded:
            raise RuntimeError("Background scene must be loaded before activation")
        prim = stage.GetPrimAtPath(self.prim_path)
        if prim is None:
            raise RuntimeError(f"Background prim not found for activation: {self.prim_path}")
        if hasattr(prim, "SetActive"):
            prim.SetActive(True)
        if hasattr(prim, "Load"):
            prim.Load()
        self.active = True

    def deactivate(self, stage: Any) -> None:
        if not self.loaded:
            return
        prim = stage.GetPrimAtPath(self.prim_path)
        if prim is None:
            raise RuntimeError(f"Background prim not found for deactivation: {self.prim_path}")
        if hasattr(prim, "SetActive"):
            prim.SetActive(False)
        self.active = False

    def capture_light_baseline(self, stage: Any) -> Dict[str, Dict[str, Any]]:
        if not self.loaded:
            raise RuntimeError("Background scene must be loaded before capturing light baseline")
        self.baseline_lights = collect_light_state(stage, self.prim_path)
        return dict(self.baseline_lights)

    def reset_lights_to_baseline(self, stage: Any) -> None:
        if not self.baseline_lights:
            return
        apply_light_state(stage, self.baseline_lights)

    def apply_light_scale(self, stage: Any, factor: float) -> None:
        if not self.baseline_lights:
            self.capture_light_baseline(stage)
        scale_light_intensities(stage, self.baseline_lights, factor=factor)

    def apply_hdri(
        self,
        stage: Any,
        texture_path: Optional[str | Path] = None,
        intensity: Optional[float] = None,
        rotation_deg: Optional[float] = None,
    ) -> None:
        if not self.loaded:
            raise RuntimeError("Background scene must be loaded before applying HDRI settings")

        for prim_path, state in self.baseline_lights.items() if self.baseline_lights else []:
            if state.get("type_name") != "DomeLight":
                continue
            prim = stage.GetPrimAtPath(prim_path)
            if prim is None:
                continue

            if texture_path is not None:
                attr = prim.GetAttribute("texture:file")
                if attr and (not hasattr(attr, "IsValid") or attr.IsValid()):
                    attr.Set(str(texture_path))

            if intensity is not None:
                attr = prim.GetAttribute("intensity")
                if attr and (not hasattr(attr, "IsValid") or attr.IsValid()):
                    attr.Set(float(intensity))

            if rotation_deg is not None:
                rotate_attr = state.get("rotate_z_attr")
                if rotate_attr:
                    attr = prim.GetAttribute(rotate_attr)
                    if attr and (not hasattr(attr, "IsValid") or attr.IsValid()):
                        attr.Set(float(rotation_deg))

    def unload(self, stage: Any) -> None:
        if not self.loaded:
            return
        prim = stage.GetPrimAtPath(self.prim_path)
        if prim is None:
            raise RuntimeError(f"Background prim not found for unload: {self.prim_path}")
        if hasattr(prim, "SetActive"):
            prim.SetActive(False)
        if hasattr(prim, "Unload"):
            prim.Unload()
        payloads = prim.GetPayloads() if hasattr(prim, "GetPayloads") else None
        if payloads is not None and hasattr(payloads, "ClearPayloads"):
            payloads.ClearPayloads()
        self.active = False
        self.loaded = False
