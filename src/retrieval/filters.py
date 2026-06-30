"""
Retrieval filter definitions for Objaverse asset search.

Filters are Pydantic models — JSON-serializable and sent from client to server
as part of each search request. The server holds all metadata in memory and
applies whichever filters the client registers.

Built-in filters (logic is fixed, parameters travel over HTTP):
    LvisFilter               – hard category match via LVIS annotations
    Step1xQualityFilter      – Step-1X-3D high-quality set membership
    LicenseFilter            – license string whitelist
    ObjAversePlusPlusFilter  – Objaverse++ attribute conditions

Custom filters (logic is user-defined, must be pre-registered on the server):
    CustomFilter             – references a named callable registered server-side

Usage (client side):
    chain = FilterChain()
    chain.add(Step1xQualityFilter())
    chain.add(LicenseFilter(allowed=["by", "by-sa", "cc0"]))
    chain.add(ObjAversePlusPlusFilter(conditions={"is_scene": False}))
    chain.add(CustomFilter(name="my_custom_check"))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseFilter(BaseModel, ABC):
    """
    Abstract base for all retrieval filters.

    Every filter is a Pydantic model so it can be serialized to JSON and sent
    over HTTP from the client to the server.  The server deserializes the spec
    and calls ``apply(uid, metadata)`` using its in-memory metadata.

    The ``type`` discriminator field is set by each concrete subclass and is
    used for polymorphic deserialization via ``AnyFilter``.
    """

    type: str  # discriminator; each subclass fixes this with Literal[...]

    @abstractmethod
    def apply(self, uid: str, metadata: Dict[str, Any]) -> bool:
        """
        Evaluate the filter for a single object.

        Args:
            uid:      Objaverse UID.
            metadata: Unified metadata dict as returned by
                      ``Retriever.get_unified_metadata(uid)``:
                      {
                          "lvis_categories": [...],
                          "is_step1x_quality": bool,
                          "license": str | None,
                          "plusplus": {...},
                          "pali": {...},
                      }

        Returns:
            True  → keep this object.
            False → discard this object.
        """


# ---------------------------------------------------------------------------
# Built-in filters
# ---------------------------------------------------------------------------

class LvisFilter(BaseFilter):
    """
    Hard category match using LVIS annotations (Layer 1).

    Keeps only objects whose UID appears in the specified LVIS category.
    This is the strongest filter and should generally be applied first.

    Example:
        LvisFilter(category="coffee_mug")
    """

    type: Literal["lvis"] = "lvis"
    category: str = Field(..., description="LVIS category name, e.g. 'coffee_mug'")

    def apply(self, uid: str, metadata: Dict[str, Any]) -> bool:
        return self.category in metadata.get("lvis_categories", [])


class Step1xQualityFilter(BaseFilter):
    """
    Step-1X-3D high-quality set membership (Layer 2).

    Keeps only objects that are in the Step-1X-3D curated high-quality set.
    No parameters needed.

    Example:
        Step1xQualityFilter()
    """

    type: Literal["step1x_quality"] = "step1x_quality"

    def apply(self, uid: str, metadata: Dict[str, Any]) -> bool:
        return metadata.get("is_step1x_quality", False)


class LicenseFilter(BaseFilter):
    """
    License whitelist (Layer 3).

    Keeps only objects whose license string is in the allowed set.
    Objects with no license information are rejected.

    Example:
        LicenseFilter(allowed=["by", "by-sa", "cc0"])
    """

    type: Literal["license"] = "license"
    allowed: List[str] = Field(
        default_factory=lambda: ["by", "by-sa", "cc0"],
        description="Allowed license strings (lowercase short form).",
    )

    def apply(self, uid: str, metadata: Dict[str, Any]) -> bool:
        return metadata.get("license") in self.allowed


class ObjAversePlusPlusFilter(BaseFilter):
    """
    Objaverse++ quality attribute conditions (Layer 4).

    ``conditions`` is a dict of attribute → expected value pairs.
    Numeric expected values use >= comparison; all other types use ==.

    Supported attributes (from annotated_800k.json):
        is_scene          (bool)   – True if the object is a full scene
        is_multi_object   (bool)   – True if multiple objects are present
        is_single_color   (bool)   – True if the object is a single solid colour
        is_transparent    (bool)   – True if the object is transparent
        is_figure         (bool)   – True if the object is a 2-D figure/poster
        score             (int)    – Quality score; use >= comparison
        density           (str)    – Mesh density category
        style             (str)    – Visual style category

    Example:
        ObjAversePlusPlusFilter(conditions={
            "is_scene": False,
            "is_multi_object": False,
            "is_single_color": False,
            "score": 3,
        })
    """

    type: Literal["objaverse_plusplus"] = "objaverse_plusplus"
    conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Attribute → expected value pairs.",
    )

    def apply(self, uid: str, metadata: Dict[str, Any]) -> bool:
        plusplus = metadata.get("plusplus", {})
        for key, expected in self.conditions.items():
            actual = plusplus.get(key)
            if isinstance(expected, (int, float)):
                if actual is None or actual < expected:
                    return False
            else:
                if actual != expected:
                    return False
        return True


class CustomFilter(BaseFilter):
    """
    Reference to a named callable registered on the server (Layer 5).

    The filter logic itself lives on the server and is pre-registered by name
    via ``Retriever.register_custom_filter(name, fn)``.  The client sends only
    the name; the server looks it up and calls the stored function.

    This allows arbitrary Python logic without needing to serialize callables.

    Example (server side):
        retriever.register_custom_filter(
            "high_density",
            lambda uid, meta: meta.get("plusplus", {}).get("density") == "high",
        )

    Example (client side):
        CustomFilter(name="high_density")
    """

    type: Literal["custom"] = "custom"
    name: str = Field(..., description="Name of the server-side registered callable.")

    def apply(self, uid: str, metadata: Dict[str, Any]) -> bool:
        # This method is only called on the server after the callable has been
        # injected via ``inject_callable``.
        raise RuntimeError(
            f"CustomFilter '{self.name}' has no callable injected. "
            "Call inject_callable() before apply()."
        )

    def inject_callable(self, fn) -> "CustomFilter":
        """Return a bound copy of this filter with the callable attached."""
        bound = _BoundCustomFilter(name=self.name, _fn=fn)
        return bound


class _BoundCustomFilter(CustomFilter):
    """Internal: CustomFilter with an injected callable (not serialized)."""

    model_config = {"arbitrary_types_allowed": True}

    _fn: Any = None

    def model_post_init(self, __context: Any) -> None:
        pass

    def apply(self, uid: str, metadata: Dict[str, Any]) -> bool:
        return self._fn(uid, metadata)


# ---------------------------------------------------------------------------
# Union type for polymorphic deserialization
# ---------------------------------------------------------------------------

AnyFilter = Union[
    LvisFilter,
    Step1xQualityFilter,
    LicenseFilter,
    ObjAversePlusPlusFilter,
    CustomFilter,
]


# ---------------------------------------------------------------------------
# FilterChain
# ---------------------------------------------------------------------------

class FilterChain(BaseModel):
    """
    Ordered collection of filters sent from client to server.

    Filters are applied in registration order.  An object must pass every
    filter to be included in the results.

    Example:
        chain = FilterChain()
        chain.add(Step1xQualityFilter())
        chain.add(LicenseFilter(allowed=["by", "cc0"]))
        chain.add(ObjAversePlusPlusFilter(conditions={"is_scene": False}))

        # Serialize to JSON for HTTP transport
        payload = chain.model_dump()

        # Deserialize on server side
        chain = FilterChain.model_validate(payload)
    """

    filters: List[AnyFilter] = Field(default_factory=list)

    def add(self, f: AnyFilter) -> "FilterChain":
        """Append a filter to the chain. Returns self for chaining."""
        self.filters.append(f)
        return self

    def remove(self, filter_type: str) -> "FilterChain":
        """
        Remove all filters of the given type string (e.g. ``"license"``).
        Returns self for chaining.
        """
        self.filters = [f for f in self.filters if f.type != filter_type]
        return self

    def clear(self) -> "FilterChain":
        """Remove all filters. Returns self for chaining."""
        self.filters.clear()
        return self

    def apply_all(self, uid: str, metadata: Dict[str, Any]) -> bool:
        """Return True only if the uid passes every registered filter."""
        return all(f.apply(uid, metadata) for f in self.filters)

    def is_empty(self) -> bool:
        return len(self.filters) == 0
