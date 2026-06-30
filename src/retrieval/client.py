"""
HTTP client for the BeTTER retrieval server.

The client maintains a local FilterChain that is serialized and attached to
every search request.  Filters can be added, removed, or cleared between calls.

The retrieval server runs in a separate Python environment (DuoduoCLIP's env)
to avoid dependency conflicts with the main simulation stack.

Example:
    from src.retrieval import RetrieverClient
    from src.retrieval.filters import (
        Step1xQualityFilter,
        LicenseFilter,
        ObjAversePlusPlusFilter,
    )

    client = RetrieverClient("http://192.168.20.173:8001")
    client.register_filter(Step1xQualityFilter())
    client.register_filter(LicenseFilter(allowed=["by", "by-sa", "cc0"]))
    client.register_filter(ObjAversePlusPlusFilter(conditions={
        "is_scene": False,
        "is_multi_object": False,
    }))

    uids  = client.search("a red coffee mug", top_k=5)
    paths = client.download(uids, download_dir="/tmp/glbs")
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import requests

from .filters import AnyFilter, FilterChain

logger = logging.getLogger(__name__)


class RetrieverClient:
    """
    Lightweight HTTP client for the retrieval server.

    Responsibilities:
    - Maintain a FilterChain that travels with every search request.
    - Provide a clean, typed API so the rest of the pipeline never touches
      raw HTTP or filter serialization.

    Args:
        server_url:       Base URL of the retrieval server, e.g.
                          ``"http://192.168.20.173:8001"``.
        search_timeout:   Seconds to wait for a search response.
        download_timeout: Seconds to wait for a download response.
    """

    def __init__(
        self,
        server_url: str,
        search_timeout: int = 60,
        download_timeout: int = 120,
    ):
        self.server_url = server_url.rstrip("/")
        self.search_timeout = search_timeout
        self.download_timeout = download_timeout
        self._chain = FilterChain()

    # ------------------------------------------------------------------
    # Filter management
    # ------------------------------------------------------------------

    def register_filter(self, f: AnyFilter) -> "RetrieverClient":
        """
        Append a filter to the chain.  Returns self for chaining.

        Example:
            client.register_filter(Step1xQualityFilter())
                  .register_filter(LicenseFilter(allowed=["cc0"]))
        """
        self._chain.add(f)
        logger.debug("Registered filter: %s", f.type)
        return self

    def unregister_filter(self, filter_type: str) -> "RetrieverClient":
        """
        Remove all filters of the given type string (e.g. ``"license"``).
        Returns self for chaining.
        """
        self._chain.remove(filter_type)
        logger.debug("Unregistered filter type: %s", filter_type)
        return self

    def clear_filters(self) -> "RetrieverClient":
        """Remove all registered filters. Returns self for chaining."""
        self._chain.clear()
        logger.debug("Cleared all filters")
        return self

    @property
    def active_filters(self) -> List[AnyFilter]:
        """Read-only view of currently registered filters."""
        return list(self._chain.filters)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        prompt: str,
        top_k: int = 10,
        offset: int = 0,
        filters: Optional[FilterChain] = None,
        exclude_uids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Retrieve Objaverse UIDs matching the text prompt.

        Uses the client's registered FilterChain unless ``filters`` is
        provided explicitly (one-off override without mutating state).

        Args:
            prompt:  Natural-language description of the desired object.
            top_k:   Maximum number of UIDs to return.
            offset:  Deterministic pagination offset.
            filters: Optional one-off FilterChain override.
            exclude_uids: Optional UID blacklist applied server-side after ranking.

        Returns:
            List of Objaverse UIDs, ordered by similarity score.

        Raises:
            RuntimeError: If the server returns a non-200 status or an error.
        """
        chain = filters if filters is not None else self._chain
        payload = {
            "prompt": prompt,
            "top_k": top_k,
            "offset": offset,
            "filters": chain.model_dump(),
            "exclude_uids": exclude_uids or [],
        }
        logger.info(
            "search: prompt=%r top_k=%d offset=%d filters=%d exclude_uids=%d",
            prompt,
            top_k,
            offset,
            len(chain.filters),
            len(exclude_uids or []),
        )

        try:
            resp = requests.post(
                f"{self.server_url}/search",
                json=payload,
                timeout=self.search_timeout,
            )
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to retrieval server at {self.server_url}. "
                "Is the server running?"
            ) from e
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Retrieval server timed out after {self.search_timeout}s."
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Retrieval server error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Retrieval server returned error: {data['error']}")

        uids = data.get("uids", [])
        logger.info("search: received %d UIDs", len(uids))
        return uids

    def search_batch(
        self,
        prompts: List[str],
        top_k: int = 10,
        offset: int = 0,
        filters: Optional[FilterChain] = None,
        exclude_uids: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """
        Retrieve UIDs for multiple prompts in a single server round-trip.

        Args:
            prompts: List of text prompts.
            top_k:   Maximum UIDs per prompt.
            offset:  Deterministic pagination offset.
            filters: Optional one-off FilterChain override.

        Returns:
            List of UID lists, one per prompt.
        """
        chain = filters if filters is not None else self._chain
        payload = {
            "prompts": prompts,
            "top_k": top_k,
            "offset": offset,
            "filters": chain.model_dump(),
            "exclude_uids": exclude_uids or [],
        }
        logger.info(
            "search_batch: prompts=%d top_k=%d offset=%d exclude_uids=%d",
            len(prompts),
            top_k,
            offset,
            len(exclude_uids or []),
        )

        try:
            resp = requests.post(
                f"{self.server_url}/search_batch",
                json=payload,
                timeout=self.search_timeout,
            )
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to retrieval server at {self.server_url}."
            ) from e
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Retrieval server timed out after {self.search_timeout}s."
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Retrieval server error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Retrieval server returned error: {data['error']}")

        return data.get("results", [])

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_with_meta(
        self,
        uids: List[str],
        download_dir: str,
        timeout_seconds: float = 120.0,
        max_workers: int = 4,
    ) -> Dict:
        """Download GLBs and return full server payload (including partial metadata)."""
        payload = {
            "uids": uids,
            "download_dir": download_dir,
            "timeout_seconds": timeout_seconds,
            "max_workers": max_workers,
        }
        logger.info(
            "download_with_meta: %d UIDs → %s timeout=%.1fs workers=%d",
            len(uids),
            download_dir,
            timeout_seconds,
            max_workers,
        )

        try:
            resp = requests.post(
                f"{self.server_url}/download",
                json=payload,
                timeout=self.download_timeout,
            )
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to retrieval server at {self.server_url}."
            ) from e
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Download server timed out after {self.download_timeout}s."
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Download server error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Download server returned error: {data['error']}")
        return data

    def download(
        self,
        uids: List[str],
        download_dir: str,
        timeout_seconds: float = 120.0,
        max_workers: int = 4,
    ) -> Dict[str, str]:
        """
        Download GLB files for the given UIDs.

        Args:
            uids:            List of Objaverse UIDs to download.
            download_dir:    Server-side directory where files are saved.
                             Must be accessible from the server process.
            timeout_seconds: Per-UID download timeout budget on server.
            max_workers:     Server-side parallel download workers.

        Returns:
            Dict mapping UID → absolute file path on the server.
            On partial download completion, only successful UID-path pairs
            are returned (timed out / failed UIDs are omitted).

        Raises:
            RuntimeError: If the server returns a non-200 status or an error.
        """
        data = self.download_with_meta(
            uids=uids,
            download_dir=download_dir,
            timeout_seconds=timeout_seconds,
            max_workers=max_workers,
        )

        paths = data.get("paths", {})
        timed_out = data.get("timed_out_uids", [])
        failed = data.get("failed_uids", [])
        partial = data.get("partial", False)

        if partial:
            logger.warning(
                "download partial: success=%d timeout=%d failed=%d",
                len(paths),
                len(timed_out),
                len(failed),
            )
        else:
            logger.info("download: received %d paths", len(paths))
        return paths
