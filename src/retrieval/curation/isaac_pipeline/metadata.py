from __future__ import annotations


def write_custom_metadata(root_prim, metadata):
    root_prim.SetCustomDataByKey("uid", metadata.get("uid", ""))
