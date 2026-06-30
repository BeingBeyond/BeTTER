from __future__ import annotations


def apply_physx_collision_attrs(mesh_prim):
    from pxr import PhysxSchema

    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
    physx_collision_api.CreateRestOffsetAttr().Set(0.0)
    physx_collision_api.CreateContactOffsetAttr().Set(0.02)
