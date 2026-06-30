from __future__ import annotations

PHYSICS_MATERIALS = {
    "plastic": (0.5, 0.4, 0.3),
    "metal": (0.4, 0.3, 0.2),
    "wood": (0.6, 0.5, 0.2),
    "glass": (0.2, 0.15, 0.3),
    "rubber": (0.9, 0.8, 0.6),
    "fabric": (0.8, 0.7, 0.1),
    "ceramic": (0.3, 0.25, 0.2),
    "paper": (0.7, 0.6, 0.1),
    "leather": (0.6, 0.5, 0.2),
    "organic matter": (0.7, 0.6, 0.2),
    "organic": (0.7, 0.6, 0.2),
    "foam": (0.8, 0.7, 0.5),
    "water": (0.0, 0.0, 0.0),
    "default": (0.5, 0.4, 0.3),
}


def get_physics_material_properties(materials_list):
    if not materials_list:
        return PHYSICS_MATERIALS["default"]

    for mat in materials_list:
        mat_lower = mat.lower()
        if mat_lower in PHYSICS_MATERIALS:
            return PHYSICS_MATERIALS[mat_lower]

    return PHYSICS_MATERIALS["default"]


def bind_physics_material(stage, mesh_prim, root_prim_path: str, materials_list):
    from pxr import Sdf, UsdPhysics

    props = get_physics_material_properties(materials_list)
    material_path = f"{root_prim_path}/PhysicsMaterial"
    material = UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(material_path, "Material"))
    material.CreateStaticFrictionAttr().Set(props[0])
    material.CreateDynamicFrictionAttr().Set(props[1])
    material.CreateRestitutionAttr().Set(props[2])

    binding_rel = mesh_prim.CreateRelationship("physics:physicsMaterialBinding", custom=False)
    binding_rel.SetTargets([Sdf.Path(material_path)])
    return material_path, props
