from __future__ import annotations

from dataclasses import dataclass

from .task_draft import TaskTemplateSpec


_TASK_DRAFT_SYSTEM_PROMPT = """You are authoring a BeTTER task draft from a fixed task template.
You must preserve the template skeleton and only fill in concrete task content.
Return JSON only.

Rules:
- Return exactly one top-level JSON object.
- The top-level object must contain exactly these keys: guidance, objects, base_variation_instruction, container_slot, primary_goal_relation, fail_relations, policy_hints.
- Do not return template_id, template_type, variation_set, slots, or any other extra keys.
- The objects field must be a flat JSON array of object payloads. Do not wrap objects inside another object.
- Every object payload must include slot_name, semantic_name, description, retrieval_query, mass_range, target_size_range, and group_hint.
- policy_hints must be an object with exactly two array fields: success_hints and fail_hints.
- Do not modify template_id, template_type, slot names, slot count bounds, or slot group classes.
- Think through a coherent physical situation that matches the user guidance.
- Fill every required slot with concrete objects.
- Allowed object group_hint values are target, distractor, decor, container.
- For each object, provide slot_name, semantic_name, description, retrieval_query, mass_range, target_size_range, and group_hint.
- mass_range must be expressed in kilograms (kg) as a length-2 array with min <= max and realistic values for the described object.
- target_size_range must be expressed in meters (m) as a length-2 array with min <= max and realistic values for the described object.
- Produce one base_variation_instruction for the BASE variation only.
- guidance is task-level authoring context, not the instruction shown to the policy.
- policy_hints must be an object with exactly two array fields: success_hints and fail_hints.
- container_slot must name one of the template slots assigned to the container group.
- primary_goal_relation must be one of the template allowed_relations.
- fail_relations must be drawn from the template fail_relations.
"""


@dataclass(frozen=True)
class TaskDraftPromptBundle:
    system_prompt: str
    user_prompt: str
    response_schema: dict[str, object]
    request_payload: dict[str, object]


def build_task_draft_prompt_bundle(
    *,
    template: TaskTemplateSpec,
    guidance: str,
) -> TaskDraftPromptBundle:
    guidance_text = guidance.strip()
    if not guidance_text:
        raise ValueError("Guidance cannot be empty.")

    response_schema = _build_response_schema(template)
    request_payload = {
        "template": _template_to_prompt_payload(template),
        "guidance": guidance_text,
        "response_contract": response_schema,
    }
    return TaskDraftPromptBundle(
        system_prompt=_TASK_DRAFT_SYSTEM_PROMPT,
        user_prompt=_build_user_prompt(template, guidance_text),
        response_schema=response_schema,
        request_payload=request_payload,
    )


def _build_user_prompt(template: TaskTemplateSpec, guidance: str) -> str:
    slot_lines = []
    for slot in template.slots:
        count_text = str(slot.min_count) if slot.min_count == slot.max_count else f"{slot.min_count}-{slot.max_count}"
        tag_text = ", ".join(slot.tags) if slot.tags else "none"
        description_text = slot.description or "-"
        slot_lines.append(
            f"- {slot.slot_name}: group={slot.group_hint}, count={count_text}, tags={tag_text}, description={description_text}"
        )

    relation_text = ", ".join(template.allowed_relations)
    fail_relation_text = ", ".join(template.fail_relations)
    slots_block = "\n".join(slot_lines)
    return (
        f"Template ID: {template.template_id}\n"
        f"Template type: {template.template_type}\n"
        f"Allowed relations: {relation_text}\n"
        f"Primary goal relation default: {template.primary_goal_relation}\n"
        f"Allowed fail relations: {fail_relation_text}\n"
        f"Slots:\n{slots_block}\n\n"
        f"Guidance:\n{guidance}\n\n"
        "Return exactly one JSON object matching the response contract.\n"
        "The top-level keys must be exactly: guidance, objects, base_variation_instruction, container_slot, primary_goal_relation, fail_relations, policy_hints.\n"
        "Do not return template_id, template_type, variation_set, slots, or any extra keys.\n"
        "The objects field must be a flat array, not a nested object.\n"
        "Every object must include slot_name, semantic_name, description, retrieval_query, mass_range, target_size_range, and group_hint.\n"
        "policy_hints must be an object with exactly two array fields: success_hints and fail_hints."
    )


def _build_response_schema(template: TaskTemplateSpec) -> dict[str, object]:
    slot_names = [slot.slot_name for slot in template.slots]
    allowed_relations = list(template.allowed_relations)
    fail_relations = list(dict.fromkeys(template.fail_relations))
    container_slots = [slot.slot_name for slot in template.slots if slot.group_hint == "container"]
    object_tags = sorted({tag for slot in template.slots for tag in slot.tags})

    object_properties: dict[str, object] = {
        "slot_name": {
            "type": "string",
            "enum": slot_names,
            "description": "Template slot to fill.",
        },
        "semantic_name": {
            "type": "string",
            "description": "Concrete object category such as apple or lunchbox.",
        },
        "description": {
            "type": "string",
            "description": "Short visual description for asset retrieval and review.",
        },
        "retrieval_query": {
            "type": "string",
            "description": "Asset search query for this object.",
        },
        "mass_range": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "number"},
        },
        "target_size_range": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "number"},
        },
        "group_hint": {
            "type": "string",
            "enum": ["target", "distractor", "decor", "container"],
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional object tags. Prefer template tag vocabulary when applicable.",
        },
    }
    if object_tags:
        object_properties["tags"]["preferred_values"] = object_tags

    guidance_placeholder = "<copy-user-guidance-verbatim>"
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "guidance",
            "objects",
            "base_variation_instruction",
            "container_slot",
            "primary_goal_relation",
            "fail_relations",
            "policy_hints",
        ],
        "properties": {
            "guidance": {
                "type": "string",
                "const": guidance_placeholder,
                "description": "Copy the user guidance verbatim into this field.",
            },
            "objects": {
                "type": "array",
                "description": "A flat sequence of object payloads. Do not wrap this in another object.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "slot_name",
                        "semantic_name",
                        "description",
                        "retrieval_query",
                        "mass_range",
                        "target_size_range",
                        "group_hint",
                    ],
                    "properties": object_properties,
                },
            },
            "base_variation_instruction": {
                "type": "string",
                "description": "Instruction text for the BASE variation.",
            },
            "container_slot": {
                "type": "string",
                "enum": container_slots,
            },
            "primary_goal_relation": {
                "type": "string",
                "enum": allowed_relations,
            },
            "fail_relations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": fail_relations,
                },
                "description": "Relations that should be considered failure when applied to distractors.",
            },
            "policy_hints": {
                "type": "object",
                "additionalProperties": False,
                "required": ["success_hints", "fail_hints"],
                "properties": {
                    "success_hints": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "fail_hints": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "examples": [
            {
                "guidance": guidance_placeholder,
                "objects": [],
                "base_variation_instruction": "Place the target objects into the container.",
                "container_slot": container_slots[0] if container_slots else "container",
                "primary_goal_relation": allowed_relations[0] if allowed_relations else "in",
                "fail_relations": fail_relations,
                "policy_hints": {"success_hints": [], "fail_hints": []},
            }
        ],
    }


def _template_to_prompt_payload(template: TaskTemplateSpec) -> dict[str, object]:
    return {
        "template_id": template.template_id,
        "template_type": template.template_type,
        "allowed_relations": list(template.allowed_relations),
        "primary_goal_relation": template.primary_goal_relation,
        "fail_relations": list(template.fail_relations),
        "slots": [
            {
                "slot_name": slot.slot_name,
                "group_hint": slot.group_hint,
                "description": slot.description,
                "min_count": slot.min_count,
                "max_count": slot.max_count,
                "tags": list(slot.tags),
            }
            for slot in template.slots
        ],
    }
