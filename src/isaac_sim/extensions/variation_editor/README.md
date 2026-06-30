# Variation Editor

The Variation Editor is the third step in the BeTTER authoring workflow. It edits task variation files after the task structure and asset bindings are ready.

## What It Is For

Use the Variation Editor to:

- inspect every variation for a task
- create or remove derived variations
- inspect inherited and authored layout slots
- edit object pose overrides
- enable or disable target, distractor, and decor object instances in a variation
- inspect resolved goal and fail predicates
- append additional success or fail conditions
- load resolved variation layouts into the current Isaac Sim stage for inspection
- capture adjusted object poses from the stage back into YAML

In the intended workflow, this editor comes after Task Editor and Assets Editor:

1. **Task Editor** - define the task structure and object metadata
2. **Assets Editor** - instantiate task objects with concrete assets
3. **Variation Editor** - construct and refine task variations

## Prerequisites

Required inputs:

- task files under `assets/tasks/`
- object registry files under `assets/objects/registry/`
- asset bindings already prepared by the Assets Editor or by direct task migration

The BeTTER Python environment should be installed with:

```bash
pip install -e .
pip install -r requirements.txt
```

The editor runs inside Isaac Sim 5.0.0 and uses the same Python environment that launched Isaac Sim.

## Data Model

The editor discovers tasks from:

- `assets/tasks/*/*/task.yaml`

For a task directory such as:

- `assets/tasks/loose_packing/Packing_a_Fruit_Lunch/`

the relevant files are:

- `task.yaml`
- `object_universe.yaml`
- `asset_bindings.yaml`
- `base_variation.yaml`
- `variations/<variation_id>.yaml`

`BASE` is stored in `base_variation.yaml`.

Every other variation is stored as:

- `variations/<variation_id>.yaml`

Derived variations can use `extends` to inherit fields from another variation. The editor resolves this inheritance through `src.task.specs.resolve_variation`.

The editor writes directly to the selected variation file. For example:

- selecting `BASE` writes `base_variation.yaml`
- selecting `TV-01` writes `variations/TV-01.yaml`

## Important Concepts

### Authored vs Resolved

An authored variation file may contain only a small override, such as:

```yaml
variation_id: TV-01
extends: BASE
enabled: true
```

The resolved variation is the final result after inheritance from `BASE` and any parent variation is applied.

This distinction matters for layout editing:

- `Layout (resolved)` shows the final slot values after inheritance
- `Layout overrides` focuses on authored overrides for the selected variation

### Object Groups

The editor treats these semantic groups as variation-editable:

- target objects
- distractor objects
- decor objects

Container objects are shown in the resolved layout, but they are not toggled in the same way as target, distractor, or decor instances.

### Conditions

The editor exposes two condition streams:

- goal conditions, used for success
- fail conditions, used for failure

The current UI appends additional predicates through `policy_overrides.success.append` and `policy_overrides.fail.append`.

It does not expose every possible policy mutation as a dedicated UI control. Existing remove rules may still be present in YAML and are respected by the resolver.

## Enable The Extension In Isaac Sim

Launch Isaac Sim from the same Python environment where BeTTER is installed.

For the detailed Isaac Sim extension setup flow, follow the official Isaac Sim documentation:

- `https://docs.isaacsim.omniverse.nvidia.com/5.0.0/utilities/updating_extensions.html`

For BeTTER, add this extension search path:

- `<repo-root>/src/isaac_sim/extensions`

Then enable:

- extension id: `variation_editor`
- manifest: `src/isaac_sim/extensions/variation_editor/config/extension.toml`

Once enabled, Isaac Sim opens the `BeTTER Variation Editor` window. If you close the window, disable and enable the extension again to recreate it.

## UI Layout

The Variation Editor uses a three-pane layout.

### Header

The header shows:

- selected task
- task registry root
- task count
- variation count
- layout view mode
- selected variation
- dirty state

Controls:

- `Reload` rescans `assets/tasks/` from disk and reloads resolved variation caches

### Left Pane: Tasks

The left pane lists discovered tasks in:

```text
template_type/task_id
```

Selecting a task updates the variation list and layout slots.

### Middle Pane: Variations And Layouts

The top section lists variation IDs. `BASE` is displayed as the base variation.

Variation controls:

- `Create` opens a form for a new variation ID and parent variation
- `Remove` removes the selected non-BASE variation after confirmation

Layout controls:

- `Layout (resolved)` shows final inherited layout slots
- `Layout overrides` shows authored layout override slots
- `Show active only` hides inactive instances
- `Hide inactive` returns to showing both active and inactive instances
- `Show inherited` shows inherited layout slots when in override-oriented inspection
- `Hide inherited` hides inherited slots

Each layout slot row is shown as:

```text
semantic_name  |  instance_id  |  active/inactive, override/inherited
```

Selecting a layout slot updates the right-pane slot editor.

### Right Pane: Variation Details

The right pane has several modes.

#### Variation Metadata

Fields and controls:

- `Variation` shows the selected variation ID
- `Instruction` edits variation-level instruction text
- `Goal relation` shows the effective goal relation
- `Enabled` toggles whether the variation is enabled
- `Load into stage` saves current form state and loads the resolved variation into the current stage
- `Load all to stage` saves current form state and loads all candidates for active instances under `/World/AllCandidates`
- `Save variation` writes current fields and selected slot override to YAML
- `Resolve preview` refreshes the resolved variation summary

Important: `Load into stage` and `Load all to stage` call `Save variation` internally before loading. If you have unsaved form values, those values are written to disk first.

#### Variation Semantics

This section shows:

- the selected slot's semantic group
- staged goal override summary
- staged fail override summary
- resolved goal condition summary

Controls:

- `Activate` or `Deactivate` toggles the selected instance in its semantic group
- `Edit Goal Overrides` opens the success-condition override editor
- `Edit Fail Overrides` opens the fail-condition override editor

Group toggles are written as `set_goal_objects`, `set_fail_objects`, or `set_decor_objects` in the selected variation file.

#### Condition Override Editor

This mode opens after `Edit Goal Overrides` or `Edit Fail Overrides`.

It shows:

- resolved conditions for the selected condition stream
- staged append rules
- subject, relation, and target selectors

Controls:

- `Add` appends the selected predicate to the staged override list
- `Remove` removes a staged append predicate
- `Back` returns to the main variation detail pane

The saved YAML shape is:

```yaml
policy_overrides:
  success:
    append:
    - subject_id: <object_instance_id>
      relation: <relation>
      target_id: <object_instance_id>
```

For fail overrides, the same shape is written under `policy_overrides.fail.append`.

#### Slot Override Editor

This section edits the selected object slot.

Fields:

- `Position`: x, y, z
- `Rotation`: qw, qx, qy, qz
- `Scale`: x, y, z

Controls:

- `Capture from stage` reads the selected object's current world pose from the loaded stage and stages it in the form
- `Reset form` resets the staged form values to the currently resolved slot
- `Clear override` removes the authored `pose_overrides` entry for the selected instance from the selected variation file

`Save variation` writes the current slot values into:

```yaml
pose_overrides:
  <instance_id>:
    position: [...]
    rotation: [...]
    scale: [...]
```

## Recommended Layout Editing Workflow

Use this workflow when you want to adjust object placement:

1. Enable `variation_editor`.
2. Select the target task in the left pane.
3. Select a variation in the middle pane.
4. Click `Layout (resolved)` to inspect final inherited slots.
5. Select the object instance you want to move.
6. Click `Load into stage`.
7. Move the object in the Isaac Sim viewport.
8. Click `Capture from stage`.
9. Inspect the updated `Position`, `Rotation`, and `Scale` values.
10. Click `Save variation`.
11. Click `Resolve preview` to confirm the resolved variation still loads.

Use `Clear override` when you want a derived variation to inherit that slot from its parent again.

## Recommended Condition Editing Workflow

Use this workflow when a variation needs an additional success or failure predicate:

1. Select the task and variation.
2. Select the object instance that should become the condition subject.
3. Click `Edit Goal Overrides` or `Edit Fail Overrides`.
4. Choose `Subject`, `Relation`, and `Target`.
5. Click `Add`.
6. Click `Back`.
7. Click `Save variation`.
8. Click `Resolve preview` and inspect the resolved condition summary.

For example, a loose packing success condition might require:

```text
apple_1_8097  in  lunchbox_container_82cd
```

## Create A New Variation

1. Select the target task.
2. Click `Create`.
3. Enter `New ID`, such as `TV-05`.
4. Choose the parent variation in `Extends`.
5. Click `Confirm`.

This creates:

```text
assets/tasks/<template_type>/<task_id>/variations/<new_id>.yaml
```

with an initial payload similar to:

```yaml
variation_id: TV-05
extends: BASE
instruction: ""
enabled: true
```

After creation, select objects, edit slots or conditions, and click `Save variation`.

## Remove A Variation

1. Select a non-BASE variation.
2. Click `Remove`.
3. Confirm the removal dialog.

`BASE` cannot be removed from the editor.

## Packing_a_Fruit_Lunch Example

For `assets/tasks/loose_packing/Packing_a_Fruit_Lunch`:

1. Select `loose_packing/Packing_a_Fruit_Lunch`.
2. Select `BASE`, `TV-01`, or an `EV-*` variation.
3. Click `Resolve preview` and inspect active targets, distractors, decor objects, and condition counts.
4. Click `Layout (resolved)` and inspect the lunchbox, fruit, distractor, and decor slots.
5. Click `Load into stage` to view the resolved layout in Isaac Sim.
6. Move an object if needed, then select its slot and click `Capture from stage`.
7. Click `Save variation`.
8. If a variation should add a condition, use `Edit Goal Overrides` or `Edit Fail Overrides`.

## What Gets Written To Disk

The editor writes only task variation YAML files:

- `base_variation.yaml`
- `variations/<variation_id>.yaml`

It does not modify object registry USD files. Use the Assets Editor for registry asset editing.

The editor can write these fields:

- `enabled`
- `instruction`
- `set_goal_objects`
- `set_fail_objects`
- `set_decor_objects`
- `pose_overrides`
- `policy_overrides.success.append`
- `policy_overrides.fail.append`

When creating or removing variations, it also creates or removes files under:

- `variations/`

## Troubleshooting

### No tasks are listed

Check that task files exist under:

- `assets/tasks/*/*/task.yaml`

The editor uses the task spec loader, so malformed task YAML can cause a task directory to be skipped.

### A variation does not resolve

Click `Resolve preview` and inspect the status area. The same status messages are also printed to the terminal with the `[VariationEditor]` prefix. Common causes are:

- the `extends` parent does not exist
- an object instance ID in a group override does not exist
- an asset binding is missing for an active object
- a relation is not allowed by the task spec

### Load into stage appears to do nothing

Click `Load into stage` or `Load all to stage`, then inspect both the status area and the terminal output. Runtime failures print a full traceback with the `[VariationEditor]` prefix.

Common causes are:

- the selected task has candidate bindings whose registry USD files are missing
- the selected variation has active objects with no asset candidates
- the current task resolves but the object asset path no longer exists under `assets/objects/registry/`

`Load into stage` creates a new USD stage automatically if Isaac Sim does not currently have one. Before loading, it clears the previous `/World/Objects` preview root. `Load all to stage` similarly clears `/World/AllCandidates`.

### Capture from stage says no loaded layout exists

Click `Load into stage` first. `Capture from stage` reads object poses from the stage that was loaded by the editor and needs the editor's loaded episode cache.

### Load into stage changes the YAML immediately

This is expected. `Load into stage` saves the current form state before loading the resolved variation. Use `Reload` to discard unsaved in-memory edits before loading if needed.

### A slot disappears in Layout overrides mode

That slot may be inherited rather than authored in the selected variation. Click `Show inherited` or switch to `Layout (resolved)`.

### Activate or Deactivate does nothing for a container

Container instances are not toggled like target, distractor, and decor instances. Edit container pose through the slot editor, and edit container identity in the task definition if needed.

## Related Documentation

- [Main README](../../../../README.md)
- [Task Editor](../task_editor/README.md)
- [Assets Editor](../assets_editor/README.md)
