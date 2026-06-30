# Task Editor

The Task Editor is the first step in the BeTTER authoring workflow. It is used to create and refine task definitions before asset instantiation and variation construction.

## What it is for

Use the Task Editor to:

- choose a task template from `assets/task_templates/`
- create or inspect task definitions
- edit object groups and object-level metadata
- adjust retrieval queries and other task-side fields before asset instantiation

In the intended workflow, this editor comes first:

1. **Task Editor** — define the task structure
2. **Assets Editor** — instantiate the task with concrete assets
3. **Variation Editor** — construct and refine task variations

## Prerequisites

- the BeTTER Python environment is installed
- Isaac Sim 5.0.0 is installed
- task generation runs inside the same Isaac Sim / BeTTER Python environment as the extension itself, so that environment must also include the `openai` Python package and your API configuration
- the repository root has been installed with:

```bash
pip install -e .
pip install -r requirements.txt
```

## Configure API access before launching Isaac Sim

If you use `+ Generate New Task`, set the API environment variables in the same shell where you will launch Isaac Sim.

Minimum setup for the default OpenAI endpoint:

```bash
export OPENAI_API_KEY="<your-api-key>"
```

Optional custom-compatible endpoint setup:

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="https://<your-compatible-endpoint>/v1"
```

Important behavior:

- `OPENAI_API_KEY` is required for task generation
- `OPENAI_BASE_URL` is optional
- the Task Editor `Endpoint` dropdown can override `OPENAI_BASE_URL` for the current generation request
- if the dropdown is left on `OpenAI (default)`, the generator falls back to `OPENAI_BASE_URL` from the environment when it is set
- set these variables before starting Isaac Sim so the extension sees them immediately

## Enable the extension in Isaac Sim

For the detailed Isaac Sim extension setup flow, follow the official documentation:

- `https://docs.isaacsim.omniverse.nvidia.com/5.0.0/utilities/updating_extensions.html`

For BeTTER, the relevant extension root is:

- `<repo-root>/src/isaac_sim/extensions`

Then enable:

- extension id: `task_editor`
- manifest: `src/isaac_sim/extensions/task_editor/config/extension.toml`

Once enabled, Isaac Sim opens the BeTTER Task Editor window.

## UI layout

The Task Editor uses a three-pane layout.

### Left pane: Tasks

This pane is the task navigator.

Main elements:

- `+ Generate New Task` opens the task-generation form in the right pane
- `Reload` rescans `assets/tasks/` from disk
- the task list shows discovered tasks in `template_type/task_id` form

Typical use:

- pick an existing task to edit
- or click `+ Generate New Task` to create a new task draft from a template

### Middle pane: Objects

This pane is the object and save-control area for the currently selected task.

Top-row controls:

- `Task Info` opens the task-level metadata editor in the right pane
- `Save to disk` writes the in-memory edits back to the task files on disk

Important: object edits, removals, and newly added distractors or other group members are not persisted when you click `Apply` or `Confirm` alone. They stay only in the editor's in-memory state until you click `Save to disk`.

Object groups shown in this pane:

- `Container`
- `Goal objects (target)`
- `Fail objects (distractor)`
- `Decor objects`

Within each group:

- clicking an object row selects it and opens its object editor in the right pane
- `+ Add to ...` opens the add-object form for that specific group

Typical use:

- select a group member to edit its semantic name, retrieval query, mass range, size range, and tags
- add new objects directly into the appropriate group
- return to `Task Info` when you need to edit task-level fields rather than object-level fields

### Right pane: Context editor

This pane changes mode depending on what you are doing.

The main modes are:

- `Generate New Task`
- `Task Info`
- `Object Editor`
- `Add Object`

#### Generate New Task mode

This mode is opened by `+ Generate New Task`.

Main fields:

- `Template` chooses one template from `assets/task_templates/`
- `Endpoint` chooses one registered generation endpoint
- `Task ID` sets the new task id
- `Model` sets the model name used by generation
- `Guidance` adds optional generation guidance
- `Template preview` shows the selected template structure before generation

Buttons:

- `Generate` creates a task draft and switches the editor to the generated task
- `Cancel` exits generation mode without creating a task

Generation happens in the same Python environment that launched Isaac Sim and loaded `task_editor`; there is no separate generation-only environment for the plugin. If `OPENAI_API_KEY` or the selected endpoint configuration is missing, generation fails inside this same session.

The draft generator expects `mass_range` in kilograms (`kg`) and `target_size_range` in meters (`m`), with realistic values for the described objects.

Use this mode when you want a draft authoring bundle to start from rather than hand-editing an existing task.

#### Task Info mode

This mode is opened by `Task Info` in the middle pane.

Main editable fields include task-level metadata such as:

- task id
- instruction
- goal relation
- description

Buttons:

- `Apply` updates the in-memory task state
- `Remove Task` opens a confirmation dialog before deleting the selected task directory
- `Save to disk` writes the current task files to disk

Use this mode for task-wide metadata. Do not use it for per-object retrieval or grouping edits.

#### Object Editor mode

This mode is opened when you select an object from the middle pane.

Main editable fields include:

- semantic name
- description
- retrieval query
- mass min / max
- size min / max
- tags

Buttons:

- `Apply` updates the current object in memory
- `Remove object` removes the selected object from its group

`Apply` does not write files by itself. Use `Save to disk` in the middle pane after finishing your object edits.

This is the core place to refine object semantics before asset retrieval.

#### Add Object mode

This mode is opened by the group-specific `+ Add to ...` button in the middle pane.

Main fields:

- semantic name
- retrieval query
- description
- mass min / max
- size min / max
- tags

Buttons:

- `Confirm` inserts the new object into the chosen group
- `Cancel` abandons the add-object form

`Confirm` only updates the current editor state. The new object is written to `object_universe.yaml` only after `Save to disk`.

Use this mode when the task needs an object that was not present in the original draft.

## Recommended workflow

A practical editing loop is:

1. Start Isaac Sim with the BeTTER environment active.
2. Enable `task_editor` in the Extension Manager.
3. In the left pane, either select an existing task or click `+ Generate New Task`.
4. In the right pane, choose the template and generation options if you are creating a new task.
5. In the middle pane, review the four object groups.
6. Open `Task Info` and refine task-level fields.
7. Click object rows one by one and use `Object Editor` to refine retrieval queries and metadata.
8. Use `+ Add to ...` when a group needs additional objects.
9. Click `Save to disk` from the middle pane once the task structure is ready.
10. Confirm that the task files under `assets/tasks/...` changed if you want to verify persistence.
11. Continue with the Assets Editor.

## Packing_a_Fruit_Lunch example

For `assets/tasks/loose_packing/Packing_a_Fruit_Lunch`, a typical Task Editor pass is:

1. Select `loose_packing/Packing_a_Fruit_Lunch` from the left pane.
2. Open `Task Info` and verify the instruction and goal relation.
3. In the middle pane, inspect:
   - the lunchbox in `Container`
   - fruit objects in `Goal objects (target)`
   - distractors in `Fail objects (distractor)`
4. Click each goal object and refine its retrieval query in `Object Editor`.
5. Add a new distractor with `+ Add to Fail objects (distractor)` if you want a more confusing retrieval setup.
6. Save to disk before moving to the Assets Editor.

## Inputs used by the editor

- task templates under `assets/task_templates/`
- task files under `assets/tasks/`
- object registry information under `assets/objects/registry/`

## Next step after task design

Once the task structure is ready, continue with the Assets Editor to instantiate the task with concrete assets. The asset-instantiation stage depends on the retrieval stack described in the Assets Editor and Retrieval Server documentation.

## Related documentation

- [Main README](../../../../README.md)
- [Assets Editor](../assets_editor/README.md)
- [Variation Editor](../variation_editor/README.md)
- [Retrieval Server](../../../../services/retrieval_server/README.md)
