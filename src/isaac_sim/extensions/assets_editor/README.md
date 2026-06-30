# Assets Editor

The Assets Editor is the second step in the BeTTER authoring workflow. It binds task object instances to concrete USD assets, lets you inspect and edit candidate assets in Isaac Sim, and writes the selected asset state back into the BeTTER object registry.

## What It Is For

Use the Assets Editor to:

- inspect object instances declared by a task
- inspect each instance's current candidate assets from `asset_bindings.yaml`
- retrieve additional candidates through the retrieval server
- download, preprocess, and adopt retrieved assets into the task binding file
- preview one candidate in the current Isaac Sim stage
- compare the active candidate against other candidates for the same object instance
- open an editable USD session, bake edits into geometry, validate the session, and publish the edited USD back to the registry

In the intended workflow, this editor comes after the Task Editor and before the Variation Editor:

1. **Task Editor** - define the task structure and object metadata
2. **Assets Editor** - instantiate task objects with concrete assets
3. **Variation Editor** - construct and refine task variations

## Prerequisites

Required for basic inspection and editing of existing candidates:

- the BeTTER Python environment is installed
- Isaac Sim 5.0.0 is installed
- the repository root has been installed with:

```bash
pip install -e .
pip install -r requirements.txt
```

- the large object registry bundle has been unpacked or symlinked into `assets/objects/registry/`
- task files exist under `assets/tasks/`

Required only when you want to search for new assets:

- the retrieval server is running
- the DuoduoCLIP dataset bundle has been unpacked under `third_party/DuoduoCLIP/dataset/`
- `Four_1to6F_bs1600_LT6.ckpt` has been placed at `third_party/DuoduoCLIP/Four_1to6F_bs1600_LT6.ckpt`

The full retrieval setup is documented in:

- [Retrieval Server](../../../../services/retrieval_server/README.md)

## Data Model

The editor discovers tasks from:

- `assets/tasks/*/*/task.yaml`

For each discovered task, it reads:

- `object_universe.yaml` for the complete list of task object instances
- `asset_bindings.yaml` for object instance to asset candidate bindings
- `assets/objects/registry/` for the actual candidate USD and metadata files

The most important files are:

- `assets/tasks/<template_type>/<task_id>/object_universe.yaml`
- `assets/tasks/<template_type>/<task_id>/asset_bindings.yaml`
- `assets/objects/registry/<template_type>/<task_id>/*.usd`
- `assets/objects/registry/<template_type>/<task_id>/*.meta.json`

Object instances are discovered from `object_universe.yaml`, not only from `asset_bindings.yaml`. This means a newly added Task Editor object with no candidates yet should still appear in the Assets Editor with candidate count `0`.

During `Prepare`, object-specific physics preprocessing parameters also come from `object_universe.yaml`:

- `target_size_range` is passed to the Isaac preprocessing pipeline as `scale_range`; the pipeline scales the asset so its largest bounding-box dimension targets `(min + max) / 2`
- `mass_range` is passed to the physics pipeline as `mass_range`; the USD `MassAPI.mass` is set to `(min + max) / 2`
- if either range is missing, the editor falls back to the preprocessing defaults: size range `(0.1, 0.3)` meters and mass range `(0.1, 1.0)` kg
- the actual ranges and midpoints used during preparation are stored in the generated registry `.meta.json` under `source.task_object`, and are also included in the prepared asset metadata

The default working directory for editor sessions is:

- `outputs/assets_editor_sessions/`

Temporary downloaded retrieval assets are stored under:

- `outputs/assets_editor_sessions/staging_downloads/`

Prepared retrieval assets are stored under:

- `outputs/assets_editor_sessions/staging_prepared/`

## Enable The Extension In Isaac Sim

Launch Isaac Sim from the same Python environment where BeTTER is installed.

For the detailed Isaac Sim extension setup flow, follow the official Isaac Sim documentation:

- `https://docs.isaacsim.omniverse.nvidia.com/5.0.0/utilities/updating_extensions.html`

For BeTTER, add this extension search path:

- `<repo-root>/src/isaac_sim/extensions`

Then enable:

- extension id: `assets_editor`
- manifest: `src/isaac_sim/extensions/assets_editor/config/extension.toml`

Once enabled, Isaac Sim opens the `BeTTER Assets Editor` window. If you close the window, disable and enable the extension again to recreate it.

## UI Layout

The Assets Editor uses a three-pane layout.

### Header

The header shows:

- the selected task
- the editable session directory
- task, instance, and candidate counts
- the currently selected asset

Controls:

- `Session dir` changes where preview/edit session USD files are written
- `Reload` rescans `assets/tasks/` and reloads task bindings from disk

### Left Pane: Tasks And Instances

The left pane is the navigator.

Sections:

- `Tasks` lists discovered tasks as `template_type/task_id`
- `Instances` lists object instances for the selected task

Each instance row is shown as:

```text
semantic_name  |  instance_id  |  candidate_count
```

Selecting an instance updates the candidate list in the middle pane.

### Middle Pane: Candidates, Assets, And Staging

The `Candidates` section lists the current assets already bound to the selected object instance. These entries come from `asset_bindings.yaml`. An object instance can still be selected when this list is empty.

Selecting a candidate makes it the active asset for `Preview`, `Compare`, `Open Editable`, `Validate`, and `Publish`.

The `Assets` controls manage retrieval-backed staging:

- `Check Server` calls `GET /health` on the retrieval server URL
- `Retrieve 5` searches for five new asset UIDs using the selected instance retrieval query when available, otherwise its semantic name
- `Reset Offset` restarts retrieval pagination for the current task, instance, and prompt
- `Save Scene Selection` records currently selected stage prim paths into the staging list as a diagnostic scratchpad
- `Download` downloads retrieval staging assets into `staging_downloads`
- `Prepare` converts downloaded retrieval assets into registry-ready USD files and metadata, applying the selected object instance's size and mass ranges
- `Adopt` appends the selected prepared staging asset to the current instance binding
- `Adopt All` appends all prepared staging assets to the current instance binding
- `Clear Staging` clears the in-memory staging list
- `Remove Staging` removes the selected staging row

Important: `Save Scene Selection` does not by itself create a registry asset. It stores selected prim paths in the staging list only. To update `asset_bindings.yaml`, use the retrieval flow through `Download`, `Prepare`, and `Adopt`.

The `Staging` section shows temporary assets before they are adopted. Status labels such as `downloaded`, `prepared`, and `adopted` show how far an asset has moved through the retrieval pipeline.

### Right Pane: Selection, View, Edit, And Status

The right pane shows details for the active candidate or staging asset.

Selection fields:

- `Task`
- `Instance`
- `Semantic`
- `Source`
- `Source UID`
- `Mode`
- `Compare`
- `Validation`
- resolved asset path
- active session path

View controls:

- `Preview` opens the active asset as a preview session mounted into the current host stage
- `Compare` opens the active asset plus up to three other candidates side by side
- `Unstage View` clears the mounted preview/edit/compare content from the host stage
- `Remove Candidate` removes the selected candidate binding from `asset_bindings.yaml` after confirmation

Edit controls:

- `Open Editable` opens an editable session USD for the active asset
- `Save` saves the current session stage
- `Bake` bakes authored transforms into mesh geometry and removes authored transform ops from the session layer
- `Validate` checks whether the session is publish-ready
- `Publish` copies the validated session USD back over the selected registry USD

The `Status` area shows the result of the last operation. The same messages are also printed to the terminal with the `[AssetsEditor]` prefix.

## Basic Candidate Inspection Workflow

Use this workflow when the task already has asset candidates and you only want to inspect them:

1. Enable `assets_editor`.
2. Select a task in the `Tasks` pane.
3. Select an object instance in the `Instances` pane.
4. Select a candidate in the `Candidates` pane.
5. Click `Preview`.
6. Inspect the asset in the Isaac Sim viewport.
7. Select another candidate and click `Preview` again, or click `Compare` to view several candidates side by side.
8. Click `Unstage View` when you want to clear the viewport.

## Retrieval And Adoption Workflow

Use this workflow when an object instance needs more candidates:

1. Start the retrieval server from the repository root:

```bash
export RETRIEVAL_CONFIG=configs/retrieval/default.yaml
uvicorn services.retrieval_server.server:app --host 0.0.0.0 --port 8001
```

2. Enable `assets_editor` in Isaac Sim.
3. Select the target task and object instance.
4. Click `Check Server`; the `Server` row should change to `Alive`.
5. Click `Retrieve 5`.
6. Inspect the new staging rows.
7. Click `Download`.
8. Click `Prepare`.
9. Select one prepared staging row and click `Adopt`, or click `Adopt All`.
10. Click `Reload` if you want to force a disk rescan.
11. Confirm that new candidate rows appear for the selected instance.

`Adopt` and `Adopt All` write to:

- `assets/tasks/<template_type>/<task_id>/asset_bindings.yaml`

If the selected object instance does not have a binding entry yet, `Adopt` creates one before appending candidates.

`Prepare` writes registry USD and metadata files under:

- `assets/objects/registry/<template_type>/<task_id>/`

## Edit, Bake, Validate, And Publish Workflow

Use this workflow when a candidate asset needs geometry or transform cleanup:

1. Select the task, object instance, and candidate.
2. Click `Open Editable`.
3. Use Isaac Sim viewport tools to make the required edits.
4. Click `Save`.
5. Click `Bake`.
6. Click `Validate`.
7. Confirm that `Validation` says `publish-ready`.
8. Click `Publish`.

`Publish` overwrites the selected candidate's registry USD. It does not create a new candidate binding. If you want to preserve the original asset, duplicate it in the registry first or adopt a separate prepared staging asset.

Validation currently requires:

- at least one mesh prim
- all mesh prims are locally owned by the session stage
- no authored transform ops remain after baking
- the stage default prim matches the editor root prim

Common validation issues:

- `non_local_mesh_ownership`: the editable session still references external mesh data
- `authored_xform_ops_present`: run `Bake` before publishing
- `default_prim_mismatch`: the session default prim does not point to the editor root
- `no_mesh_prims`: the session contains no mesh geometry

## Packing_a_Fruit_Lunch Example

For `assets/tasks/loose_packing/Packing_a_Fruit_Lunch`:

1. Select `loose_packing/Packing_a_Fruit_Lunch`.
2. Select `apple_1_8097`, `banana_1_770f`, or another object instance.
3. Select one candidate from the `Candidates` pane.
4. Click `Preview` to inspect it in the viewport.
5. Click `Compare` to compare it with other candidates for the same object.
6. If the candidate is usable but needs cleanup, click `Open Editable`, edit, `Save`, `Bake`, `Validate`, and `Publish`.
7. If the candidate pool is weak, start the retrieval server and use `Retrieve 5`, `Download`, `Prepare`, and `Adopt`.

## Troubleshooting

### The extension does not show up

Make sure the Extension Manager search path is exactly:

- `<repo-root>/src/isaac_sim/extensions`

Then search for extension id `assets_editor`.

### No tasks are listed

Check that task files exist under:

- `assets/tasks/*/*/task.yaml`

Also check that each task has a valid `object_universe.yaml` and `base_variation.yaml`. A task does not need pre-existing asset candidates to appear in the Assets Editor.

### Retrieval server is offline

Click `Check Server` and inspect the status area. The default URL is:

- `http://127.0.0.1:8001`

Verify with:

```bash
curl http://127.0.0.1:8001/health
```

### Retrieve works but Download or Prepare fails

Check that the retrieval dataset and checkpoint are installed as described in the Retrieval Server README. `Prepare` also runs Isaac Sim physics preprocessing, so the command must run in the same environment that can import BeTTER and Isaac Sim.

### Compare fails while editing

If an editable session is opened directly as its own stage, compare overlay is unavailable in that direct stage. Save your edits, return to the host stage or unstage the editable session, then use `Compare`.

### Publish fails

Run `Validate` first and read the validation issue list. In the common case, run `Bake` before `Validate` so transforms are baked into geometry and authored transform ops are removed.

## Related Documentation

- [Main README](../../../../README.md)
- [Task Editor](../task_editor/README.md)
- [Variation Editor](../variation_editor/README.md)
- [Retrieval Server](../../../../services/retrieval_server/README.md)
