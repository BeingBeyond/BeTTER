# Retrieval Server

The retrieval server provides the asset-retrieval backend used by the BeTTER asset workflow.

## What it is for

The server exposes retrieval endpoints that support asset search and download during task instantiation.

Current API surface:

- `POST /search`
- `POST /search_batch`
- `POST /download`
- `GET /health`

## Recommended environment

Use the current BeTTER Python environment.

From the repository root:

```bash
pip install -e .
pip install -r requirements.txt
```

The requirements file also installs the vendored `open_clip_mod` package from `third_party/DuoduoCLIP/open_clip_mod`, which is required by `DuoduoCLIP` at runtime.

This installs the retrieval server against the pip-installable `faiss-cpu==1.8.0` package. For the current open-source setup, this is the recommended path because it avoids conda solver churn in the BeTTER environment.

The retrieval stack also depends on the Lightning packages used by `DuoduoCLIP`. The open-source requirements pin these to the same versions used in the working `lohobench` environment:

- `lightning==2.5.6`
- `pytorch-lightning==2.5.6`

The retrieval stack also depends on the Lightning packages used by `DuoduoCLIP`. The open-source requirements pin these to the same versions used in the working `lohobench` environment:

- `lightning==2.5.6`
- `pytorch-lightning==2.5.6`

BeTTER vendors a lightly modified copy of `DuoduoCLIP` under `third_party/DuoduoCLIP/` so that the retrieval stack can be installed and used from the BeTTER environment directly. If you later find that your local setup is missing retrieval-specific third-party dependencies, install the additional requirement files under `third_party/DuoduoCLIP/` as needed.

If `faiss` is still missing in an already-created environment, the intended fix is:

```bash
pip install "faiss-cpu==1.8.0"
```

Avoid switching to conda `faiss` unless you explicitly want conda to solve and possibly replace numeric stack packages in the environment.

## External retrieval assets

The retrieval stack depends on two additional resources beyond the source tree:

1. **DuoduoCLIP dataset assets**
   - unpack [`duoduoclip_dataset.tar`](https://huggingface.co/datasets/Seaman05/BeTTER-assets/blob/main/duoduoclip_dataset.tar) so that it lands at `third_party/DuoduoCLIP/dataset/`
   - this dataset bundle is distributed through the BeTTER Hugging Face dataset repository
   - after unpacking, the retrieval config should be able to resolve paths such as:
     - `third_party/DuoduoCLIP/dataset/data/objaverse_embeddings/Four_1to6F_bs1600_LT6`
     - `third_party/DuoduoCLIP/dataset/objaverse_meta`

2. **DuoduoCLIP checkpoint**
   - the default BeTTER config expects `Four_1to6F_bs1600_LT6.ckpt`
   - this checkpoint is not redistributed in the BeTTER repository
   - download it from the official Hugging Face release:
     - `https://huggingface.co/3dlg-hcvc/DuoduoCLIP/blob/b31da21a7c983b1feb893745010da56493c5ab5d/Four_1to6F_bs1600_LT6.ckpt`
   - place it at `third_party/DuoduoCLIP/Four_1to6F_bs1600_LT6.ckpt`

## Start the server

After the dataset bundle and checkpoint are in place, start the server from `<repo-root>`:

```bash
export RETRIEVAL_CONFIG=configs/retrieval/default.yaml
uvicorn services.retrieval_server.server:app --host 0.0.0.0 --port 8001
```

A minimal health check is:

```bash
curl http://127.0.0.1:8001/health
```

## Configuration

The server reads its configuration from:

- `configs/retrieval/default.yaml`

You can override this with the `RETRIEVAL_CONFIG` environment variable.

The default config expects:

- `third_party/DuoduoCLIP/` for the vendored source tree
- `third_party/DuoduoCLIP/dataset/` for the retrieval dataset assets
- `third_party/DuoduoCLIP/Four_1to6F_bs1600_LT6.ckpt` for the checkpoint

## Role in the workflow

The retrieval server is primarily used together with the Assets Editor.

Workflow order:

1. **Task Editor** — define the task structure
2. **Assets Editor** — retrieve and instantiate assets
3. **Variation Editor** — construct task variations

## Related documentation

- [Main README](../../README.md)
- [Task Editor](../../src/isaac_sim/extensions/task_editor/README.md)
- [Assets Editor](../../src/isaac_sim/extensions/assets_editor/README.md)
- [Variation Editor](../../src/isaac_sim/extensions/variation_editor/README.md)
