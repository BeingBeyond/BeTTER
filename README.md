# BeTTER: Diagnose the Illusion of Embodied Reasoning in Vision-Language-Action Models

[![arXiv](https://img.shields.io/badge/arXiv-2604.18000-b31b1b.svg)](https://arxiv.org/abs/2604.18000)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://research.beingbeyond.com/better)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **BeTTER: Diagnose the Illusion of Embodied Reasoning in Vision-Language-Action Models**  
> Haiweng Xu, Sipeng Zheng, Hao Luo, Wanpeng Zhang, Zongqing Lu  
> *ECCV 2026*  
> *Peking University, BeingBeyond*

This repository is the official code release for **BeTTER**, the ECCV 2026 paper *BeTTER: Diagnose the Illusion of Embodied Reasoning in Vision-Language-Action Models*. It provides a benchmark and tooling stack for building and editing embodied reasoning tasks in Isaac Sim.

<p align="center">
  <img src="docs/static/images/teaser.png" alt="BeTTER Teaser" width="100%">
</p>

## About

Recent Vision-Language-Action models report strong scores on standard robotic benchmarks, but those scores can hide weak semantic grounding and brittle sequential reasoning. **BeTTER** is designed to expose those failures by constructing targeted task variations while keeping low-level execution factors controlled.

## Repository contents

This initial public release includes:

- Isaac Sim editor plugins for BeTTER workflows:
  - **Task Editor**
  - **Assets Editor**
  - **Variation Editor**
- Task templates and endpoint configuration under `assets/task_templates/`
- Task templates and task definitions under `assets/`
- Robot embodiment configuration and the released Franka + Robotiq asset
- Retrieval server source files under `services/retrieval_server/`
- Supporting Python utilities and command-line entrypoints for teleoperation and rendering

## Large assets

Large assets are distributed through the companion Hugging Face dataset repository: [Seaman05/BeTTER-assets](https://huggingface.co/datasets/Seaman05/BeTTER-assets).

Currently published asset bundles include:

- [`scenes.tar`](https://huggingface.co/datasets/Seaman05/BeTTER-assets/blob/main/scenes.tar)
- [`objects_registry.tar`](https://huggingface.co/datasets/Seaman05/BeTTER-assets/blob/main/objects_registry.tar)

The published asset bundles should be unpacked with relative paths into the repository tree.

For local development, download the asset bundles from Hugging Face and either unpack them directly into the corresponding `assets/` subdirectories or symlink those directories into your local checkout.

## System requirements

Because BeTTER is built on **Isaac Sim 5.0.0**, the practical requirements are those of Isaac Sim plus a GPU capable of running RTX rendering.

### Hardware

- **GPU**: NVIDIA **RTX-series** GPU or NVIDIA **H-series** GPU is recommended
- **Display**: a workstation with a physical display is strongly recommended
- **Why headed is better**: the current editor plugins are GUI-centric and depend on Isaac Sim's interactive UI. Headless setups are possible for some workflows, but they are not the recommended path for first-time setup or day-to-day editing.

### Software

- **OS**: Linux is the primary supported environment for the current release
- **Python**: **3.11**
- **Isaac Sim**: **5.0.0**

## Installation

We recommend creating a fresh Python 3.11 environment and installing Isaac Sim before installing BeTTER itself.

### 1. Create a Python 3.11 environment

Using `conda`:

```bash
conda create -n better python=3.11
conda activate better
```

Or using `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install Isaac Sim 5.0.0

```bash
pip install isaacsim[all,extscache]==5.0.0 --extra-index-url https://pypi.nvidia.com
```

### 3. Install BeTTER

From the repository root:

```bash
pip install -e .
```

### 4. Install additional Python dependencies

Additional Python dependencies used by the released tooling are collected in the project-root `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Core workflows

### Design Your Task

Use the Task Editor to define a task template instance, inspect task files, and refine object-level metadata before asset instantiation.

- [Task Editor README](src/isaac_sim/extensions/task_editor/README.md)

### Instantiate Your Task with Assets

Use the retrieval server together with the Assets Editor to search for concrete assets, compare candidates, and update object registry content.

- [Retrieval Server README](services/retrieval_server/README.md)
- [Assets Editor README](src/isaac_sim/extensions/assets_editor/README.md)

### Construct Task Variation

Use the Variation Editor to inspect and refine variation layouts, goal predicates, and fail predicates after task structure and asset instantiation are ready.

- [Variation Editor README](src/isaac_sim/extensions/variation_editor/README.md)

### Collect Human Demonstrations

Use the task-aware teleoperation script to load a BeTTER task episode, choose a concrete variation, and record human demonstrations against the resolved task conditions.

- [Teleoperation README](tools/teleop/README.md)

### Render Your Demonstrations

Use the rendering script to replay recorded demonstration pickle files and export MP4 videos from the fixed BeTTER camera views.

- [Rendering README](tools/render/README.md)

## Recommended setup notes

- For an efficient task-design and editor-development workflow, a **headed workstation** is strongly recommended.
- If you are only using the benchmark for evaluation, a headed workstation is not required; meeting the GPU requirements for Isaac Sim is the main constraint.
- Install and verify **Isaac Sim 5.0.0** first before debugging any BeTTER-specific issue.
- Keep the BeTTER Python environment separate from unrelated research environments when possible.
- If you only need to inspect task files and editor code, `pip install -e .` is enough; install `requirements.txt` when you need the additional released dependencies.

## Development Roadmap

This release focuses on the task authoring, asset curation, variation editing,
teleoperation, and rendering workflow. Additional tasks, benchmark components,
and evaluation utilities will be added as they are validated.

## License

The BeTTER source code is released under the MIT License; see [LICENSE](LICENSE).

Some files in this repository are third-party assets or vendored third-party
software and are not covered by the BeTTER MIT License. In particular, the
included RoboLab-derived Franka + Robotiq asset is distributed under
CC BY-NC 4.0 terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
the current third-party notices and license boundaries.

## Citation

If you find BeTTER useful in your research, please cite:

```bibtex
@article{xu2026unmasking,
  title={Unmasking the Illusion of Embodied Reasoning in Vision-Language-Action Models},
  author={Xu, Haiweng and Zheng, Sipeng and Luo, Hao and Zhang, Wanpeng and Xi, Ziheng and Lu, Zongqing},
  journal={arXiv preprint arXiv:2604.18000},
  year={2026}
}
```

## Acknowledgements

We thank the open-source community, especially the developers of [Objaverse](https://objaverse.allenai.org/), [MimicGen](https://mimicgen.github.io/), and [InfiniGen](https://github.com/princeton-vl/infinigen), whose tools and assets have been useful for this project. In particular, our scene assets are generated with InfiniGen.
