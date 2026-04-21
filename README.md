# 🤖 BeTTER: Unmasking the Illusion of Embodied Reasoning in Vision-Language-Action Models

[![arXiv](https://img.shields.io/badge/arXiv-2604.18000-b31b1b.svg)](https://arxiv.org/abs/2604.18000)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://research.beingbeyond.com/better)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **[Unmasking the Illusion of Embodied Reasoning in Vision-Language-Action Models]** <br>
> Haiweng Xu, Sipeng Zheng, Hao Luo, Wanpeng Zhang, Ziheng Xi, Zongqing Lu <br>
> *Peking University, Tsinghua University, BeingBeyond*

This is the official repository for the paper **"Unmasking the Illusion of Embodied Reasoning in Vision-Language-Action Models"**, which introduces the **BeTTER** benchmark.

<p align="center">
  <img src="docs/static/images/teaser.png" alt="BeTTER Teaser: Unmasking the Illusion of Embodied Reasoning" width="100%">
</p>


## 📖 About
Recent Vision-Language-Action (VLA) models report impressive success rates on standard robotic benchmarks, projecting an illusion of robust semantic grounding and sequential planning. **BeTTER** is a diagnostic benchmark designed to break this illusion. By applying targeted causal interventions while enforcing *kinematic isolation*, BeTTER explicitly decouples high-level reasoning failures from low-level execution limits, unmasking severe cognitive deficits such as behavioral inertia and semantic feature collapse in state-of-the-art VLAs.

## 🚀 Release Roadmap
We are actively working to clean up and open-source the codebase. To ensure high quality, we will release the components progressively. **Watch 👀 and Star ⭐ this repository to stay updated!**

- [x] **Paper Release:** ArXiv preprint available.
- [ ] **Phase 1: Asset Curation & Task Generation Pipeline**
  - VLM-guided task instantiation templates.
  - Open-vocabulary 3D asset retrieval and integration (via Objaverse).
- [ ] **Phase 2: The BeTTER Benchmark Suite & Evaluation**
  - The complete suite of 10 base manipulation tasks and 60 diagnostic variations.
  - Standardized evaluation scripts and testing environments.
- [ ] **Phase 3: Data Augmentation & Privileged Logging**
  - Teleoperation trajectory amplification pipeline (incorporating MimicGen).
  - Deterministic privileged state logging and VQA generation scripts.

## 🛠️ Installation & Usage
*(Code and instructions are coming soon. Please stay tuned!)*

## 📝 Citation
If you find our benchmark, analysis, or data pipelines useful in your research, please consider citing our work:

```bibtex
@article{xu2026unmasking,
  title={Unmasking the Illusion of Embodied Reasoning in Vision-Language-Action Models},
  author={Xu, Haiweng and Zheng, Sipeng and Luo, Hao and Zhang, Wanpeng and Xi, Ziheng and Lu, Zongqing},
  journal={arXiv preprint arXiv:2604.18000},
  year={2026}
}
```

## 🙏 Acknowledgements
We would like to thank the open-source community, particularly the developers of [Objaverse](https://objaverse.allenai.org/) and [MimicGen](https://mimicgen.github.io/), whose foundational tools greatly facilitated the development of this benchmark.

