**Neural Evasion via Meta-Ensemble, Spectral Invariance, and Semantic Subversion**
© 2025 Qamar Muneer Akbar. All Rights Reserved.

---

## Abstract

NEMESIS is a research-grade targeted adversarial attack method for evaluating neural network robustness. It combines defense-aware gradient simulation, adaptive step scheduling, momentum, and joint L∞/L2 projection to produce minimal, high-transfer perturbations. NEMESIS is provided for academic and authorized security evaluation only. Two specialized attack methods[NEMESIS-V.py, NEMESIS-CNN.py] and one general [NEMESIS.py]. The general attack method can be used for both VLLM's & CNN's, NEMESIS-V is used only for VLLMS and NEMESIS-CNN.py is used only for CNN'S. Please note that VLLM's method is still a work-in-progress as there have been issues with the token inversion. 

---

## Contents

* Abstract
* Requirements
* Quick start
* API / Parameters
* Defense proxies used
* Evaluation recommendations
* Reproducibility checklist
* Responsible use, license and disclaimer
* Citation and contact

---

## Requirements

* Python 3.8+
* PyTorch >= 1.10 (CPU or CUDA build)

Install PyTorch according to your environment ([https://pytorch.org](https://pytorch.org)) and ensure a compatible CUDA driver for GPU execution.

---

## Quick start

Example usage:

```python
import torch
import torchvision.models as models
from nemesis import NEMESIS   # `nemesis.py` contains the NEMESIS class

model = models.resnet18(pretrained=True).eval().to('cuda')
images = torch.rand(8, 3, 224, 224).to('cuda')  # inputs in [0,1]
target_class = 5

attack = NEMESIS(
    model=model,
    target_class=target_class,
    epsilon=2/255,
    max_iterations=20,
    num_restarts=2,
    attack_mode="whitebox",
    defenses_to_defeat=["jpeg","gaussian","bit_depth"]
)

adversarial_images = attack.attack(images)
```

---

## API / Main parameters

`NEMESIS(model, target_class, epsilon=2/255, max_iterations=30, num_restarts=3, input_range=(0,1), device=None, attack_mode="whitebox", defenses_to_defeat=None, query_budget=120, surrogate_models=None, momentum=0.9, dual_norm=True)`

* `model` (nn.Module): target model to attack (set to `eval()` prior to passing).
* `target_class` (int): class index to force predictions toward (targeted attack).
* `epsilon` (float): L∞ perturbation budget (default 2/255).
* `max_iterations` (int): iterations per restart.
* `num_restarts` (int): number of random restarts.
* `input_range` (tuple): expected input value range, default `(0, 1)`.
* `device` (torch.device or None): device to run on; if None, inferred from model params.
* `attack_mode` (str): `"whitebox"`, `"blackbox"` or `"transfer"`.
* `defenses_to_defeat` (list[str] or None): list of differentiable defense proxies to average gradients across (default: all supported).
* `query_budget` (int): total queries budget for black-box estimators.
* `surrogate_models` (list[nn.Module] or None): surrogate(s) used in transfer mode.
* `momentum` (float): momentum coefficient used in gradient accumulation.
* `dual_norm` (bool): whether to apply dual-norm (joint L∞ + L2) projection.

Return: `attack.attack(images)` → adversarial tensor with same shape and device as input, values clamped to `input_range`.

---

## Defense proxies included

The implementation uses differentiable or approximated transformations to model common input defenses for gradient simulation:

* `jpeg` — quantization-based proxy (rounding proxy)
* `gaussian` — differentiable Gaussian blur
* `bit_depth` — reduced bit-depth quantization (differentiable proxy)
* `random_noise` — additive Gaussian noise
* `feature_squeeze` — bit-depth-like channel reduction
* `tv_denoise` — box-filter TV denoising proxy

Notes:

* These proxies are intended to provide defense-aware gradients for whitebox evaluation. They are **not** full, exact implementations of production defenses (e.g., full JPEG DCT + quantization tables, non-differentiable median filters). Use non-differentiable defenses in evaluation via expectation-over-transformations (EOT) or query-based black-box testing where appropriate.

---

## Evaluation recommendations

1. **Datasets and models**: Use standardized datasets and pretrained checkpoints. Report model version, preprocessing pipeline, and exact class indices used for targeted attacks.
2. **Baselines**: Compare against FGSM, PGD, and established transfer attacks (e.g., MI-FGSM).
3. **Metrics**:

   * Attack success rate (ASR): fraction of samples classified as the `target_class`.
   * Query cost (black-box mode): average queries per input.
   * Perturbation norms: report L∞ and mean L2.
   * Perceptual quality: report PSNR or LPIPS where applicable.
4. **Defense evaluation**: Evaluate NEMESIS under (a) raw model; (b) models with differentiable defenses (as used in the attack); and (c) non-differentiable defenses via EOT or post-processing.
5. **Ablations**: Provide ablation studies for masking, dual-norm projection, surrogate ensembling, and number of defense transforms averaged.

---


## Responsible use, license and disclaimer

This release is intended for academic research and authorized security evaluation only. By using this code you agree to:

* Use NEMESIS only on models and data you own or have explicit permission to test.
* Comply with applicable laws and institutional security policies.
* Include a clear responsible-disclosure statement in any public release of evaluations or vulnerability reports derived from use of this code.

**License**: This software is provided “as is” for **non-commercial academic research** only. You may use, modify, and distribute the code only in non-commercial academic settings, provided this notice and the copyright header are retained.

**Disclaimer**: THE AUTHOR DISCLAIMS ALL WARRANTIES. The author is not liable for any damages or misuse arising from use of this software.

---

## Repository contents

```
README.md               # this file
nemesis.py              # attack method implementation (NEMESIS class)
NEMESIS-V.PY            # attack method specialized for VLLM's
NEMESIS-CNN.PY            # attack method specialized for CNN's 
LICENSE.txt             # full license and academic-use terms
CITATION.bib            # bibliographic entry for the method
```

---

## Citation

If you use NEMESIS in academic work, please cite:

Akbar, Q. M. (2025). NEMESIS: Neural Evasion via Meta-Ensemble, Spectral Invariance, and Semantic Subversion. ORCID: 0009-0003-6671-9253.

---

## Contact

Qamar Muneer Akbar
Email: [qamar@ftiuae.com](mailto:qamar@ftiuae.com)
GitHub: [https://github.com/qmamab](https://github.com/qmamab)

---

End of README.
