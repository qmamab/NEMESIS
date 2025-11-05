# ==============================================================================
# NEMESIS: Neural Evasion via Meta-Ensemble, Spectral Invariance, and Semantic Subversion (NEMESIS)
# ==============================================================================
#
# Copyright (c) 2025 Qamar Muneer Akbar. All rights reserved.
#
# Author:
#   Qamar Muneer Akbar
#   ORCID: 0009-0003-6671-9253
#   Email: qamar@ftiuae.com
#   GitHub: https://github.com/qmamab
#
# LICENSE:
#   This software is provided "as is" for **academic research purposes only**.
#   You may use, modify, and distribute this code **exclusively in non-commercial,
#   academic settings**, provided this notice is retained in full.
#
# DISCLAIMER:
#   THE AUTHOR DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT
#   LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
#   AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY CLAIM,
#   DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE,
#   ARISING FROM, OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
# WARNING:
#   This tool generates adversarial examples that may bypass machine learning defenses.
#   Use only on models and data you own or have explicit authorization to test.
#   Unauthorized use may violate laws (e.g., CFAA, GDPR, UAE Cybercrime Law).
#   The author assumes **no responsibility** for misuse by third parties.
#
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


# ======================
# DEFENSE TRANSFORMS (Differentiable or Approximated)
# ======================

def jpeg_compression(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """
    Approximate JPEG compression via uniform rounding (proxy only).
    WARNING: This is NOT a true JPEG simulation (no DCT, quantization tables, etc.).
    It mimics bit-depth reduction and is used as a differentiable proxy for evaluation.
    """
    x = torch.clamp(x, 0.0, 1.0)
    x = torch.round(x * 255.0) / 255.0
    return x

def gaussian_blur(x: torch.Tensor, kernel_size: int = 3, sigma: float = 0.8) -> torch.Tensor:
    """Differentiable Gaussian blur."""
    device = x.device
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    kernel = kernel[None, None, :, :].repeat(x.shape[1], 1, 1, 1)
    padding = kernel_size // 2
    return F.conv2d(x, kernel, groups=x.shape[1], padding=padding)

def reduce_bit_depth(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Differentiable bit-depth reduction (e.g., 4-bit → 16 levels)."""
    levels = 2 ** bits
    x = torch.clamp(x, 0.0, 1.0)
    x = torch.floor(x * (levels - 1)) / (levels - 1)
    return x

def add_random_noise(x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """Add differentiable Gaussian noise and clamp to valid range."""
    noise = torch.randn_like(x) * std
    return torch.clamp(x + noise, 0.0, 1.0)

def feature_squeeze(x: torch.Tensor, bits: int = 5) -> torch.Tensor:
    """Alias for reduce_bit_depth."""
    return reduce_bit_depth(x, bits=bits)

def total_variance_denoising(x: torch.Tensor, weight: float = 0.1) -> torch.Tensor:
    """Differentiable TV-denoising proxy using box filtering."""
    kernel = torch.ones(1, 1, 3, 3, device=x.device) / 9.0
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, groups=x.shape[1], padding=1)


# ======================
# VALID DEFENSES (ALL DIFFERENTIABLE OR SAFE PROXIES)
# ======================
# NOTE: Non-differentiable defenses (e.g., median filter) are EXCLUDED
# to ensure gradient computation remains valid during attack optimization.
DEFENSES = {
    "jpeg": jpeg_compression,
    "gaussian": gaussian_blur,
    "bit_depth": lambda x: reduce_bit_depth(x, bits=4),
    "random_noise": lambda x: add_random_noise(x, std=0.01),
    "feature_squeeze": lambda x: feature_squeeze(x, bits=5),
    "tv_denoise": total_variance_denoising,
}


# ======================
# NEMESIS ATTACK CLASS
# ======================

class NEMESIS:
    """
    Production-grade adaptive adversarial attack resilient to common input transformations.
    
    Supports:
      - Targeted whitebox attacks with defense-aware gradient simulation
      - Query-limited blackbox attacks via gradient estimation
      - Surrogate model ensembling for transfer-based blackbox attacks
      - Joint L∞ and L2 perturbation budgets
      - Adaptive step scheduling and momentum

    ⚠️ FOR ACADEMIC USE ONLY. SEE LICENSE AND DISCLAIMER ABOVE.
    """

    def __init__(
        self,
        model: nn.Module,
        target_class: int,
        epsilon: float = 2.0 / 255.0,
        max_iterations: int = 20,
        num_restarts: int = 1,
        input_range: Tuple[float, float] = (0.0, 1.0),
        device: Optional[torch.device] = None,
        attack_mode: str = "whitebox",  # "whitebox", "blackbox", "transfer"
        defenses_to_defeat: Optional[List[str]] = None,
        query_budget: int = 100,
        surrogate_models: Optional[List[nn.Module]] = None,
        momentum: float = 0.9,
        dual_norm: bool = True,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError("`model` must be a torch.nn.Module.")
        if not (0 <= target_class <= 10000):
            raise ValueError("`target_class` must be a valid class index.")
        if epsilon <= 0:
            raise ValueError("`epsilon` must be positive.")
        if max_iterations < 1 or num_restarts < 1:
            raise ValueError("Iterations and restarts must be >= 1.")
        if input_range[0] >= input_range[1]:
            raise ValueError("Invalid input_range.")
        if attack_mode not in {"whitebox", "blackbox", "transfer"}:
            raise ValueError("`attack_mode` must be 'whitebox', 'blackbox', or 'transfer'.")

        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.target_class = int(target_class)
        self.epsilon = float(epsilon)
        self.max_iterations = int(max_iterations)
        self.num_restarts = int(num_restarts)
        self.input_range = input_range
        self.momentum = float(momentum)
        self.dual_norm = bool(dual_norm)
        self.query_budget = max(1, int(query_budget))
        self.attack_mode = attack_mode

        # Resolve device
        self.device = device or next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device("cpu")
        self.model.to(self.device)

        # Validate and prepare defenses
        available_defenses = set(DEFENSES.keys())
        if defenses_to_defeat is None:
            self.defense_names = list(available_defenses)
        else:
            invalid = set(defenses_to_defeat) - available_defenses
            if invalid:
                raise ValueError(f"Unknown defenses: {invalid}. Available: {available_defenses}")
            self.defense_names = defenses_to_defeat

        self.defense_fns = [DEFENSES[name] for name in self.defense_names]

        # Handle surrogates
        self.surrogate_models = []
        if surrogate_models:
            for sm in surrogate_models:
                if not isinstance(sm, nn.Module):
                    raise TypeError("All surrogates must be nn.Module")
                sm.eval()
                for p in sm.parameters():
                    p.requires_grad_(False)
                self.surrogate_models.append(sm.to(self.device))

    def _project_perturbation(self, delta: torch.Tensor, epsilon: float, x_shape: Tuple) -> torch.Tensor:
        """Project delta to respect L∞ and (optionally) L2 bounds."""
        delta = torch.clamp(delta, -epsilon, epsilon)
        if self.dual_norm:
            n = x_shape[-1] * x_shape[-2] * x_shape[-3]
            l2_max = epsilon * math.sqrt(n)
            l2_norm = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
            exceed = l2_norm > l2_max
            if exceed.any():
                delta[exceed] = delta[exceed] * (l2_max / (l2_norm[exceed].view(-1, 1, 1, 1) + 1e-12))
        return delta

    def _estimate_blackbox_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate gradient via finite differences (query-based)."""
        batch_size = x.shape[0]
        queries_per_sample = max(1, self.query_budget // (2 * batch_size))
        grad = torch.zeros_like(x)
        eps = 1e-4

        for _ in range(queries_per_sample):
            u = torch.randn_like(x)
            u = u / (u.view(u.shape[0], -1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-12)

            x_plus = torch.clamp(x + eps * u, self.input_range[0], self.input_range[1])
            x_minus = torch.clamp(x - eps * u, self.input_range[0], self.input_range[1])

            with torch.no_grad():
                f_plus = self.model(x_plus).softmax(dim=1)[:, self.target_class]
                f_minus = self.model(x_minus).softmax(dim=1)[:, self.target_class]

            grad += (f_plus - f_minus).view(-1, 1, 1, 1) * u

        return grad / queries_per_sample

    def _compute_defense_aware_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient averaged over differentiable defense simulations."""
        x = x.clone().detach().requires_grad_(True)
        total_grad = torch.zeros_like(x)
        count = 0

        for defense_fn in self.defense_fns:
            try:
                x_def = defense_fn(x)
                x_def = torch.clamp(x_def, self.input_range[0], self.input_range[1])
                logits = self.model(x_def)
                loss = -logits[:, self.target_class].mean()
                grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
                total_grad += grad
                count += 1
            except Exception:
                # Skip if defense causes numerical issues (unlikely with current DEFENSES)
                continue

        if count == 0:
            # Fallback: no defense worked → use raw input
            logits = self.model(torch.clamp(x, self.input_range[0], self.input_range[1]))
            loss = -logits[:, self.target_class].mean()
            total_grad = torch.autograd.grad(loss, x)[0]
            count = 1

        return total_grad / count

    def _run_whitebox_attack(self, images: torch.Tensor) -> torch.Tensor:
        images = images.clone().detach().to(self.device)
        batch_size = images.shape[0]
        best_adv = images.clone()
        best_success = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for restart in range(self.num_restarts):
            if restart == 0:
                adv = images.clone()
            else:
                noise = torch.randn_like(images) * self.epsilon * 0.5
                adv = torch.clamp(images + noise, self.input_range[0], self.input_range[1])

            momentum_vec = torch.zeros_like(images)

            for t in range(self.max_iterations):
                adv = adv.detach().requires_grad_(True)

                # Use defense-aware gradients (same for whitebox/transfer in this loop)
                grad = self._compute_defense_aware_grad(adv)

                # Normalize & accumulate momentum
                grad = grad / (grad.abs().mean(dim=(1,2,3), keepdim=True) + 1e-12)
                momentum_vec = self.momentum * momentum_vec + grad

                # Cosine-annealed step size
                lr = self.epsilon * (0.5 * (1 + math.cos(math.pi * t / self.max_iterations)))

                # Update
                adv = adv - lr * momentum_vec.sign()
                delta = adv - images
                delta = self._project_perturbation(delta, self.epsilon, images.shape)
                adv = torch.clamp(images + delta, self.input_range[0], self.input_range[1])

                # Early stopping
                with torch.no_grad():
                    logits = self.model(adv)
                    preds = logits.argmax(dim=1)
                    success = (preds == self.target_class)
                    improved = success & (~best_success)
                    if improved.any():
                        best_adv[improved] = adv[improved]
                        best_success[improved] = True

                if best_success.all():
                    break

        return best_adv

    def attack(self, images: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial examples.
        Input: images in [input_range[0], input_range[1]], shape (B, C, H, W)
        Output: adversarial images in same range
        """
        if not isinstance(images, torch.Tensor):
            raise TypeError("`images` must be a torch.Tensor")
        if images.dim() != 4:
            raise ValueError("Input must be (B, C, H, W)")
        if images.device != self.device:
            images = images.to(self.device)

        images = torch.clamp(images, self.input_range[0], self.input_range[1])

        if self.attack_mode == "blackbox":
            adv = images.clone().detach()
            momentum_vec = torch.zeros_like(images)

            for t in range(self.max_iterations):
                grad = self._estimate_blackbox_grad(adv)
                grad = grad / (grad.abs().mean(dim=(1,2,3), keepdim=True) + 1e-12)
                momentum_vec = self.momentum * momentum_vec + grad

                lr = self.epsilon * (0.5 * (1 + math.cos(math.pi * t / self.max_iterations)))
                adv = adv - lr * momentum_vec.sign()

                delta = adv - images
                delta = self._project_perturbation(delta, self.epsilon, images.shape)
                adv = torch.clamp(images + delta, self.input_range[0], self.input_range[1])

            return adv

        elif self.attack_mode == "transfer" and self.surrogate_models:
            # FUTURE WORK: Replace with true ensemble gradient over all surrogates.
            # Current: uses first surrogate as proxy (common but suboptimal).
            surrogate_attacker = NEMESIS(
                model=self.surrogate_models[0],
                target_class=self.target_class,
                epsilon=self.epsilon,
                max_iterations=self.max_iterations,
                num_restarts=self.num_restarts,
                input_range=self.input_range,
                device=self.device,
                attack_mode="whitebox",
                defenses_to_defeat=self.defense_names,
                dual_norm=self.dual_norm,
                momentum=self.momentum,
            )
            return surrogate_attacker.attack(images)

        else:
            # Default: whitebox (or transfer without surrogates → treated as whitebox)
            return self._run_whitebox_attack(images)


# ==============================================================================
# END OF NEMESIS ATTACK METHOD
# If you use this method, please cite 
# "Akbar, Q. M. (2025). NEMESIS: Neural Evasion via Meta-Ensemble, Spectral Invariance, and Semantic Subversion (NEMESIS). ORCID: 0009-0003-6671-9253"
# ==============================================================================