# ==============================================================================
# NEMESIS-CNN: Spectrally Enhanced Momentum Attack for Robust CNNs
# Copyright (c) 2025 Qamar Muneer Akbar
# ORCID: 0009-0003-6671-9253 | qamar@ftiuae.com | www.ftiuae.com
# Academic Use Only — Non-commercial research with attribution.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import List, Tuple, Optional, Set, Dict


# ==============================================================================
# UTILITY: FFT-BASED HIGH-FREQUENCY AMPLIFICATION
# ==============================================================================

def amplify_high_freq(grad: torch.Tensor, gain: float = 1.5) -> torch.Tensor:
    """Amplify high-frequency components via FFT to exploit CNN sensitivity."""
    B, C, H, W = grad.shape
    grad_fft = torch.fft.fft2(grad, dim=(-2, -1))
    fy = (torch.arange(H, device=grad.device).float() - (H // 2)) / H
    fx = (torch.arange(W, device=grad.device).float() - (W // 2)) / W
    freq_dist = torch.sqrt(fy[:, None]**2 + fx[None, :]**2)
    weight = 1.0 + gain * freq_dist
    weight = weight[None, None, :, :].expand(B, C, H, W)
    grad_fft_amp = grad_fft * weight
    grad_amp = torch.fft.ifft2(grad_fft_amp, dim=(-2, -1)).real
    return grad_amp


# ==============================================================================
# STABLE DEFENSE TRANSFORMS (Batch-Correct & Shape-Safe)
# ==============================================================================

def identity(x): 
    return x

def jpeg_quant(x, q=0.85):
    levels = max(2, int(256 * q))
    return torch.floor(torch.clamp(x, 0, 1) * (levels - 1)) / (levels - 1)

def gaussian_blur(x, sigma=0.5):
    k = 5
    device = x.device
    coords = torch.arange(-(k//2), k//2 + 1, dtype=torch.float32, device=device)
    g = torch.exp(-coords**2 / (2 * sigma**2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    kernel = kernel[None, None, :, :].repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, groups=x.shape[1], padding=k//2)

def tv_denoise(x, weight=0.1):
    B, C, H, W = x.shape
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy[:, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]
    dxx = torch.zeros_like(x)
    dyy = torch.zeros_like(x)
    dxx[:, :, 1:, :] = dx[:, :, :-1, :] - dx[:, :, 1:, :]
    dyy[:, :, :, 1:] = dy[:, :, :, :-1] - dy[:, :, :, 1:]
    return x - weight * (dxx + dyy)

def random_resize(x, scale_range=(0.9, 1.0)):
    if random.random() < 0.7 and x.shape[-1] > 32:
        h, w = x.shape[-2:]
        scale = random.uniform(*scale_range)
        new_h, new_w = max(32, int(h * scale)), max(32, int(w * scale))
        x_resized = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        pad_h, pad_w = h - new_h, w - new_w
        if pad_h > 0 or pad_w > 0:
            top = random.randint(0, pad_h)
            left = random.randint(0, pad_w)
            x_padded = F.pad(x_resized, (left, pad_w - left, top, pad_h - top), value=0.5)
            return x_padded
    return x

def add_noise(x, std=0.01):
    return torch.clamp(x + torch.randn_like(x) * std, 0, 1)

def bit_squeeze(x, bits=4):
    levels = 2 ** bits
    return torch.floor(torch.clamp(x, 0, 1) * (levels - 1)) / (levels - 1)

def elastic_deform(x, alpha=0.3, sigma=2.0):
    """Batch-consistent elastic deformation."""
    B, C, H, W = x.shape
    device = x.device
    dx = torch.randn(B, 1, H, W, device=device)
    dy = torch.randn(B, 1, H, W, device=device)
    
    # Create 2D Gaussian kernel
    k = 5
    coords = torch.arange(-(k//2), k//2 + 1, dtype=torch.float32, device=device)
    g_1d = torch.exp(-coords**2 / (2 * sigma**2))
    g_1d = g_1d / g_1d.sum()
    g_2d = g_1d[:, None] * g_1d[None, :]  # Shape: [k, k]
    kernel = g_2d[None, None, :, :]  # Shape: [1, 1, k, k]

    dx = F.conv2d(dx.view(-1, 1, H, W), kernel, padding=k//2).view(B, 1, H, W)
    dy = F.conv2d(dy.view(-1, 1, H, W), kernel, padding=k//2).view(B, 1, H, W)
    dx *= alpha
    dy *= alpha

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
    grid = torch.cat([grid_x + dx, grid_y + dy], dim=1).permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, align_corners=True, padding_mode='border')

def color_jitter(x, strength=0.1):
    x = x * (1 + (torch.rand(1, device=x.device) - 0.5) * strength * 2)
    mean = x.mean(dim=(2,3), keepdim=True)
    x = (x - mean) * (1 + (torch.rand(1, device=x.device) - 0.5) * strength * 2) + mean
    return torch.clamp(x, 0, 1)

def rand_smooth_proxy(x, sigma=0.05):
    return add_noise(x, std=sigma)

def dropout_mask(x, p=0.05):
    mask = torch.bernoulli(torch.full_like(x, 1 - p))
    return x * mask / (1 - p + 1e-8)

DEFENSES = [
    identity,
    jpeg_quant,
    lambda x: jpeg_quant(x, q=0.7),
    lambda x: gaussian_blur(x, sigma=0.3),
    lambda x: gaussian_blur(x, sigma=1.0),
    tv_denoise,
    lambda x: tv_denoise(x, weight=0.2),
    random_resize,
    lambda x: add_noise(x, std=0.005),
    lambda x: add_noise(x, std=0.02),
    lambda x: bit_squeeze(x, bits=3),
    lambda x: bit_squeeze(x, bits=5),
    elastic_deform,
    lambda x: elastic_deform(x, alpha=0.5),
    color_jitter,
    lambda x: color_jitter(strength=0.2),
    rand_smooth_proxy,
    lambda x: rand_smooth_proxy(sigma=0.1),
    dropout_mask,
    lambda x: dropout_mask(p=0.1),
]


# ==============================================================================
# FEATURE EXTRACTOR
# ==============================================================================

class FeatureExtractor:
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.target_names: Set[str] = set(layer_names)
        self.features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.target_names:
                hook = module.register_forward_hook(
                    lambda m, inp, out, n=name: self.features.update({n: out})
                )
                self.hooks.append(hook)

    def __call__(self, x: torch.Tensor):
        self.features.clear()
        output = self.model(x)
        return output, self.features

    def remove(self):
        for h in self.hooks:
            h.remove()


# ==============================================================================
# SENSITIVITY MAP — FIXED: use .reshape()
# ==============================================================================

def compute_sensitivity_map(model, x, layer_names, device, samples=5):
    model.eval()
    B, C, H, W = x.shape
    total_grad = torch.zeros_like(x)
    extractor = FeatureExtractor(model, layer_names)
    for _ in range(samples):
        x_pert = x + torch.randn_like(x) * 0.01
        x_pert.requires_grad_(True)
        _, feats = extractor(x_pert)
        if not feats:
            extractor.remove()
            return torch.ones_like(x)
        loss = sum(f.abs().mean() for f in feats.values())
        grad = torch.autograd.grad(loss, x_pert, retain_graph=False)[0]
        total_grad += grad.abs()
        x_pert.grad = None
    extractor.remove()
    # FIXED: Use .reshape() to handle non-contiguous tensors from FFT
    norm = total_grad.reshape(B, -1).max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    mask = total_grad / norm.reshape(B, 1, 1, 1)
    return 0.5 + 0.5 * mask


# ==============================================================================
# NEMESIS-CNN – FULL IMPLEMENTATION (ONLY ONE FIX: SINGLE FORWARD PASS)
# ==============================================================================

class NEMESIS_CNN:
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 2.0 / 255.0,
        max_iterations: int = 20,
        targeted: bool = True,
        target_class: Optional[int] = None,
        device: Optional[torch.device] = None,
        input_range: Tuple[float, float] = (0.0, 1.0),
        momentum: float = 0.98,
        momentum_decay: float = 0.02,
        lambda_feat: float = 0.25,
        lambda_entropy: float = 0.15,
        lambda_kl: float = 0.4,
        temperature: float = 6.0,
        num_defenses: int = 4,
        early_stop_patience: int = 4,
        fft_gain: float = 1.3,
        smooth_grad_sigma: float = 0.5,
    ):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.epsilon = epsilon
        self.T_max = max_iterations
        self.targeted = targeted
        self.target_class = target_class
        self.device = device or next(model.parameters()).device
        self.input_range = input_range
        self.mu0 = momentum
        self.decay = momentum_decay
        self.l_f = lambda_feat
        self.l_h = lambda_entropy
        self.l_k = lambda_kl
        self.T = temperature
        self.K = num_defenses
        self.patience = early_stop_patience
        self.fft_gain = fft_gain
        self.smooth_sigma = smooth_grad_sigma

        act_layers = [
            name for name, mod in model.named_modules()
            if isinstance(mod, (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU))
        ]
        self.feat_layers = act_layers[-6:] if act_layers else []
        self.defense_impact: Dict[str, float] = {f"d{i}": 1.0 for i in range(len(DEFENSES))}
        self.defense_names = [f"d{i}" for i in range(len(DEFENSES))]

    def _ce_loss(self, logits, labels):
        return -F.cross_entropy(logits, labels, reduction='mean') if self.targeted \
               else F.cross_entropy(logits, labels, reduction='mean')

    def _entropy_loss(self, logits):
        probs = F.softmax(logits / self.T, dim=1)
        return -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean()

    def _kl_loss(self, p_logits, q_logits):
        p = F.softmax(p_logits, dim=1)
        q = F.softmax(q_logits, dim=1)
        return (p * (p.log() - q.log())).sum(dim=1).mean()

    def _sample_defenses(self) -> List[Tuple[int, callable]]:
        weights = [self.defense_impact[name] for name in self.defense_names]
        total = sum(weights)
        if total == 0:
            probs = [1.0 / len(weights)] * len(weights)
        else:
            probs = [w / total for w in weights]
        selected_indices = random.choices(range(len(DEFENSES)), weights=probs, k=self.K)
        return [(i, DEFENSES[i]) for i in selected_indices]

    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self.targeted:
            assert self.target_class is not None
            attack_labels = torch.full_like(labels, self.target_class)
        else:
            attack_labels = labels

        delta = torch.zeros_like(images)
        momentum = torch.zeros_like(images)
        best_adv = images.clone()
        best_success = torch.zeros(images.size(0), dtype=torch.bool, device=self.device)
        no_improve = 0

        sens_map = compute_sensitivity_map(
            self.model, images, self.feat_layers, self.device
        ) if self.feat_layers else torch.ones_like(images)

        if self.feat_layers:
            L = len(self.feat_layers)
            layer_weights = [(i + 1) / L for i in range(L)]
        else:
            layer_weights = []

        extractor = FeatureExtractor(self.model, self.feat_layers) if self.feat_layers else None

        for t in range(self.T_max):
            base_alpha = self.epsilon * (1.0 / self.T_max) * (1 + math.cos(math.pi * t / self.T_max))
            sens_scalar = sens_map.reshape(sens_map.size(0), -1).mean(dim=1).reshape(-1,1,1,1)
            alpha = base_alpha * sens_scalar

            mu_t = self.mu0 * math.exp(-self.decay * t)

            grad_total = torch.zeros_like(images)
            selected = self._sample_defenses()
            success_impact = torch.zeros(len(selected), device=self.device)

            for idx, (def_idx, defense) in enumerate(selected):
                x_adv = (images + delta).detach().requires_grad_(True)
                x_def = torch.clamp(defense(x_adv), *self.input_range)

                # ✅ FIXED: SINGLE FORWARD PASS to get both logits and features
                if extractor is not None:
                    logits_adv, feats = extractor(x_def)
                else:
                    logits_adv = self.model(x_def)
                    feats = {}

                loss_ce = self._ce_loss(logits_adv, attack_labels)
                loss_ent = self._entropy_loss(logits_adv)

                with torch.no_grad():
                    x_clean_def = torch.clamp(defense(images), *self.input_range)
                    logits_clean = self.model(x_clean_def)

                logits_adv_sharp = logits_adv / self.T
                logits_adv_sharp = logits_adv_sharp - logits_adv_sharp.mean(dim=1, keepdim=True)
                logits_clean_centered = logits_clean - logits_clean.mean(dim=1, keepdim=True)
                loss_kl = self._kl_loss(logits_clean_centered, logits_adv_sharp)

                # Combine main losses
                total_loss = loss_ce + self.l_h * loss_ent + self.l_k * loss_kl

                # Add feature loss if features exist
                if feats:
                    feat_vals = [f.abs().mean() for f in feats.values()]
                    loss_feat = sum(w * v for w, v in zip(layer_weights, feat_vals))
                    total_loss = total_loss + self.l_f * loss_feat

                # SINGLE backward pass — now safe
                grad = torch.autograd.grad(total_loss, x_adv, retain_graph=False)[0]
                grad_total += grad

                with torch.no_grad():
                    success_impact[idx] = total_loss.item()

            # Update defense impact
            for idx, (def_idx, _) in enumerate(selected):
                name = self.defense_names[def_idx]
                self.defense_impact[name] = 0.9 * self.defense_impact[name] + 0.1 * success_impact[idx].item()

            grad_avg = grad_total / max(1, len(selected))
            grad_avg = gaussian_blur(grad_avg, sigma=self.smooth_sigma)

            if self.fft_gain > 1.0:
                grad_avg = amplify_high_freq(grad_avg, gain=self.fft_gain)

            grad_norm = grad_avg / (grad_avg.abs().mean(dim=(1,2,3), keepdim=True) + 1e-12)

            if t == 0:
                momentum = grad_norm
            else:
                momentum = mu_t * momentum + grad_norm

            update_dir = momentum / (momentum.norm(p=2, dim=(1,2,3), keepdim=True) + 1e-12)
            update_dir = update_dir.sign()

            # Correct attack direction
            attack_sign = 1.0 if self.targeted else -1.0
            delta = delta + attack_sign * alpha * update_dir * sens_map

            # Projection
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            adv = torch.clamp(images + delta, self.input_range[0], self.input_range[1])
            delta = adv - images

            # Early success check
            with torch.no_grad():
                logits_check = self.model(adv)
                success = (logits_check.argmax(1) == self.target_class) if self.targeted \
                         else (logits_check.argmax(1) != labels)
                improved = success & (~best_success)
                if improved.any():
                    best_adv[improved] = adv[improved]
                    best_success[improved] = True
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience and t > 5:
                        break

        if extractor:
            extractor.remove()
        return best_adv.detach()