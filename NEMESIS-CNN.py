# ==============================================================================
# NEMESIS-CNN: Spectrally Enhanced Momentum Attack for Robust CNNs
# Copyright (c) 2025 Qamar Muneer Akbar
# ORCID: 0009-0003-6671-9253 | qamar@ftiuae.com | www.ftiuae.com
# Academic Use Only — Non-commercial research with attribution.
# ==============================================================================

import math
import random
from typing import List, Tuple, Optional, Set, Dict, Callable
import sys
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from torchvision.utils import save_image

# ==============================================================================
# ADVANCED GRADIENT PROCESSING UTILITIES
# ==============================================================================

def amplify_high_freq_fft(grad: torch.Tensor, gain: float = 1.5) -> torch.Tensor:
    """
    FFT-based high-frequency amplification for better transferability.
    Inspired by spectrum analysis in C&W attacks.
    """
    B, C, H, W = grad.shape
    grad_fft = torch.fft.fft2(grad, dim=(-2, -1))
    grad_fft_shifted = torch.fft.fftshift(grad_fft, dim=(-2, -1))

    fy = torch.linspace(-0.5, 0.5, H, device=grad.device, dtype=grad.dtype)
    fx = torch.linspace(-0.5, 0.5, W, device=grad.device, dtype=grad.dtype)
    freq_dist = torch.sqrt(fy[:, None]**2 + fx[None, :]**2)
    
    weight = 1.0 + gain * freq_dist
    weight = weight[None, None, :, :].expand(B, C, H, W)
    
    grad_fft_amp = grad_fft_shifted * weight
    grad_fft_amp = torch.fft.ifftshift(grad_fft_amp, dim=(-2, -1))
    
    grad_amp = torch.fft.ifft2(grad_fft_amp, dim=(-2, -1)).real
    return grad_amp


def input_diversity_transform(x: torch.Tensor, prob: float = 0.5) -> torch.Tensor:
    """
    Input diversity for improved transferability (DI-FGSM technique).
    Randomly resize and pad the input.
    """
    if random.random() > prob:
        return x
    
    B, C, H, W = x.shape
    resize_ratio = random.uniform(0.875, 1.0)
    resized_h, resized_w = int(H * resize_ratio), int(W * resize_ratio)
    
    x_resized = F.interpolate(x, size=(resized_h, resized_w), mode='bilinear', align_corners=False)
    
    pad_h, pad_w = H - resized_h, W - resized_w
    top = random.randint(0, pad_h)
    left = random.randint(0, pad_w)
    
    x_padded = F.pad(x_resized, (left, pad_w - left, top, pad_h - top), value=0.5)
    return x_padded


def gaussian_kernel_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Gaussian blur for gradient smoothing (inspired by MI-FGSM).
    """
    device, dtype = x.device, x.dtype
    coords = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=dtype, device=device)
    gauss = torch.exp(-coords**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel[None, None, :, :].expand(x.shape[1], 1, kernel_size, kernel_size).contiguous()
    return F.conv2d(x, kernel, groups=x.shape[1], padding=kernel_size // 2)


def variance_tuning(grad: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    """
    Variance tuning to stabilize gradients across iterations.
    Helps with convergence in deep networks.
    """
    grad_mean = grad.mean(dim=(2, 3), keepdim=True)
    grad_var = grad.var(dim=(2, 3), keepdim=True)
    return (grad - grad_mean) / (torch.sqrt(grad_var) + 1e-8) * gamma + grad


# ==============================================================================
# ENHANCED DEFENSE TRANSFORMS FOR ROBUSTNESS
# ==============================================================================

def identity(x: torch.Tensor) -> torch.Tensor:
    return x

def jpeg_compression_sim(x: torch.Tensor, quality: float = 0.85) -> torch.Tensor:
    """Simulate JPEG compression effects"""
    levels = max(2, int(256 * quality))
    return torch.floor(torch.clamp(x, 0.0, 1.0) * (levels - 1)) / (levels - 1)

def gaussian_noise_defense(x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """Add Gaussian noise as a defense"""
    return torch.clamp(x + torch.randn_like(x) * std, 0.0, 1.0)

def gaussian_blur_defense(x: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """Gaussian blur defense"""
    return gaussian_kernel_blur(x, kernel_size=5, sigma=sigma)

def bit_depth_reduction(x: torch.Tensor, bits: int = 5) -> torch.Tensor:
    """Reduce bit depth"""
    levels = 2 ** bits
    return torch.floor(torch.clamp(x, 0.0, 1.0) * (levels - 1)) / (levels - 1)

def random_resizing(x: torch.Tensor, scale_range: Tuple[float, float] = (0.9, 1.0)) -> torch.Tensor:
    """Random resize and pad"""
    if random.random() < 0.5:
        return x
    
    B, C, H, W = x.shape
    scale = random.uniform(*scale_range)
    new_h, new_w = max(32, int(H * scale)), max(32, int(W * scale))
    
    x_resized = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    if new_h < H or new_w < W:
        pad_h, pad_w = H - new_h, W - new_w
        top = random.randint(0, pad_h) if pad_h > 0 else 0
        left = random.randint(0, pad_w) if pad_w > 0 else 0
        x_resized = F.pad(x_resized, (left, pad_w - left, top, pad_h - top), value=0.5)
    
    return x_resized

# Defense pool for ensemble robustness
DEFENSE_TRANSFORMS: List[Callable[[torch.Tensor], torch.Tensor]] = [
    identity,
    lambda x: jpeg_compression_sim(x, quality=0.85),
    lambda x: jpeg_compression_sim(x, quality=0.75),
    lambda x: gaussian_noise_defense(x, std=0.005),
    lambda x: gaussian_noise_defense(x, std=0.01),
    lambda x: gaussian_blur_defense(x, sigma=0.5),
    lambda x: gaussian_blur_defense(x, sigma=1.0),
    lambda x: bit_depth_reduction(x, bits=5),
    lambda x: bit_depth_reduction(x, bits=4),
    lambda x: random_resizing(x, scale_range=(0.85, 0.95)),
]


# ==============================================================================
# FEATURE EXTRACTION FOR C&W-STYLE ATTACKS
# ==============================================================================

class FeatureExtractor:
    """Extract intermediate features for feature-space attacks"""
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.target_names: Set[str] = set(layer_names)
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.target_names:
                def make_hook(n):
                    def hook(m, inp, out):
                        self.features[n] = out.clone()
                    return hook
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)

    def __call__(self, x: torch.Tensor):
        self.features.clear()
        output = self.model(x)
        return output, dict(self.features)

    def remove(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []


# ==============================================================================
# NEMESIS-CNN: UNIFIED ATTACK FRAMEWORK
# ==============================================================================

class NEMESIS_CNN:
    """
    NEMESIS-CNN: A unified adversarial attack framework combining:
    - FGSM: Fast Gradient Sign Method (basic gradient sign attack)
    - BIM: Basic Iterative Method (iterative FGSM)
    - MIM: Momentum Iterative Method (momentum-based gradients)
    - PGD: Projected Gradient Descent (constrained optimization)
    - C&W: Carlini & Wagner (feature-space optimization, confidence-based)
    - DI-FGSM: Diverse Inputs (input diversity for transferability)
    - TI-FGSM: Translation Invariant (kernel smoothing)
    - SI-FGSM: Scale Invariant (multi-scale gradients)
    
    Key innovations:
    1. Adaptive momentum with decay scheduling
    2. Multi-objective loss (logit, feature, confidence, diversity)
    3. Ensemble defense simulation for robustness
    4. FFT-based gradient amplification
    5. Variance tuning for gradient stability
    6. Input diversity transformation
    7. Cosine annealing step size schedule
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8.0 / 255.0,
        max_iterations: int = 100,
        targeted: bool = True,
        target_class: Optional[int] = None,
        device: Optional[torch.device] = None,
        input_range: Tuple[float, float] = (0.0, 1.0),
        
        # Momentum parameters (MIM)
        momentum: float = 1.0,
        momentum_decay_schedule: str = 'cosine',  # 'none', 'linear', 'cosine', 'exponential'
        
        # Step size parameters (PGD/BIM)
        alpha_multiplier: float = 2.5,
        use_cosine_annealing: bool = True,
        
        # C&W-style parameters
        use_feature_loss: bool = True,
        feature_weight: float = 0.1,
        confidence_weight: float = 0.0,  # C&W confidence margin
        
        # Transferability enhancements
        use_input_diversity: bool = True,
        diversity_prob: float = 0.7,
        use_gradient_smoothing: bool = True,
        smoothing_kernel_size: int = 5,
        use_fft_amplification: bool = True,
        fft_gain: float = 1.5,
        use_variance_tuning: bool = True,
        
        # Defense simulation (for robustness)
        num_defenses: int = 3,
        defense_sampling_strategy: str = 'random',  # 'random', 'adaptive', 'all'
        
        # Advanced features
        use_gradient_accumulation: bool = True,
        num_accumulation_steps: int = 1,
        
        **kwargs  # Accept extra arguments
    ):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        # Basic parameters
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iterations)
        self.targeted = bool(targeted)
        self.target_class = target_class
        
        try:
            self.device = device or next(model.parameters()).device
        except StopIteration:
            self.device = device or torch.device('cpu')
        
        self.input_range = (float(input_range[0]), float(input_range[1]))
        
        # Momentum (MIM)
        self.momentum_factor = float(momentum)
        self.momentum_decay_schedule = momentum_decay_schedule
        
        # Step size (PGD/BIM)
        self.alpha_multiplier = float(alpha_multiplier)
        self.use_cosine_annealing = use_cosine_annealing
        
        # C&W parameters
        self.use_feature_loss = use_feature_loss
        self.feature_weight = float(feature_weight)
        self.confidence_weight = float(confidence_weight)
        
        # Transferability
        self.use_input_diversity = use_input_diversity
        self.diversity_prob = float(diversity_prob)
        self.use_gradient_smoothing = use_gradient_smoothing
        self.smoothing_kernel_size = int(smoothing_kernel_size)
        self.use_fft_amplification = use_fft_amplification
        self.fft_gain = float(fft_gain)
        self.use_variance_tuning = use_variance_tuning
        
        # Defense simulation
        self.num_defenses = int(num_defenses)
        self.defense_sampling_strategy = defense_sampling_strategy
        
        # Gradient accumulation
        self.use_gradient_accumulation = use_gradient_accumulation
        self.num_accumulation_steps = int(num_accumulation_steps)
        
        # Feature extraction setup (C&W-style)
        if self.use_feature_loss:
            act_layers = [
                name for name, mod in model.named_modules()
                if isinstance(mod, (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.Tanh))
            ]
            # Select diverse layers from different depths
            if len(act_layers) >= 6:
                self.feat_layers = [act_layers[i] for i in [len(act_layers)//4, len(act_layers)//2, 3*len(act_layers)//4, -1]]
            else:
                self.feat_layers = act_layers[-3:] if len(act_layers) >= 3 else act_layers
        else:
            self.feat_layers = []
        
        print(f"\n⚙️  NEMESIS-CNN Configuration:")
        print(f"   Epsilon: {self.epsilon:.5f} ({self.epsilon*255:.1f}/255)")
        print(f"   Iterations: {self.max_iter}")
        print(f"   Momentum: {self.momentum_factor} ({self.momentum_decay_schedule} decay)")
        print(f"   Step size: {self.alpha_multiplier}x scaled")
        print(f"   Features: {'Enabled' if self.use_feature_loss else 'Disabled'} ({len(self.feat_layers)} layers)")
        print(f"   Input diversity: {'Enabled' if self.use_input_diversity else 'Disabled'}")
        print(f"   Gradient smoothing: {'Enabled' if self.use_gradient_smoothing else 'Disabled'}")
        print(f"   FFT amplification: {'Enabled' if self.use_fft_amplification else 'Disabled'}")
        print(f"   Defense simulation: {self.num_defenses} transforms")

    def _compute_momentum_decay(self, iteration: int) -> float:
        """Compute momentum decay factor based on schedule"""
        progress = iteration / self.max_iter
        
        if self.momentum_decay_schedule == 'none':
            return self.momentum_factor
        elif self.momentum_decay_schedule == 'linear':
            return self.momentum_factor * (1.0 - 0.5 * progress)
        elif self.momentum_decay_schedule == 'cosine':
            return self.momentum_factor * (0.5 + 0.5 * math.cos(math.pi * progress))
        elif self.momentum_decay_schedule == 'exponential':
            return self.momentum_factor * math.exp(-2.0 * progress)
        else:
            return self.momentum_factor

    def _compute_step_size(self, iteration: int) -> float:
        """Compute adaptive step size with annealing"""
        base_alpha = self.alpha_multiplier * self.epsilon / self.max_iter
        
        if self.use_cosine_annealing:
            progress = iteration / self.max_iter
            alpha = base_alpha * (1.0 + math.cos(math.pi * progress)) / 2.0
            alpha = max(alpha, base_alpha * 0.1)  # Minimum step size
            return alpha
        else:
            return base_alpha

    def _compute_multiobjective_loss(
        self, 
        logits: torch.Tensor, 
        target_labels: torch.Tensor,
        features: Dict[str, torch.Tensor],
        clean_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Multi-objective loss combining:
        1. Cross-entropy (classification objective)
        2. Feature loss (C&W-style feature matching)
        3. Confidence margin (C&W confidence)
        """
        # Primary classification loss
        ce_loss = F.cross_entropy(logits, target_labels, reduction='mean')
        
        if not self.targeted:
            ce_loss = -ce_loss
        
        total_loss = ce_loss
        
        # Feature diversity loss (C&W-style)
        if self.use_feature_loss and features:
            feat_loss = sum(f.abs().mean() for f in features.values()) / len(features)
            total_loss = total_loss + self.feature_weight * feat_loss
        
        # Confidence margin (C&W)
        if self.confidence_weight > 0:
            if self.targeted:
                # Maximize target class confidence
                target_logits = logits[:, target_labels[0]]
                other_logits = torch.cat([logits[:, :target_labels[0]], logits[:, target_labels[0]+1:]], dim=1)
                max_other = other_logits.max(dim=1)[0]
                confidence_loss = torch.clamp(max_other - target_logits + self.confidence_weight, min=0.0).mean()
            else:
                # Minimize true class confidence
                true_logits = logits[range(len(target_labels)), target_labels]
                other_logits_list = []
                for i, label in enumerate(target_labels):
                    other = torch.cat([logits[i, :label], logits[i, label+1:]])
                    other_logits_list.append(other)
                max_other = torch.stack([o.max() for o in other_logits_list])
                confidence_loss = torch.clamp(true_logits - max_other + self.confidence_weight, min=0.0).mean()
            
            total_loss = total_loss + confidence_loss
        
        return total_loss

    def _sample_defenses(self) -> List[Callable[[torch.Tensor], torch.Tensor]]:
        """Sample defense transforms based on strategy"""
        if self.defense_sampling_strategy == 'all':
            return DEFENSE_TRANSFORMS
        elif self.defense_sampling_strategy == 'random':
            k = min(self.num_defenses, len(DEFENSE_TRANSFORMS))
            return random.sample(DEFENSE_TRANSFORMS, k=k)
        else:  # adaptive or default to random
            k = min(self.num_defenses, len(DEFENSE_TRANSFORMS))
            return random.sample(DEFENSE_TRANSFORMS, k=k)

    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Execute the unified adversarial attack.
        
        Args:
            images: Clean input images [B, C, H, W]
            labels: True labels for untargeted attack [B]
        
        Returns:
            Adversarial images [B, C, H, W]
        """
        # CRITICAL: Clone and detach to avoid modifying original
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # Set attack labels
        if self.targeted:
            assert self.target_class is not None, "target_class required for targeted attack"
            attack_labels = torch.full((images.shape[0],), self.target_class, 
                                      dtype=labels.dtype, device=self.device)
        else:
            attack_labels = labels
        
        # Initialize perturbation and momentum
        delta = torch.zeros_like(images, requires_grad=False)
        momentum = torch.zeros_like(images)
        
        # Feature extractor
        extractor = FeatureExtractor(self.model, self.feat_layers) if self.feat_layers else None
        
        # Get clean features for reference (C&W)
        clean_features = None
        if extractor and self.use_feature_loss:
            with torch.no_grad():
                _, clean_features = extractor(images)
        
        # Best result tracking
        best_adv = images.clone()
        best_loss = float('inf') if self.targeted else float('-inf')
        
        print(f"\n🚀 Starting unified attack...")
        print(f"   Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
        print(f"   Epsilon constraint: {self.epsilon:.6f}")
        
        for iteration in range(self.max_iter):
            # Compute adaptive parameters
            current_momentum = self._compute_momentum_decay(iteration)
            current_alpha = self._compute_step_size(iteration)
            
            # Gradient accumulation buffer
            grad_accum = torch.zeros_like(images)
            
            # Sample defense transforms
            defense_transforms = self._sample_defenses()
            
            # Accumulate gradients across defenses and steps
            for accum_step in range(self.num_accumulation_steps):
                for defense_fn in defense_transforms:
                    # Create adversarial candidate
                    x_adv = images + delta
                    x_adv = torch.clamp(x_adv, self.input_range[0], self.input_range[1])
                    
                    # Apply input diversity (DI-FGSM)
                    if self.use_input_diversity:
                        x_adv = input_diversity_transform(x_adv, prob=self.diversity_prob)
                    
                    # Apply defense transform
                    x_def = defense_fn(x_adv)
                    x_def = torch.clamp(x_def, self.input_range[0], self.input_range[1])
                    x_def.requires_grad_(True)
                    
                    # Forward pass with feature extraction
                    if extractor:
                        logits, features = extractor(x_def)
                    else:
                        logits = self.model(x_def)
                        features = {}
                    
                    if isinstance(logits, tuple):  # Handle inception_v3
                        logits = logits[0]
                    
                    # Compute multi-objective loss
                    loss = self._compute_multiobjective_loss(logits, attack_labels, features, clean_features)
                    
                    # Backward pass
                    loss.backward()
                    grad = x_def.grad.detach()
                    
                    # Accumulate gradient
                    grad_accum += grad
            
            # Average accumulated gradients
            grad_avg = grad_accum / (len(defense_transforms) * self.num_accumulation_steps)
            
            # Apply gradient processing techniques
            
            # 1. Variance tuning (stabilization)
            if self.use_variance_tuning:
                grad_avg = variance_tuning(grad_avg, gamma=0.5)
            
            # 2. Translation-invariant (TI-FGSM): Gaussian smoothing
            if self.use_gradient_smoothing:
                grad_avg = gaussian_kernel_blur(grad_avg, kernel_size=self.smoothing_kernel_size, sigma=1.0)
            
            # 3. FFT-based high-frequency amplification
            if self.use_fft_amplification:
                grad_avg = amplify_high_freq_fft(grad_avg, gain=self.fft_gain)
            
            # 4. Normalize gradient
            grad_norm = grad_avg / (grad_avg.abs().mean(dim=(1,2,3), keepdim=True) + 1e-8)
            
            # 5. Update momentum (MIM)
            momentum = current_momentum * momentum + grad_norm
            
            # 6. Apply update (sign-based for robustness)
            if self.targeted:
                delta = delta - current_alpha * momentum.sign()
            else:
                delta = delta + current_alpha * momentum.sign()
            
            # 7. Project to epsilon ball (PGD constraint)
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            
            # 8. Ensure valid range
            delta = torch.clamp(images + delta, self.input_range[0], self.input_range[1]) - images
            
            # Track best result
            with torch.no_grad():
                current_adv = images + delta
                
                logits_check = self.model(current_adv)
                if isinstance(logits_check, tuple):
                    logits_check = logits_check[0]
                
                loss_check = F.cross_entropy(logits_check, attack_labels)
                if not self.targeted:
                    loss_check = -loss_check
                
                if (self.targeted and loss_check < best_loss) or (not self.targeted and loss_check > best_loss):
                    best_loss = loss_check.item()
                    best_adv = current_adv.clone()
            
            # Logging
            if (iteration + 1) % 10 == 0 or iteration == 0:
                with torch.no_grad():
                    logits_eval = self.model(images + delta)
                    if isinstance(logits_eval, tuple):
                        logits_eval = logits_eval[0]
                    probs = F.softmax(logits_eval, dim=1)
                    pred_class = logits_eval.argmax(dim=1).item()
                    
                    if self.targeted:
                        target_prob = probs[0, self.target_class].item()
                        success = pred_class == self.target_class
                        print(f"  Iter {iteration+1:3d}: Target prob={target_prob:.4f}, "
                              f"Pred={pred_class}, Loss={loss.item():.4f}, "
                              f"{'✓ SUCCESS' if success else 'attacking...'}")
                        
                        if target_prob > 0.99:
                            print(f"  🎯 Very high confidence! Early stopping.")
                            break
                    else:
                        true_prob = probs[0, labels[0]].item()
                        success = pred_class != labels[0].item()
                        print(f"  Iter {iteration+1:3d}: True prob={true_prob:.4f}, "
                              f"Pred={pred_class}, Loss={loss.item():.4f}, "
                              f"{'✓ SUCCESS' if success else 'attacking...'}")
                        
                        if true_prob < 0.01:
                            print(f"  🎯 Very low true class probability! Early stopping.")
                            break
        
        # Cleanup
        if extractor:
            extractor.remove()
        
        return best_adv.detach()


# ==============================================================================
# MAIN DEMO
# ==============================================================================

def get_class_prob(class_id, classes, probs):
    try:
        return probs[classes.index(class_id)]
    except ValueError:
        return 0.0

def main():
    print("=" * 80)
    print(" " * 20 + "NEMESIS-CNN: Unified Attack Framework")
    print("=" * 80)
    
    # Configuration
    model_name = 'vgg16'
    epsilon_255 = 8.0
    max_iter = 100
    
    # Check for interactive environment
    is_interactive = 'ipykernel' in sys.modules or os.environ.get('COLAB_GPU')

    if is_interactive:
        model_name_input = input("\nEnter model name (e.g., vgg16): ").strip().lower()
        if model_name_input:
            model_name = model_name_input

        epsilon_input = input("Enter attack budget (epsilon in [0,255], e.g., 8): ").strip()
        if epsilon_input:
            try:
                epsilon_255 = float(epsilon_input)
            except ValueError:
                pass

        iter_input = input("Enter max iterations (default=100): ").strip()
        if iter_input:
            try:
                max_iter = int(iter_input)
            except ValueError:
                pass
    
    epsilon = epsilon_255 / 255.0
    
    # Model zoo
    model_zoo = {
        'vgg16': lambda: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
        'densenet121': lambda: models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1),
        'resnet50': lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
        'inception_v3': lambda: models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=True),
        'efficientnet_b0': lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    }

    if model_name not in model_zoo:
        print(f"Invalid model. Using vgg16.")
        model_name = 'vgg16'

    # Setup
    print(f"\nLoading pretrained {model_name} from PyTorch...")
    model = model_zoo[model_name]()
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    CAT_CLASS = 281
    ARMORED_CAR_CLASS = 407

    # Load image
    image_path = 'cat.jpg'
    if not os.path.exists(image_path):
        print("Downloading sample cat image...")
        image_url = "https://live.staticflickr.com/5081/5283401569_514571165a_z.jpg"
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(image_url, image_path)
        except Exception as e:
            print(f"Error downloading: {e}")
            sys.exit(1)

    # Preprocess
    if model_name == 'inception_v3':
        preprocess = models.Inception_V3_Weights.IMAGENET1K_V1.transforms()
    else:
        preprocess = models.VGG16_Weights.IMAGENET1K_V1.transforms()
    
    image = Image.open(image_path).convert('RGB')
    x = preprocess(image).unsqueeze(0).to(device)
    true_label = torch.tensor([CAT_CLASS], device=device)

    # Original predictions
    def get_top_predictions(img_tensor, k=5):
        with torch.no_grad():
            logits = model(img_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            topk_prob, topk_id = torch.topk(probs, k)
        return topk_id[0].cpu().tolist(), topk_prob[0].cpu().tolist()

    print("\n" + "=" * 80)
    print("ATTACK CONFIGURATION")
    print("=" * 80)
    print(f"   Model: {model_name}")
    print(f"   Source class: {CAT_CLASS} (cat)")
    print(f"   Target class: {ARMORED_CAR_CLASS} (armored car)")
    print(f"   Attack type: Targeted")

    print("\n🔍 Original image predictions:")
    orig_classes, orig_probs = get_top_predictions(x)
    for i, (c, p) in enumerate(zip(orig_classes, orig_probs)):
        marker = " ← SOURCE" if c == CAT_CLASS else ""
        print(f"  {i+1}. Class {c:4d}: {p:.5f}{marker}")

    # Run NEMESIS-CNN attack
    attacker = NEMESIS_CNN(
        model=model,
        epsilon=epsilon,
        max_iterations=max_iter,
        targeted=True,
        target_class=ARMORED_CAR_CLASS,
        device=device,
        
        # Momentum configuration (MIM)
        momentum=1.0,
        momentum_decay_schedule='cosine',
        
        # Step size (PGD/BIM)
        alpha_multiplier=2.5,
        use_cosine_annealing=True,
        
        # C&W-style features
        use_feature_loss=True,
        feature_weight=0.1,
        confidence_weight=0.0,
        
        # Transferability enhancements
        use_input_diversity=True,
        diversity_prob=0.7,
        use_gradient_smoothing=True,
        smoothing_kernel_size=5,
        use_fft_amplification=True,
        fft_gain=1.5,
        use_variance_tuning=True,
        
        # Defense simulation
        num_defenses=3,
        defense_sampling_strategy='random',
        
        # Gradient accumulation
        use_gradient_accumulation=True,
        num_accumulation_steps=1,
    )

    adv_x = attacker.attack(x, true_label)

    # Results
    print("\n" + "=" * 80)
    print("ADVERSARIAL IMAGE PREDICTIONS")
    print("=" * 80)
    adv_classes, adv_probs = get_top_predictions(adv_x)
    for i, (c, p) in enumerate(zip(adv_classes, adv_probs)):
        marker = " ← TARGET! ✓" if c == ARMORED_CAR_CLASS else ""
        print(f"  {i+1}. Class {c:4d}: {p:.5f}{marker}")

    success = adv_classes[0] == ARMORED_CAR_CLASS
    target_prob_final = get_class_prob(ARMORED_CAR_CLASS, adv_classes, adv_probs)
    
    print("\n" + "=" * 80)
    print("ATTACK RESULTS")
    print("=" * 80)
    if success and target_prob_final >= 0.9:
        print(f"  🎉 EXCELLENT SUCCESS! Target probability: {target_prob_final:.2%}")
    elif success:
        print(f"  ✅ Attack succeeded! Target probability: {target_prob_final:.2%}")
    else:
        print(f"  ⚠️  Attack did not achieve top-1. Target probability: {target_prob_final:.2%}")
    
    print(f"\n  Source class ({CAT_CLASS}) probability:")
    print(f"    Before: {get_class_prob(CAT_CLASS, orig_classes, orig_probs):.6f}")
    print(f"    After:  {get_class_prob(CAT_CLASS, adv_classes, adv_probs):.6f}")
    print(f"    Change: {get_class_prob(CAT_CLASS, adv_classes, adv_probs) - get_class_prob(CAT_CLASS, orig_classes, orig_probs):+.6f}")
    
    print(f"\n  Target class ({ARMORED_CAR_CLASS}) probability:")
    print(f"    Before: {get_class_prob(ARMORED_CAR_CLASS, orig_classes, orig_probs):.6f}")
    print(f"    After:  {get_class_prob(ARMORED_CAR_CLASS, adv_classes, adv_probs):.6f}")
    print(f"    Change: {get_class_prob(ARMORED_CAR_CLASS, adv_classes, adv_probs) - get_class_prob(ARMORED_CAR_CLASS, orig_classes, orig_probs):+.6f}")

    # Calculate perturbation statistics
    pert = (adv_x - x).abs()
    l_inf = pert.max().item()
    l_2 = pert.pow(2).sum().sqrt().item()
    l_0 = (pert > 0).float().sum().item()
    
    print(f"\n  Perturbation statistics:")
    print(f"    L∞ norm: {l_inf:.6f} ({l_inf * 255:.2f}/255)")
    print(f"    L2 norm: {l_2:.4f}")
    print(f"    L0 norm: {l_0:.0f} pixels modified")
    print(f"    PSNR: {20 * math.log10(1.0 / (pert.mean().item() + 1e-10)):.2f} dB")

    # Save results
    save_image(x, "nemesis_original.png")
    save_image(adv_x, "nemesis_adversarial.png")
    save_image((adv_x - x).abs() * 10, "nemesis_perturbation.png")  # Amplified for visibility
    
    print("\n💾 Files saved:")
    print("   - nemesis_original.png (original image)")
    print("   - nemesis_adversarial.png (adversarial image)")
    print("   - nemesis_perturbation.png (perturbation visualization, 10x amplified)")
    print("=" * 80)

if __name__ == '__main__':
    main()
