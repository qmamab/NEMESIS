"""
NEMESIS-V (image-space) - Single-file adversarial image generator for VLLMs.

Usage example:
    python nemesis_v_image_attack.py \
        --input input.jpg \
        --output adv_output.png \
        --target "a photo of a dog" \
        --method ES \
        --epsilon 0.05


"""

import argparse
import math
import warnings
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Try to import CLIP as a surrogate; optional
try:
    import clip  # type: ignore
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

# ======================
# Utilities: image load / save / preprocess
# ======================

def load_image(path: str, target_size: Optional[tuple] = None) -> torch.Tensor:
    """
    Load image as torch tensor in range [-1,1], shape [1, C, H, W].
    """
    img = Image.open(path).convert('RGB')
    if target_size is not None:
        img = img.resize(target_size, Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32)  # H, W, C
    # Normalize to [-1,1]
    arr = arr / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1, C, H, W
    return tensor


def save_image_tensor(tensor: torch.Tensor, path: str):
    """
    Save image tensor in [-1,1] (1, C, H, W) to disk as PNG.
    """
    tensor = tensor.detach().cpu().clamp(-1.0, 1.0)
    arr = ((tensor.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(path)


def tensor_to_flat_pixels(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Flatten image tensor shape (1, C, H, W) -> (flat_dim,)
    """
    return img_tensor.detach().flatten()


def flat_pixels_to_tensor(flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return flat.view(shape)


# ======================
# Optimizers (ES + RGF) - adapted to pixel space
# ======================

class EvolutionaryStrategyOptimizer:
    """
    Simple ES optimizer for pixel-space black-box optimization.
    Samples gaussian perturbations around mean and updates mean by weighted recombination.

    Reference formula (weighted recombination):
    new_mean = Σ_i w_i * x_i

    where weights w_i are proportional to exp(-rank/τ) or softmax over fitness.
    """
    def __init__(self, dimension: int, population_size: int = 50, sigma: float = 0.1, device: torch.device = torch.device('cpu')):
        self.dim = dimension
        self.pop_size = population_size
        self.sigma = sigma
        self.device = device

        self.mean = torch.zeros(self.dim, device=self.device)

    def ask(self) -> torch.Tensor:
        # Returns candidates shape [pop_size, dim]
        noise = torch.randn(self.pop_size, self.dim, device=self.device)
        samples = self.mean.unsqueeze(0) + self.sigma * noise
        return samples, noise

    def tell(self, noise: torch.Tensor, fitness: torch.Tensor):
        # fitness: higher is better
        # Use softmax weights
        weights = F.softmax(fitness - fitness.mean(), dim=0)  # normalized
        # Map weights to noise to update mean (natural gradient style)
        update = (weights.unsqueeze(1) * noise).sum(dim=0) * self.sigma
        self.mean = self.mean + update


class RandomizedGradientFreeOptimizer:
    """
    Two-point RGF (spherical smoothing) gradient estimator for black-box pixel-space.
    """
    def __init__(self, dimension: int, sigma: float = 0.1, device: torch.device = torch.device('cpu')):
        self.dim = dimension
        self.sigma = sigma
        self.device = device

    def estimate_gradient(self, objective_fn: Callable[[torch.Tensor], float], x: torch.Tensor, num_samples: int = 100):
        grad = torch.zeros_like(x, device=self.device)
        for _ in range(num_samples):
            u = torch.randn_like(x, device=self.device)
            u = u / (u.norm() + 1e-12)
            f_plus = objective_fn(x + self.sigma * u)
            f_minus = objective_fn(x - self.sigma * u)
            # two-point estimate: (f(x+σu) - f(x-σu)) / (2σ) * u
            grad += ((f_plus - f_minus) / (2 * self.sigma)) * u
        return grad / float(num_samples)


# ======================
# Objective wrappers (model or surrogate)
# ======================

class VLMImageObjective:
    """
    Wraps either a provided VLM model (if you have one locally accessible),
    or a surrogate (CLIP) for cross-modal similarity.

    The objective returns a scalar to maximize (higher means candidate is better).
    For soft-label NLL (if model gives logits) this could be negative NLL -> higher is better.
    For hard-label only models, objective returns similarity score (Jaccard on generated text)
    """

    def __init__(self, vlm_model: Optional[object], target_text: str, device: torch.device):
        self.model = vlm_model
        self.target_text = target_text
        self.device = device

        self.use_clip = False
        if self.model is None:
            if _HAS_CLIP:
                try:
                    self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)  # may be downloaded by user
                    self.use_clip = True
                    # Pre-tokenize target text
                    self.clip_text_tokens = clip.tokenize([self.target_text]).to(self.device)
                except Exception as e:
                    warnings.warn(f"CLIP load failed: {e}. Surrogate will be random.")
                    self.use_clip = False
            else:
                warnings.warn("No VLM model provided and CLIP not available — surrogate objective will be random.")
        else:
            # Model provided: check capabilities
            # Supported conveniences (if present):
            # - generate_text(image_tensor) -> List[str]
            # - get_text_logits(image_tensor, text_tokens) -> logits (soft)
            # - encode_image(image_tensor) -> image_embeddings
            # - get_text_embeddings(text_str) -> text_embeddings
            pass

    def objective_for_candidate(self, image_tensor: torch.Tensor) -> float:
        """
        Return a scalar fitness (higher is better).
        """
        # Ensure tensor on device and shape [1,C,H,W]
        x = image_tensor.to(self.device).detach()
        # 1) If model provides get_text_logits -> compute negative NLL of target sequence -> higher is better
        try:
            if self.model is not None and hasattr(self.model, 'get_text_logits'):
                logits = self.model.get_text_logits(image_tensor=x, text_tokens=None)  # model-specific API
                # User's model API may differ. Here we show a placeholder for how you'd compute NLL.
                # If logits shape is [1, seq_len, V], and you have tokenized target:
                if hasattr(self.model, 'tokenizer'):
                    tokenized = self.model.tokenizer(self.target_text, return_tensors='pt').to(self.device)
                    labels = tokenized.input_ids
                    # compute NLL (approx): sum log probs of labels (this snippet may need to be adapted)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    return float(-nll.item())  # higher is better
            # 2) If model has a generate_text API -> use hard-label similarity scoring
            if self.model is not None and hasattr(self.model, 'generate_text'):
                generated = self.model.generate_text(x)  # assumed returns list[str]
                # Compute Jaccard + exact match bonus (same as earlier)
                score = self._compute_text_similarity(generated, self.target_text)
                return float(score)
            # 3) If model provides embeddings for both modalities
            if self.model is not None and hasattr(self.model, 'encode_image') and hasattr(self.model, 'get_text_embeddings'):
                img_emb = self.model.encode_image(x)  # shape [1, D]
                txt_emb = self.model.get_text_embeddings(self.target_text)  # shape [1, D]
                cos = F.cosine_similarity(img_emb, txt_emb, dim=1)
                return float(cos.mean().item())
        except Exception as e:
            warnings.warn(f"Model-based objective evaluation failed: {e}")

        # 4) Surrogate CLIP
        if self.use_clip:
            try:
                image_input = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                image_input = (image_input + 1.0) / 2.0  # convert to [0,1] for clip
                # CLIP expects images in range [0,1]; we need to preprocess per CLIP preprocess if available
                # Clip model expects list of PIL images or preprocessed tensors; below we follow a direct approach:
                image_input = torch.nn.functional.interpolate(image_input, size=(224, 224))
                # encode image and text
                with torch.no_grad():
                    img_emb = self.clip_model.encode_image((image_input * 255).type(torch.uint8)) if False else self.clip_model.encode_image(image_input)
                    txt_emb = self.clip_model.encode_text(self.clip_text_tokens)
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                    sim = (img_emb @ txt_emb.T).item()
                    return float(sim)
            except Exception as e:
                warnings.warn(f"CLIP surrogate objective failed: {e}")

        # 5) Fallback: simple surrogate - structural similarity or color histogram matching to an approximate target?
        # For now: random small score to keep algorithm functioning
        return float(torch.rand(1).item())

    @staticmethod
    def _compute_text_similarity(generated_texts, target_text):
        target_words = set(target_text.lower().split())
        similarities = []
        for text in generated_texts:
            if not isinstance(text, str):
                text = str(text)
            generated_words = set(text.lower().split())
            intersection = len(target_words.intersection(generated_words))
            union = len(target_words.union(generated_words))
            jaccard = intersection / union if union > 0 else 0.0
            exact = 1.0 if target_text.lower() in text.lower() else 0.0
            similarities.append(jaccard + 0.5 * exact)
        return sum(similarities) / len(similarities) if similarities else 0.0


# ======================
# Attack wrapper (produces perturbed image file)
# ======================

class NemesisImageAttacker:
    """
    High-level attacker:
    - Accepts input image tensor in [-1,1]
    - Runs either ES or RGF in pixel space to maximize surrogate/model objective
    - Projects perturbation to L-inf epsilon
    """

    def __init__(self,
                 vlm_model: Optional[object],
                 target_text: str,
                 device: torch.device = torch.device('cpu'),
                 epsilon: float = 0.05,  # fraction of pixel range [0,1]
                 population_size: int = 50,
                 sigma: float = 0.08):
        self.device = device
        self.vlm_model = vlm_model
        self.target_text = target_text
        self.epsilon = float(epsilon)
        self.population_size = population_size
        self.sigma = sigma

        self.objective = VLMImageObjective(vlm_model=vlm_model, target_text=target_text, device=self.device)

    def _project_linf(self, orig: torch.Tensor, pert: torch.Tensor) -> torch.Tensor:
        """Project perturbed tensor to L-inf ball around orig with radius eps*255 (in normalized [-1,1])."""
        # orig, pert in [-1,1]
        # Map epsilon fraction on [0,1] to the [-1,1] scale: pixel range is 2, so eps_norm = eps * 2
        eps_norm = self.epsilon * 2.0
        delta = pert - orig
        delta = torch.clamp(delta, -eps_norm, eps_norm)
        return (orig + delta).clamp(-1.0, 1.0)

    def attack_blackbox_es(self, orig_img: torch.Tensor, max_generations: int = 200, queries_budget: Optional[int] = None) -> torch.Tensor:
        """
        ES black-box attack in pixel space.

        Steps:
        - Flatten image to vector x in R^d (d = C*H*W)
        - Maintain ES mean m, sample population m + sigma * N(0,I)
        - Evaluate each candidate via objective (higher better)
        - Update mean via weighted recombination (softmax over fitness)
        """
        orig = orig_img.to(self.device).detach()
        shape = orig.shape  # 1,C,H,W
        flat_dim = int(np.prod(shape))

        es = EvolutionaryStrategyOptimizer(dimension=flat_dim, population_size=self.population_size, sigma=self.sigma, device=self.device)
        es.mean = tensor_to_flat_pixels(orig).to(self.device)

        best = es.mean.clone()
        best_score = -float('inf')

        total_queries = 0
        max_gens = max_generations

        for gen in range(max_gens):
            candidates_flat, noise = es.ask()  # candidates shape [pop_size, dim]
            fitness = []
            for i in range(candidates_flat.shape[0]):
                cand_flat = candidates_flat[i]
                cand_tensor = flat_pixels_to_tensor(cand_flat, shape)
                # Project to L-inf around original
                cand_tensor = self._project_linf(orig, cand_tensor)
                score = self.objective.objective_for_candidate(cand_tensor)
                fitness.append(score)
                total_queries += 1
            fitness_tensor = torch.tensor(fitness, device=self.device, dtype=torch.float32)
            # Update ES
            es.tell(noise, fitness_tensor)

            # Track best
            cur_best_idx = int(torch.argmax(fitness_tensor).item())
            cur_best_score = float(fitness_tensor[cur_best_idx].item())
            if cur_best_score > best_score:
                best_score = cur_best_score
                best = candidates_flat[cur_best_idx].clone()

            # optional early stop
            # (user may want to base on threshold)
        pert_tensor = flat_pixels_to_tensor(best, shape)
        pert_tensor = self._project_linf(orig, pert_tensor)
        return pert_tensor

    def attack_rgf(self, orig_img: torch.Tensor, max_iters: int = 200, samples_per_step: int = 64, lr: float = 0.01) -> torch.Tensor:
        """
        RGF spherical smoothing attack in pixel space.

        Update rule per step (approx):
            x_{t+1} = x_t + η * g_est
        where g_est is estimated by averaging directional finite differences.

        We minimize negative objective -> ascend objective.
        """
        orig = orig_img.to(self.device).detach()
        shape = orig.shape
        flat = tensor_to_flat_pixels(orig).clone()
        rgf = RandomizedGradientFreeOptimizer(dimension=flat.numel(), sigma=self.sigma, device=self.device)

        x = flat.clone().to(self.device)

        def scalar_objective(flat_x: torch.Tensor):
            img = flat_pixels_to_tensor(flat_x, shape)
            img = self._project_linf(orig, img)
            return self.objective.objective_for_candidate(img)

        for it in range(max_iters):
            grad_est = rgf.estimate_gradient(lambda v: scalar_objective(v), x, num_samples=samples_per_step)
            # Ascend objective
            x = x + lr * grad_est
            # Project to feasible set around original
            x = tensor_to_flat_pixels(self._project_linf(orig, flat_pixels_to_tensor(x, shape)))
            # Optionally reduce lr
            lr *= 0.995

        final_img = flat_pixels_to_tensor(x, shape)
        final_img = self._project_linf(orig, final_img)
        return final_img

    def generate_adversarial(self, input_image: torch.Tensor, method: str = "ES", **kwargs) -> torch.Tensor:
        """
        High-level call: returns perturbed image tensor in [-1,1].
        """
        input_image = input_image.to(self.device)
        if method.upper() == "ES":
            return self.attack_blackbox_es(input_image, **kwargs)
        elif method.upper() == "RGF":
            return self.attack_rgf(input_image, **kwargs)
        else:
            raise ValueError("Unknown method. Choose 'ES' or 'RGF'.")


# ======================
# CLI for convenience
# ======================

def parse_args():
    parser = argparse.ArgumentParser(description="NEMESIS-V image-space adversarial generator")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, required=True, help="Output adversarial image path (PNG recommended)")
    parser.add_argument("--target", type=str, required=True, help="Target text you want the VLM to produce (for surrogate objective)")
    parser.add_argument("--method", type=str, default="ES", choices=["ES", "RGF"], help="Attack method")
    parser.add_argument("--epsilon", type=float, default=0.05, help="L-inf epsilon as fraction of pixel range [0,1] (e.g. 0.05)")
    parser.add_argument("--pop", type=int, default=50, help="Population size for ES")
    parser.add_argument("--gens", type=int, default=200, help="Generations / iterations")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--sigma", type=float, default=0.08, help="Sampling sigma for ES/RGF")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Load input image
    img = load_image(args.input)
    # Note: no resizing done. If your VLM expects specific size, preprocess separately.

    # No external VLM model is provided here; if you have one, instantiate and pass it to NemesisImageAttacker
    vlm_model = None  # Replace with your model object if available

    attacker = NemesisImageAttacker(vlm_model=vlm_model, target_text=args.target, device=device,
                                   epsilon=args.epsilon, population_size=args.pop, sigma=args.sigma)

    print(f"Starting attack method={args.method} epsilon={args.epsilon} pop={args.pop} gens={args.gens}")
    if args.method.upper() == "ES":
        adv = attacker.generate_adversarial(img, method="ES", max_generations=args.gens)
    else:
        adv = attacker.generate_adversarial(img, method="RGF", max_iters=args.gens, samples_per_step=64, lr=0.02)

    save_image_tensor(adv, args.output)
    print(f"Saved adversarial image to {args.output}")


if __name__ == "__main__":
    main()
