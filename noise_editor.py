"""
PFCIS: Prompt-Free Coverless Image Steganography with Diffusion Models
Core Latent Noise Editing Algorithms
"""

import torch
import numpy as np
from typing import Optional


class NoiseEditor:
    """
    Symmetric-key governed latent noise editing module for prompt-free coverless steganography.
    Provides forward noise perturbation operations:
      1. Noise Shuffling (Spatial Permutation)
      2. Stepwise Noising (Diffusion Forward Noising Step)
      3. Noise Flipping (Sign Inversion)
      4. Dual-Key Shuffling (Hierarchical Multi-Seed Permutation)
    """

    def __init__(self):
        pass

    def shuffle_tensor(self, x: torch.Tensor, seed: int, ratio: float = 1.0) -> torch.Tensor:
        """
        Deterministic noise shuffling governed by a symmetric key seed.
        
        Args:
            x (torch.Tensor): Inverted latent noise tensor of shape (B, C, H, W).
            seed (int): Symmetric key seed derived from user secret key.
            ratio (float): Proportion of elements to shuffle in [0.0, 1.0].
            
        Returns:
            torch.Tensor: Spatially permuted noise tensor.
        """
        if ratio <= 0.0:
            return x.clone()

        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if x.is_cuda:
            torch.cuda.manual_seed_all(seed)

        x_flat = x.flatten()
        total_elem = x_flat.numel()
        n_shuffle = int(total_elem * ratio)

        if n_shuffle == 0:
            return x.clone()

        # Deterministic shuffle positions and order
        shuffle_pos = torch.randperm(total_elem, generator=gen, device=x.device)[:n_shuffle]
        shuffle_order = torch.randperm(n_shuffle, generator=gen, device=x.device)

        # Apply spatial permutation
        x_shuffled_flat = x_flat.clone()
        x_shuffled_flat[shuffle_pos] = x_shuffled_flat[shuffle_pos[shuffle_order]]

        return x_shuffled_flat.reshape(x.shape)

    def add_stepped_noise(
        self,
        x: torch.Tensor,
        seed: int,
        ratio: float = 0.97,
        alpha_cum: Optional[float] = None
    ) -> torch.Tensor:
        """
        Stepwise forward noising to inject key-governed cryptographic noise.
        
        Args:
            x (torch.Tensor): Inverted latent noise tensor of shape (B, C, H, W).
            seed (int): Symmetric key seed.
            ratio (float): Timestep ratio controlling the injected noise scale.
            alpha_cum (Optional[float]): Cumulative diffusion product. If None, derived from ratio.
            
        Returns:
            torch.Tensor: Noised latent tensor.
        """
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if x.is_cuda:
            torch.cuda.manual_seed_all(seed)

        # Default alpha_cum calculated based on standard linear diffusion schedule
        if alpha_cum is None:
            # For ratio in [0, 1], alpha_cum varies smoothly from 1.0 down to near 0.0
            # Default standard benchmark at ratio=0.97 yields alpha_cum ≈ 0.0066
            alpha_val = float(np.clip(1.0 - ratio, 1e-4, 1.0)) ** 2
        else:
            alpha_val = float(np.clip(alpha_cum, 1e-8, 1.0))

        # Generate standard normal noise governed by the key seed
        noise = torch.randn(x.shape, dtype=x.dtype, generator=gen, device=x.device)

        # Closed-form forward diffusion transformation: x_c = sqrt(alpha)*x + sqrt(1-alpha)*noise
        x_noised = np.sqrt(alpha_val) * x + np.sqrt(1.0 - alpha_val) * noise
        return x_noised

    def noise_flip(self, x: torch.Tensor, seed: int, ratio: float = 0.02) -> torch.Tensor:
        """
        Stochastic sign flipping governed by a symmetric key seed.
        
        Args:
            x (torch.Tensor): Inverted latent noise tensor of shape (B, C, H, W).
            seed (int): Symmetric key seed.
            ratio (float): Ratio of elements whose sign will be inverted.
            
        Returns:
            torch.Tensor: Sign-inverted noise tensor.
        """
        if ratio <= 0.0:
            return x.clone()

        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if x.is_cuda:
            torch.cuda.manual_seed_all(seed)

        rand_tensor = torch.rand(x.shape, dtype=torch.float32, generator=gen, device=x.device)
        flip_mask = (rand_tensor < ratio).to(dtype=x.dtype)

        # Invert sign for selected entries: x * (1 - 2 * mask)
        flipped_x = x * (1.0 - 2.0 * flip_mask)
        return flipped_x

    def shuffle_tensor_dual_key(
        self,
        x: torch.Tensor,
        seed1: int,
        seed2: int,
        ratio1: float = 0.3,
        ratio2: float = 0.7
    ) -> torch.Tensor:
        """
        Hierarchical dual-key noise shuffling for multi-party access control.
        
        Args:
            x (torch.Tensor): Inverted latent noise tensor.
            seed1 (int): Public key seed.
            seed2 (int): Private key seed.
            ratio1 (float): Proportion controlled by seed1.
            ratio2 (float): Proportion controlled by seed2.
            
        Returns:
            torch.Tensor: Dual-shuffled noise tensor.
        """
        if not np.isclose(ratio1 + ratio2, 1.0):
            raise ValueError(f"ratio1 + ratio2 must equal 1.0, got {ratio1 + ratio2}")

        x_flat = x.flatten()
        total_elem = x_flat.numel()

        gen1 = torch.Generator(device=x.device)
        gen1.manual_seed(seed1)
        gen2 = torch.Generator(device=x.device)
        gen2.manual_seed(seed2)

        # Partition positions
        perm_all = torch.randperm(total_elem, generator=gen1, device=x.device)
        n1 = int(total_elem * ratio1)
        idx_part1 = perm_all[:n1]
        idx_part2 = perm_all[n1:]

        # Shuffle each partition independently
        order1 = torch.randperm(n1, generator=gen1, device=x.device)
        order2 = torch.randperm(total_elem - n1, generator=gen2, device=x.device)

        x_dual_flat = x_flat.clone()
        x_dual_flat[idx_part1] = x_dual_flat[idx_part1[order1]]
        x_dual_flat[idx_part2] = x_dual_flat[idx_part2[order2]]

        return x_dual_flat.reshape(x.shape)
