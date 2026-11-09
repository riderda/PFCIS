"""
PFCIS: Demonstration of Symmetric-Key Latent Noise Editing
"""

import torch
from noise_editor import NoiseEditor


def main():
    print("=== PFCIS: Prompt-Free Coverless Image Steganography ===")
    editor = NoiseEditor()

    # Simulate a standard latent noise tensor (Batch=1, Channels=4, Height=64, Width=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(100)
    latent_noise = torch.randn(1, 4, 64, 64, device=device)
    print(f"Original latent noise shape: {latent_noise.shape}, Device: {device}")

    key_seed = 42

    # 1. Noise Shuffling
    shuffled = editor.shuffle_tensor(latent_noise, seed=key_seed, ratio=1.0)
    diff_shuffle = torch.abs(latent_noise - shuffled).mean().item()
    print(f"[Strategy 1] Noise Shuffling (ratio=1.0) -> Mean L1 displacement: {diff_shuffle:.4f}")

    # 2. Noise Flipping
    flipped = editor.noise_flip(latent_noise, seed=key_seed, ratio=0.02)
    diff_flip = torch.abs(latent_noise - flipped).mean().item()
    print(f"[Strategy 2] Noise Flipping  (ratio=0.02) -> Mean L1 displacement: {diff_flip:.4f}")

    # 3. Dual-Key Shuffling
    dual_shuffled = editor.shuffle_tensor_dual_key(latent_noise, seed1=42, seed2=100, ratio1=0.3, ratio2=0.7)
    diff_dual = torch.abs(latent_noise - dual_shuffled).mean().item()
    print(f"[Extension]  Dual-Key Shuffling (0.3 / 0.7) -> Mean L1 displacement: {diff_dual:.4f}")

    print("\nNoise editing completed successfully!")


if __name__ == "__main__":
    main()
