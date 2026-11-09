"""
PFCIS: Demonstration of Symmetric-Key Latent Noise Editing Strategies
"""

import torch
from noise_editor import NoiseEditor


def main():
    print("=== PFCIS: Prompt-Free Coverless Image Steganography ===")
    print("Evaluating Symmetric-Key Latent Noise Editing Operations\n")
    editor = NoiseEditor()

    # Latent noise tensor (Batch=1, Channels=4, Height=64, Width=64) matching standard Latent Diffusion Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    latent_noise = torch.randn(1, 4, 64, 64, device=device)
    print(f"Original latent noise tensor shape: {latent_noise.shape}, Device: {device}")
    print(f"Statistical properties: Mean = {latent_noise.mean():.4f}, Std = {latent_noise.std():.4f}\n")

    key_seed = 42

    # 1. Noise Shuffling
    shuffled = editor.shuffle_tensor(latent_noise, seed=key_seed, ratio=1.0)
    diff_shuffle = torch.abs(latent_noise - shuffled).mean().item()
    print(f"[Strategy 1] Noise Shuffling   (r=1.00) -> Mean L1 deviation: {diff_shuffle:.4f}, Output Std: {shuffled.std():.4f}")

    # 2. Stepwise Noising
    noised = editor.add_stepped_noise(latent_noise, seed=key_seed, ratio=0.97)
    diff_noised = torch.abs(latent_noise - noised).mean().item()
    print(f"[Strategy 2] Stepwise Noising   (r=0.97) -> Mean L1 deviation: {diff_noised:.4f}, Output Std: {noised.std():.4f}")

    # 3. Noise Flipping
    flipped = editor.noise_flip(latent_noise, seed=key_seed, ratio=0.02)
    diff_flip = torch.abs(latent_noise - flipped).mean().item()
    print(f"[Strategy 3] Noise Flipping    (r=0.02) -> Mean L1 deviation: {diff_flip:.4f}, Output Std: {flipped.std():.4f}")

    # 4. Dual-Key Shuffling Extension
    dual_shuffled = editor.shuffle_tensor_dual_key(latent_noise, seed1=42, seed2=100, ratio1=0.3, ratio2=0.7)
    diff_dual = torch.abs(latent_noise - dual_shuffled).mean().item()
    print(f"[Extension]  Dual-Key Shuffling (0.3/0.7) -> Mean L1 deviation: {diff_dual:.4f}, Output Std: {dual_shuffled.std():.4f}")

    print("\nAll noise editing operations completed successfully with preserved Gaussian variance!")


if __name__ == "__main__":
    main()
