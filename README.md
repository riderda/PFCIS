# Prompt-Free Coverless Image Steganography with Diffusion Models (PFCIS)

Official reference implementation for the latent noise editing algorithms presented in *"Prompt-Free Coverless Image Steganography with Diffusion Models"*.

---

## 📖 Overview

Existing diffusion-based steganography methods heavily rely on text prompts as keys or conditional guidance, which exposes the system to prompt guessing, semantic leakage, and cross-attention inversion errors. 

**PFCIS** introduces a **prompt-free** coverless image steganography framework. By abandoning conditional text guidance and directly manipulating the latent noise representations via symmetric keys, PFCIS achieves:
- **Zero semantic leakage** from prompt text.
- **Strict key-dependent security** with astronomical permutation search space ($>10^{1000}$).
- **High image synthesis quality and naturalness** adhering to the unconditional prior distribution of pre-trained diffusion models.

---

## 🛠️ Project Structure

```text
PFCIS/
├── noise_editor.py      # Core symmetric-key governed latent noise editing algorithms
├── demo.py              # Self-contained evaluation and verification script
├── requirements.txt     # Python dependencies
├── LICENSE              # Open source license
└── README.md            # Documentation
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/riderda/PFCIS.git
cd PFCIS
pip install -r requirements.txt
```

### 2. Run Demo

To evaluate all four latent noise editing strategies (Noise Shuffling, Stepwise Noising, Noise Flipping, and Dual-Key Shuffling):

```bash
python demo.py
```

### 3. Noise Editing API Example

```python
import torch
from noise_editor import NoiseEditor

editor = NoiseEditor()
# Standard latent noise tensor in diffusion models (B, C, H, W)
latent_noise = torch.randn(1, 4, 64, 64)

# 1. Noise Shuffling (Spatial Permutation)
shuffled = editor.shuffle_tensor(latent_noise, seed=42, ratio=1.0)

# 2. Stepwise Noising (Diffusion Forward Noising Step)
noised = editor.add_stepped_noise(latent_noise, seed=42, ratio=0.97)

# 3. Noise Flipping (Sign Inversion)
flipped = editor.noise_flip(latent_noise, seed=42, ratio=0.02)

# 4. Dual-Key Shuffling Extension (Hierarchical Multi-Seed Permutation)
dual_shuffled = editor.shuffle_tensor_dual_key(
    latent_noise, seed1=42, seed2=100, ratio1=0.3, ratio2=0.7
)
```

---

## 📜 Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{chen2026promptfree,
  title={Prompt-Free Coverless Image Steganography with Diffusion Models},
  author={Chen, Weida and Kou, Zheng and Chen, Weizhao and Yang, Zhen and Yu, Hang},
  journal={Scientific Reports},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the Apache 2.0 / MIT License - see the [LICENSE](LICENSE) file for details.
