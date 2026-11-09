"""
PFCIS: Prompt-Free Coverless Image Steganography with Diffusion Models
Sender Module: Prompt-Free Image Hiding Workflow
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Tuple
from diffusers import StableDiffusionPipeline, DDIMScheduler
from noise_editor import NoiseEditor


class PromptFreeStegoSender:
    """
    Sender-side workflow for prompt-free coverless image steganography.
    Discards text prompts entirely and relies solely on symmetric keys.
    """

    def __init__(
        self,
        model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        num_inference_steps: int = 200,
        guidance_scale: float = 3.5,
        device: Optional[torch.device] = None
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32
        ).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.step_size = 1000 // num_inference_steps
        self.editor = NoiseEditor()

        self._transform = transforms.Compose([
            transforms.Resize((512, 512), transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def preprocess_image(self, img_path: str) -> Tuple[Image.Image, torch.Tensor]:
        raw_img = Image.open(img_path).convert("RGB")
        img_tensor = self._transform(raw_img).unsqueeze(0).to(self.device, dtype=torch.float32)
        return raw_img, img_tensor

    @torch.no_grad()
    def invert_to_noise(self, latents: torch.Tensor) -> torch.Tensor:
        """DDIM inversion with an empty text prompt."""
        text_embeddings = self.pipe._encode_prompt("", self.device, 1, True, "")
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = reversed(self.pipe.scheduler.timesteps)

        cur_latents = latents.clone()
        for i in range(1, self.num_inference_steps - 1):
            t = timesteps[i]
            latent_input = torch.cat([cur_latents] * 2)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)

            noise_pred = self.pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
            uncond, text = noise_pred.chunk(2)
            noise_pred = uncond + self.guidance_scale * (text - uncond)

            curr_t = max(0, t.item() - self.step_size)
            next_t = t
            alpha_curr = self.pipe.scheduler.alphas_cumprod[curr_t]
            alpha_next = self.pipe.scheduler.alphas_cumprod[next_t]

            cur_latents = (cur_latents - torch.sqrt(1 - alpha_curr) * noise_pred) * (
                torch.sqrt(alpha_next) / torch.sqrt(alpha_curr)
            ) + torch.sqrt(1 - alpha_next) * noise_pred

        return cur_latents

    @torch.no_grad()
    def generate_image(self, start_noise: torch.Tensor) -> Image.Image:
        """Deterministic DDIM sampling to produce steganographic image."""
        text_embeddings = self.pipe._encode_prompt("", self.device, 1, True, "")
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        latents = start_noise.clone()
        for i in range(self.num_inference_steps):
            t = timesteps[i]
            latent_input = torch.cat([latents] * 2)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)

            noise_pred = self.pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
            uncond, text = noise_pred.chunk(2)
            noise_pred = uncond + self.guidance_scale * (text - uncond)

            prev_t = max(1, t.item() - self.step_size)
            alpha_t = self.pipe.scheduler.alphas_cumprod[t.item()]
            alpha_prev = self.pipe.scheduler.alphas_cumprod[prev_t]

            pred_x0 = (latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            dir_xt = torch.sqrt(1 - alpha_prev) * noise_pred
            latents = torch.sqrt(alpha_prev) * pred_x0 + dir_xt

        latents = latents / self.pipe.vae.config.scaling_factor
        img = self.pipe.vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        pil_img = Image.fromarray((img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255).astype("uint8"))
        return pil_img

    def hide(
        self,
        img_path: str,
        key: int,
        strategy: str = "shuffle",
        ratio: float = 1.0
    ) -> Image.Image:
        """
        End-to-end prompt-free hiding:
          1. Encode image to VAE latents
          2. DDIM Inversion to initial latent noise
          3. Symmetric-key latent noise editing
          4. DDIM sampling to synthesize stego image
        """
        _, img_tensor = self.preprocess_image(img_path)
        with torch.no_grad():
            latents = self.pipe.vae.encode(img_tensor).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor

        noise = self.invert_to_noise(latents)

        # Apply forward noise editing
        if strategy == "shuffle":
            edited_noise = self.editor.shuffle_tensor(noise, seed=key, ratio=ratio)
        elif strategy == "stepwise":
            self.pipe.scheduler.set_timesteps(1000, device=self.device)
            edited_noise = self.editor.add_stepped_noise(noise, self.pipe.scheduler, seed=key, ratio=ratio)
        elif strategy == "flip":
            edited_noise = self.editor.noise_flip(noise, seed=key, ratio=ratio)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        stego_image = self.generate_image(edited_noise)
        return stego_image
