from typing import Any
from PIL import Image
import torch
from diffusers import StableDiffusionUpscalePipeline


def apply_upscale(image: Image.Image, cfg: Any) -> Image.Image:
    """Upscale the given image using a Stable Diffusion upscaler."""
    up_cfg = getattr(cfg, "upscale", None)
    if not up_cfg:
        return image

    model = up_cfg.get("model", "stabilityai/stable-diffusion-x4-upscaler")
    steps = up_cfg.get("num_steps", 20)
    guidance = up_cfg.get("cfg_scale", cfg.cfg_scale)
    prompt = up_cfg.get("prompt", cfg.prompt)
    negative_prompt = up_cfg.get("negative_prompt", cfg.negative_prompt)

    pipe = StableDiffusionUpscalePipeline.from_pretrained(model, torch_dtype=torch.float16)
    if cfg.use_cuda and torch.cuda.is_available():
        pipe.to("cuda")
    else:
        pipe.enable_sequential_cpu_offload()

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]
    return result

