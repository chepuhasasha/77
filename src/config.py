import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict
from colorama import Fore, Style

@dataclass
class Config:
    mode: str = "generate"
    model: Optional[str] = "stabilityai/stable-diffusion-3.5-medium"
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    refiner_model: Optional[str] = None
    embedding_model: Optional[str] = None
    num_steps: int = 28
    cfg_scale: float = 7.0
    use_cuda: bool = True
    image_path: Optional[str] = None
    mask_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    width: int = 512
    height: int = 512
    sampling_method: str = "Euler"
    clip_skip: int = 0
    upscale: Optional[Dict] = field(default=None)


def load_config(json_path: str = "prompt.json") -> Config:
    cfg = Config()
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        print(f"{Fore.GREEN}Loaded {json_path}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}{json_path} not found, using defaults{Style.RESET_ALL}")
    # Валидация
    if not (cfg.model or cfg.checkpoint_path):
        raise ValueError("Нужно указать model или checkpoint_path")
    if cfg.mode == "inpaint" and not (cfg.image_path and cfg.mask_path):
        raise ValueError("Для inpaint нужны image_path и mask_path")
    return cfg