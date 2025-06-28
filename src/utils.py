import os
import json
import psutil
import torch
from dataclasses import asdict
from datetime import datetime
from urllib.parse import quote
from colorama import Fore, Style


def print_memory(tag=""):
    ram = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"[{tag}] RAM {ram:.1f}MB  VRAM {vram:.1f}MB")


def save_image(img, cfg, out_dir="images"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.join(out_dir, timestamp)
    os.makedirs(dir_path, exist_ok=True)
    img_path = os.path.join(dir_path, "image.png")
    img.save(img_path)
    cfg_path = os.path.join(dir_path, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=4)
    print(f"{Fore.GREEN}Saved: {img_path}{Style.RESET_ALL}")
    print("file:///" + quote(img_path.replace("\\", "/")))
