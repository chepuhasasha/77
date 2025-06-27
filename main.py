import argparse
from src.config import load_config
from src.generate import generate
from src.utils import save_image, print_memory

def main():
    parser = argparse.ArgumentParser(description="SDXL image generator")
    parser.add_argument(
        "--config",
        default="prompt.json",
        help="Path to JSON config file (по умолчанию prompt.json в рабочей папке)"
    )
    parser.add_argument(
        "--prompt",
        help="Override prompt из конфига"
    )
    
    args = parser.parse_args()
    cfg = load_config(json_path=args.config)

    # Переопределяем prompt, если передали через CLI
    if args.prompt:
        cfg.prompt = args.prompt

    # Выбираем режим
    if cfg.mode == "generate":
        img = generate(cfg)

    save_image(img, cfg)
    print_memory("end")

if __name__ == "__main__":
    main()