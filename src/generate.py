from src.pipeline_factory import create_pipeline
from src.utils import print_memory
from src.autodetailer import apply_autodetailer

def generate(cfg):
    print_memory("before load")
    pipe_result = create_pipeline(cfg)

    # Если вернулся кортеж (pipe, refiner)
    if isinstance(pipe_result, tuple):
        pipe, refiner = pipe_result
        print("[refiner]", type(refiner).__name__)
        image = pipe(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            num_inference_steps=cfg.num_steps,
            guidance_scale=cfg.cfg_scale,
            width=cfg.width,
            height=cfg.height
        ).images[0]

        # Прогон через refiner
        image = refiner(
            image=image,
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            num_inference_steps=20,
            guidance_scale=cfg.cfg_scale,
            strength=0.3  # можно вынести в cfg
        ).images[0]

    else:
        pipe = pipe_result
        print("[refiner] not used")
        image = pipe(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            num_inference_steps=cfg.num_steps,
            guidance_scale=cfg.cfg_scale,
            width=cfg.width,
            height=cfg.height
        ).images[0]

    if getattr(cfg, "autodetailer", False):
        image = apply_autodetailer(image, cfg)

    return image
