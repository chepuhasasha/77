import os
import shutil
import torch
from src.embed_loader import inject_sdxl_embedding
# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    DPMSolverMultistepScheduler
)

def create_pipeline(cfg):
    # Prepare models root
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'models'))
    os.makedirs(root, exist_ok=True)

    # Load base or checkpoint
    if cfg.checkpoint_path and os.path.isfile(cfg.checkpoint_path):
        cp_dir = os.path.join(root, 'checkpoints')
        os.makedirs(cp_dir, exist_ok=True)
        local_cp = os.path.join(cp_dir, os.path.basename(cfg.checkpoint_path))
        if not os.path.isfile(local_cp):
            shutil.copy(cfg.checkpoint_path, local_cp)
        try:
            pipe = StableDiffusionXLPipeline.from_single_file(
                local_cp, torch_dtype=torch.float16, safety_checker=None
            )
        except Exception:
            pipe = StableDiffusionPipeline.from_single_file(
                local_cp, torch_dtype=torch.float16, safety_checker=None
            )
        # Приведение компонентов к float32
        if hasattr(pipe, 'text_encoder'):
            pipe.text_encoder.to(dtype=torch.float32)
        if hasattr(pipe, 'text_encoder_2'):
            pipe.text_encoder_2.to(dtype=torch.float32)
        if hasattr(pipe, 'unet'):
            pipe.unet.to(dtype=torch.float32)
    else:
        # HF model
        parts = cfg.model.split('/')
        hf_dir = os.path.join(root, *parts)
        os.makedirs(hf_dir, exist_ok=True)

        def _load(cls):
            if os.path.isdir(hf_dir) and os.listdir(hf_dir):
                return cls.from_pretrained(hf_dir, torch_dtype=torch.float16, local_files_only=True, safety_checker=None)
            m = cls.from_pretrained(cfg.model, torch_dtype=torch.float16, safety_checker=None)
            m.save_pretrained(hf_dir)
            return m

        m_lower = cfg.model.lower()
        if 'inpaint' in m_lower:
            cls = StableDiffusionInpaintPipeline
        elif '3.5' in m_lower and not m_lower.endswith('inpaint'):
            cls = StableDiffusion3Pipeline
        elif 'xl' in m_lower:
            cls = StableDiffusionXLPipeline
        else:
            cls = StableDiffusionPipeline
        pipe = _load(cls)
        # Приведение компонентов к float32
        if hasattr(pipe, 'text_encoder'):
            pipe.text_encoder.to(dtype=torch.float32)
        if hasattr(pipe, 'text_encoder_2'):
            pipe.text_encoder_2.to(dtype=torch.float32)
        if hasattr(pipe, 'unet'):
            pipe.unet.to(dtype=torch.float32)

    # Inject embedding if present
    if getattr(cfg, 'embedding_model', None):
        emb_path = cfg.embedding_model
        if not os.path.isabs(emb_path):
            emb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, emb_path))
        print(f"[create_pipeline] Loading embedding from: {emb_path}")
        inject_sdxl_embedding(pipe, emb_path, token="sdxl_cyberrealistic_simpleneg")

    # Scheduler
    sched = cfg.sampling_method.lower()
    if sched.startswith('dpm'):
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type='sde-dpmsolver++', use_karras_sigmas=True)
    elif 'flowmatch' in sched:
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # Clip-skip
    skip = getattr(cfg, 'clip_skip', 0)
    if skip > 0:
        enc = pipe.text_encoder
        enc.text_model.encoder.layers = enc.text_model.encoder.layers[:-skip]
        if hasattr(pipe, 'text_encoder_2'):
            enc2 = pipe.text_encoder_2
            enc2.text_model.encoder.layers = enc2.text_model.encoder.layers[:-skip]

    # Device
    if cfg.use_cuda and torch.cuda.is_available():
        print('Используется CUDA')
        pipe.to('cuda')
    else:
        print('Используется CPU с выгрузкой')
        pipe.enable_sequential_cpu_offload()

    # Refiner
    refiner = None
    if getattr(cfg, 'refiner_model', None):
        ref_path = cfg.refiner_model
        if not os.path.isabs(ref_path):
            ref_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, ref_path))
        print(f"[create_pipeline] Checking refiner path: {ref_path}")
        if os.path.isfile(ref_path):
            print(f"[create_pipeline] Using local refiner: {ref_path}")
            refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(ref_path, torch_dtype=torch.float16, safety_checker=None)
        elif os.path.isdir(ref_path):
            print(f"[create_pipeline] Loading HF refiner from: {ref_path}")
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(ref_path, torch_dtype=torch.float16, local_files_only=True, safety_checker=None)
        else:
            print('[create_pipeline] Refiner path not found')
        if refiner:
            if cfg.use_cuda and torch.cuda.is_available():
                refiner.to('cuda')
            else:
                refiner.enable_sequential_cpu_offload()

    if refiner:
        print(f"[create_pipeline] Refiner loaded: {type(refiner).__name__}")
        return (pipe, refiner)
    print('[create_pipeline] Refiner not used')
    return pipe