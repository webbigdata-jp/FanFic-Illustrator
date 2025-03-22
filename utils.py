import gc
import os
import random
import numpy as np
import json
import torch
import uuid
from PIL import Image, PngImagePlugin
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple, Any, List
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    AutoencoderKL,
    StableDiffusionXLPipeline,
)
import logging

MAX_SEED = np.iinfo(np.int32).max

def is_space_environment():
    return "SPACE_ID" in os.environ and os.environ.get("SYSTEM") == "spaces"

def seed_everything(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def parse_aspect_ratio(aspect_ratio: str) -> Optional[Tuple[int, int]]:
    if aspect_ratio == "Custom":
        return None
    width, height = aspect_ratio.split(" x ")
    return int(width), int(height)


def aspect_ratio_handler(
    aspect_ratio: str, custom_width: int, custom_height: int
) -> Tuple[int, int]:
    if aspect_ratio == "Custom":
        return custom_width, custom_height
    else:
        width, height = parse_aspect_ratio(aspect_ratio)
        return width, height


def get_scheduler(scheduler_config: Dict, name: str) -> Optional[Callable]:
    scheduler_factory_map = {
        "DPM++ 2M Karras": lambda: DPMSolverMultistepScheduler.from_config(
            scheduler_config, use_karras_sigmas=True
        ),
        "DPM++ SDE Karras": lambda: DPMSolverSinglestepScheduler.from_config(
            scheduler_config, use_karras_sigmas=True
        ),
        "DPM++ 2M SDE Karras": lambda: DPMSolverMultistepScheduler.from_config(
            scheduler_config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
        ),
        "Euler": lambda: EulerDiscreteScheduler.from_config(scheduler_config),
        "Euler a": lambda: EulerAncestralDiscreteScheduler.from_config(
            scheduler_config
        ),
        "DDIM": lambda: DDIMScheduler.from_config(scheduler_config),
    }
    return scheduler_factory_map.get(name, lambda: None)()


def free_memory() -> None:
    """Free up GPU and system memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def preprocess_prompt(
    style_dict,
    style_name: str,
    positive: str,
    negative: str = "",
    add_style: bool = True,
) -> Tuple[str, str]:
    p, n = style_dict.get(style_name, style_dict["(None)"])

    if add_style and positive.strip():
        formatted_positive = p.format(prompt=positive)
    else:
        formatted_positive = positive

    combined_negative = n
    if negative.strip():
        if combined_negative:
            combined_negative += ", " + negative
        else:
            combined_negative = negative

    return formatted_positive, combined_negative


def common_upscale(
    samples: torch.Tensor,
    width: int,
    height: int,
    upscale_method: str,
) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        samples, size=(height, width), mode=upscale_method
    )


def upscale(
    samples: torch.Tensor, upscale_method: str, scale_by: float
) -> torch.Tensor:
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    return common_upscale(samples, width, height, upscale_method)


def preprocess_image_dimensions(width, height):
    if width % 8 != 0:
        width = width - (width % 8)
    if height % 8 != 0:
        height = height - (height % 8)
    return width, height


def save_image(image, metadata, output_dir, is_colab):
    if is_colab:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{current_time}.png"   
    else:
        filename = str(uuid.uuid4()) + ".png"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    metadata_str = json.dumps(metadata)
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", metadata_str)
    image.save(filepath, "PNG", pnginfo=info)
    return filepath
    
    
def is_google_colab():
    try:
        import google.colab
        return True
    except:
        return False


def load_pipeline(model_name: str, device: torch.device, hf_token: Optional[str] = None, vae: Optional[AutoencoderKL] = None) -> Any:
    """Load the Stable Diffusion pipeline."""
    try:
        logging.info(f"Loading pipeline from {model_name}...")

        # Choose the right loading method based on file path or model ID
        if os.path.exists(model_name) and os.path.isdir(model_name):
            # It's a local directory path
            if os.path.exists(os.path.join(model_name, "animagine-xl-4.0.safetensors")):
                # Load from single file if it exists
                pipe = StableDiffusionXLPipeline.from_single_file(
                    os.path.join(model_name, "animagine-xl-4.0.safetensors"),
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    custom_pipeline="lpw_stable_diffusion_xl",
                    add_watermarker=False
                )
            else:
                # Load the VAE first to ensure it's not None
                vae_path = os.path.join(model_name, "vae")
                if vae is None and os.path.exists(vae_path):
                    logging.info(f"Loading VAE from {vae_path}...")
                    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)

                # Load pipeline from directory
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    vae=vae,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    custom_pipeline="lpw_stable_diffusion_xl",
                    add_watermarker=False
                )
        elif model_name.endswith(".safetensors"):
            # It's a single file
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                custom_pipeline="lpw_stable_diffusion_xl",
                add_watermarker=False
            )
        else:
            # It's a Hugging Face model ID
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                vae=vae,
                token=hf_token,
                torch_dtype=torch.float16,
                use_safetensors=True,
                custom_pipeline="lpw_stable_diffusion_xl",
                add_watermarker=False
            )

        # デバイス移動の部分を修正
        if "SPACE_ID" in os.environ and os.environ.get("SYSTEM") == "spaces":
            # Stateless GPU環境ではデバイス移動を特別に扱う
            return pipe
        else:
            # 通常の環境では以前のコードを使用
            pipe.to(device)
            return pipe

        logging.info("Pipeline loaded successfully!")
        return pipe
    except Exception as e:
        logging.error(f"Failed to load pipeline: {str(e)}", exc_info=True)
        raise


