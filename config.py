import os
import tomli
from typing import Dict, Any, List

def load_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), 'config.toml')
    with open(config_path, 'rb') as f:
        config = tomli.load(f)
        return config

# Load configuration
config = load_config()

# Export variables for backward compatibility
MODEL = os.getenv("MODEL", config['model'].get('path', 'cagliostrolab/animagine-xl-4.0'))

MIN_IMAGE_SIZE = int(os.getenv("MIN_IMAGE_SIZE", config['model']['min_image_size']))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", config['model']['max_image_size']))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", str(config['model']['use_torch_compile'])).lower() == "true"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", str(config['model']['enable_cpu_offload'])).lower() == "true"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", config['model']['output_dir'])

DEFAULT_NEGATIVE_PROMPT = config['prompts']['default_negative']
DEFAULT_ASPECT_RATIO = config['prompts']['default_aspect_ratio']

sampler_list = config['samplers']['list']
aspect_ratios = config['aspect_ratios']['list']
style_list = config['styles']

# Text to prompt settings
TEXT_TO_PROMPT_ENABLED = config.get('text_to_prompt', {}).get('enabled', False)
DEFAULT_CATEGORY = config.get('text_to_prompt', {}).get('default_category', 'sfw')
DEFAULT_SERIES = config.get('text_to_prompt', {}).get('default_series', 'original')
DEFAULT_CHARACTER = config.get('text_to_prompt', {}).get('default_character', 'original character')

# シリーズとキャラクターのリストを取得
series_list = config.get('text_to_prompt', {}).get('series', {}).get('list', [])
character_list = config.get('text_to_prompt', {}).get('characters', {}).get('list', [])
category_list = config.get('text_to_prompt', {}).get('categories', {}).get('list', [])
