import os
import gc
import gradio as gr
import numpy as np
import torch
import json
import random
import config
import utils
import logging
import prompt_generator
from PIL import Image, PngImagePlugin
from datetime import datetime
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from config import (
    MODEL,
    MIN_IMAGE_SIZE,
    MAX_IMAGE_SIZE,
    USE_TORCH_COMPILE,
    ENABLE_CPU_OFFLOAD,
    OUTPUT_DIR,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_ASPECT_RATIO,
    sampler_list,
    aspect_ratios,
    style_list,
    # 設定
    TEXT_TO_PROMPT_ENABLED,
    DEFAULT_CATEGORY,
    DEFAULT_SERIES,
    DEFAULT_CHARACTER,
    series_list,
    character_list,
    category_list,
)
import time
from typing import List, Dict, Tuple, Optional

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
IS_COLAB = utils.is_google_colab() or os.getenv("IS_COLAB") == "1"
HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"

# PyTorch settings for better performance and determinism
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#torch.backends.cuda.matmul.allow_tf32 = True
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#logger.info(f"Using device: {device}")

# グローバル変数としてパイプラインを定義
pipe = None
vae = None

# スタイルリストから名前のみを抽出
style_names = [style["name"] for style in style_list]

def initialize_llm():
    """アプリケーション起動時にLLMだけを初期化する関数"""

    if TEXT_TO_PROMPT_ENABLED:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True

        logger.info("Loading LLM for prompt generation first...")
        prompt_generator.load_model()
        return "LLM loaded successfully"
    return "LLM loading skipped (disabled in config)"


def cleanup_old_images(output_dir, max_age_hours=1):
    """
    指定されたディレクトリ内の古い画像ファイル（PNG）を削除します

    Args:
        output_dir: 画像ファイルが保存されているディレクトリのパス
        max_age_hours: この時間（時間単位）より古いファイルを削除する
    """
    import os
    import time
    from datetime import datetime

    logger.info(f"Cleaning up images older than {max_age_hours} hours in {output_dir}")
    current_time = time.time()
    max_age_seconds = max_age_hours * 60 * 60
    deleted_count = 0

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        return 0

    # ディレクトリ内のすべてのファイルをチェック
    for filename in os.listdir(output_dir):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(output_dir, filename)
            file_age = current_time - os.path.getmtime(file_path)

            # 指定された時間より古いファイルを削除
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {str(e)}")

    if deleted_count > 0:
        logger.info(f"Deleted {deleted_count} old image files from {output_dir}")
    return deleted_count


# シリーズとキャラクターの紐付けを処理する
def get_character_series_mapping():
    try:
        mapping = config.config.get('text_to_prompt', {}).get('character_series_mapping', {})
        return mapping
    except Exception as e:
        logger.error(f"Failed to get character-series mapping: {str(e)}")
        return {}

# シリーズ名と表示名を分割
def parse_series_list():
    series_dict = {}
    display_series_list = []
    
    for item in series_list:
        if '|' in item:
            code, display = item.split('|', 1)
            series_dict[code] = display
            display_series_list.append(f"{code} / {display}")
        else:
            series_dict[item] = item
            display_series_list.append(item)
            
    return series_dict, display_series_list

# キャラクター名と表示名を分割
def parse_character_list():
    character_dict = {}
    display_character_list = []
    
    for item in character_list:
        if '|' in item:
            code, display = item.split('|', 1)
            character_dict[code] = display
            display_character_list.append(f"{code} / {display}")
        else:
            character_dict[item] = item
            display_character_list.append(item)
            
    return character_dict, display_character_list

# カテゴリー名と表示名を分割
def parse_category_list():
    category_dict = {}
    display_category_list = []
    
    for item in category_list:
        # カテゴリの表示が英語のみなので、そのまま表示
        category_dict[item] = item
        display_category_list.append(item)
            
    return category_dict, display_category_list

# 逆引き辞書の作成
def create_reverse_dict(original_dict):
    return {v: k for k, v in original_dict.items()}

# 表示名から内部コードを取得する関数
def get_code_from_display(display_name, reverse_dict):
    # 表示名が "code / display" 形式の場合
    if " / " in display_name:
        code = display_name.split(" / ")[0]
        return code
    # 元のコードの場合はそのまま返す
    return reverse_dict.get(display_name, display_name)

# 辞書とマッピングの作成
series_dict, display_series_list = parse_series_list()
character_dict, display_character_list = parse_character_list()
category_dict, display_category_list = parse_category_list()
reverse_series_dict = create_reverse_dict(series_dict)
reverse_character_dict = create_reverse_dict(character_dict)
character_series_mapping = get_character_series_mapping()

# 特定のシリーズに属するキャラクターのリストを取得
def get_characters_for_series(series_display_name):
    try:
        # 表示名からシリーズコードを取得
        series_code = get_code_from_display(series_display_name, reverse_series_dict)
        
        if not series_code:
            logger.warning(f"Unknown series: {series_display_name}")
            return display_character_list
            
        character_codes = character_series_mapping.get(series_code, [])
        if not character_codes:
            logger.warning(f"No characters found for series: {series_code}")
            return display_character_list
            
        # コードから表示名へ変換
        characters = [f"{code} / {character_dict.get(code, code)}" for code in character_codes]
        return characters
    except Exception as e:
        logger.error(f"Error getting characters for series: {str(e)}")
        return display_character_list

class GenerationError(Exception):
    """Custom exception for generation errors"""
    pass

def validate_prompt(prompt: str) -> str:
    """Validate and clean up the input prompt."""
    if not isinstance(prompt, str):
        raise GenerationError("Prompt must be a string")
    try:
        # Ensure proper UTF-8 encoding/decoding
        prompt = prompt.encode('utf-8').decode('utf-8')
        # Add space between ! and ,
        prompt = prompt.replace("!,", "! ,")
    except UnicodeError:
        raise GenerationError("Invalid characters in prompt")

    # Only check if the prompt is completely empty or only whitespace
    if not prompt or prompt.isspace():
        raise GenerationError("Prompt cannot be empty")
    return prompt.strip()

def validate_dimensions(width: int, height: int) -> None:
    """Validate image dimensions."""
    if not MIN_IMAGE_SIZE <= width <= MAX_IMAGE_SIZE:
        raise GenerationError(f"Width must be between {MIN_IMAGE_SIZE} and {MAX_IMAGE_SIZE}")

    if not MIN_IMAGE_SIZE <= height <= MAX_IMAGE_SIZE:
        raise GenerationError(f"Height must be between {MIN_IMAGE_SIZE} and {MAX_IMAGE_SIZE}")

def convert_text_to_prompt(
    novel_text: str,
    series_display_name: str = series_dict.get(DEFAULT_SERIES, DEFAULT_SERIES),
    character_display_name: str = character_dict.get(DEFAULT_CHARACTER, DEFAULT_CHARACTER),
    category: str = DEFAULT_CATEGORY,
) -> Tuple[str, str]:
    """テキストからプロンプトを生成する関数"""
    if not TEXT_TO_PROMPT_ENABLED:
        return "Text to Prompt機能は無効になっています", novel_text
    
    # 表示名からコードに変換
    series_name = get_code_from_display(series_display_name, reverse_series_dict)
    character_name = get_code_from_display(character_display_name, reverse_character_dict)

    try:
        logger.info(f"prompt_generator.generate_prompt")
        thinking, prompt = prompt_generator.generate_prompt(
            novel_text, series_name, character_name, category
        )
        return thinking, prompt
    except Exception as e:
        logger.error(f"Error in convert_text_to_prompt: {str(e)}")
        return f"エラーが発生しました: {str(e)}", novel_text

def load_image_model(timeout_seconds=120):
    """画像生成モデルをロードする関数"""
    global pipe, vae
    
    # LLMがロードされていれば解放
    if TEXT_TO_PROMPT_ENABLED:
        prompt_generator.unload_model()

    logger.info("Loading image generation model...")
    styles = {style["name"]: (style["prompt"], style.get("negative_prompt", "")) for style in style_list}
    
    # VAEを明示的にロード - subfolder パラメータを使用
    vae = AutoencoderKL.from_pretrained(
        MODEL,               # モデル名（例: "cagliostrolab/animagine-xl-4.0"）
        subfolder="vae",     # サブフォルダ名
        torch_dtype=torch.float16
    )
    
    # パイプラインにVAEを渡す
    pipe = utils.load_pipeline(MODEL, device, HF_TOKEN, vae=vae)
    
    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        logger.info("Model compiled with torch.compile")
    
    return "Image generation model loaded successfully"

def generate(
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = -1,
    custom_width: int = 1024,
    custom_height: int = 1024,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 28,
    sampler: str = "Euler a",
    aspect_ratio_selector: str = DEFAULT_ASPECT_RATIO,
    style_selector: str = "(None)",
    use_upscaler: bool = False,
    upscaler_strength: float = 0.55,
    upscale_by: float = 1.5,
    add_quality_tags: bool = True,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[List[str], Dict]:
    """Generate images based on the given parameters."""
    global pipe
    
    if pipe is None:
        load_image_model()
        logger.info(f"Loading image model status: {status}")
    
    start_time = time.time()
    upscaler_pipe = None
    backup_scheduler = None
    styles = {style["name"]: (style["prompt"], style.get("negative_prompt", "")) for style in style_list}

    try:
        # Memory management
        cleanup_old_images(OUTPUT_DIR)
        torch.cuda.empty_cache()
        gc.collect()

        # Input validation
        prompt = validate_prompt(prompt)
        if negative_prompt:
            negative_prompt = negative_prompt.encode('utf-8').decode('utf-8')

        validate_dimensions(custom_width, custom_height)

        # Set up generation
        if seed == 0:  # 0が入力された場合、ランダムなシード値を生成
            seed = random.randint(0, utils.MAX_SEED)
        generator = utils.seed_everything(seed)


        width, height = utils.aspect_ratio_handler(
            aspect_ratio_selector,
            custom_width,
            custom_height,
        )

        # Process prompts
        if add_quality_tags:
            prompt = "{prompt}, masterpiece, high score, great score, absurdres".format(prompt=prompt)

        prompt, negative_prompt = utils.preprocess_prompt(
            styles, style_selector, prompt, negative_prompt
        )

        width, height = utils.preprocess_image_dimensions(width, height)

        # Set up pipeline
        backup_scheduler = pipe.scheduler
        pipe.scheduler = utils.get_scheduler(pipe.scheduler.config, sampler)

        if use_upscaler:
            upscaler_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)

        # Prepare metadata
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "resolution": f"{width} x {height}",
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "style_preset": style_selector,
            "seed": seed,
            "sampler": sampler,
            "Model": "Animagine XL 4.0 Opt",
            "Model hash": "6327eca98b",
        }

        if use_upscaler:
            new_width = int(width * upscale_by)
            new_height = int(height * upscale_by)
            metadata["use_upscaler"] = {
                "upscale_method": "nearest-exact",
                "upscaler_strength": upscaler_strength,
                "upscale_by": upscale_by,
                "new_resolution": f"{new_width} x {new_height}",
            }
        else:
            metadata["use_upscaler"] = None

        logger.info(f"Starting generation with parameters: {json.dumps(metadata, indent=4)}")

        # Generate images
        if use_upscaler:
            latents = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="latent",
            ).images
            upscaled_latents = utils.upscale(latents, "nearest-exact", upscale_by)
            images = upscaler_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=upscaled_latents,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=upscaler_strength,
                generator=generator,
                output_type="pil",
            ).images
        else:
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pil",
            ).images

        # Save images
        if images:
            total = len(images)
            image_paths = []
            for idx, image in enumerate(images, 1):
                progress(idx/total, desc="Saving images...")
                path = utils.save_image(image, metadata, OUTPUT_DIR, IS_COLAB)
                image_paths.append(path)
                logger.info(f"Image {idx}/{total} saved as {path}")

        generation_time = time.time() - start_time
        logger.info(f"Generation completed successfully in {generation_time:.2f} seconds")
        metadata["generation_time"] = f"{generation_time:.2f}s"

        return image_paths, metadata

    except GenerationError as e:
        logger.warning(f"Generation validation error: {str(e)}")
        raise gr.Error(str(e))
    except Exception as e:
        logger.exception("Unexpected error during generation")
        raise gr.Error(f"Generation failed: {str(e)}")
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

        if upscaler_pipe is not None:
            del upscaler_pipe

        if backup_scheduler is not None and pipe is not None:
            pipe.scheduler = backup_scheduler

        utils.free_memory()

# シリーズが変更されたときに対応するキャラクターを更新
def update_character_list(series_display_name):
    characters = get_characters_for_series(series_display_name)
    if characters and len(characters) > 0:
        default_character = characters[0]
    else:
        default_character = display_character_list[0]
    
    return gr.update(choices=characters, value=default_character)

# テキストからプロンプトを生成する関数を追加
def process_text_to_prompt(
    novel_text: str,
    series_display_name: str = series_dict.get(DEFAULT_SERIES, DEFAULT_SERIES),
    character_display_name: str = character_dict.get(DEFAULT_CHARACTER, DEFAULT_CHARACTER),
    category: str = DEFAULT_CATEGORY,
) -> Tuple[str, str, Dict]:
    """テキストからプロンプトを生成して、UIに表示する関数"""
    try:
        # 必要に応じてLLMをロード
        if TEXT_TO_PROMPT_ENABLED and not hasattr(prompt_generator, "_model") or prompt_generator._model is None:
            prompt_generator.load_model()
        
        thinking, prompt_text = convert_text_to_prompt(novel_text, series_display_name, character_display_name, category)

        # プロンプト生成に関するメタデータ
        metadata = {
            "novel_text": novel_text[:100] + "..." if len(novel_text) > 100 else novel_text,
            "series": series_display_name,
            "character": character_display_name,
            "category": category,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return thinking, prompt_text, metadata

    except Exception as e:
        logger.exception("Error in process_text_to_prompt")
        error_message = f"プロンプト生成中にエラーが発生しました: {str(e)}"
        return error_message, "", {"error": str(e)}

# 生成されたプロンプトを画像生成パラメータにコピーする関数
def copy_prompt_to_generation(prompt_text):
    return prompt_text, gr.update(visible=False)

# スタイルが変更された時にプロンプトを更新する関数
def update_prompt_with_style(prompt_text, current_style, new_style):
    if prompt_text.strip() == "":
        return prompt_text
    
    # スタイル情報を取得
    styles = {style["name"]: (style["prompt"], style.get("negative_prompt", "")) for style in style_list}
    
    # 現在のスタイルのプロンプト部分を取得
    current_style_prompt = ""
    if current_style != "(None)":
        current_style_template = styles.get(current_style, ("", ""))[0]
        # {prompt} の部分を除外してスタイル部分だけを抽出
        if "{prompt}" in current_style_template:
            current_style_prompt = current_style_template.replace("{prompt}", "").strip()
            if current_style_prompt.startswith(","):
                current_style_prompt = current_style_prompt[1:].strip()
    
    # 新しいスタイルのプロンプト部分を取得
    new_style_prompt = ""
    if new_style != "(None)":
        new_style_template = styles.get(new_style, ("", ""))[0]
        # {prompt} の部分を除外してスタイル部分だけを抽出
        if "{prompt}" in new_style_template:
            new_style_prompt = new_style_template.replace("{prompt}", "").strip()
            if new_style_prompt.startswith(","):
                new_style_prompt = new_style_prompt[1:].strip()
    
    # 現在のプロンプトからスタイル部分を削除
    base_prompt = prompt_text
    if current_style_prompt:
        style_part = f", {current_style_prompt}"
        if style_part in base_prompt:
            base_prompt = base_prompt.replace(style_part, "")
    
    # 新しいスタイルを追加
    if new_style_prompt:
        if base_prompt.strip():
            base_prompt = f"{base_prompt.strip()}, {new_style_prompt}"
        else:
            base_prompt = new_style_prompt
    
    return base_prompt


initialize_llm()


# Create CSS with improved buttons and styling
custom_css = """
.header {
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(to right, #4a69bd, #6a89cc);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.title {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
}

.subtitle {
    font-size: 1.1rem;
    margin-top: 0.5rem;
    opacity: 0.9;
}

.subtitle-inline {
    font-size: 1.3rem;
    font-weight: 400;
    opacity: 0.9;
}

.section {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    border: 1px solid #e1e4e8;
}

.section-title {
    font-size: 1.3rem;
    margin-top: 0;
    margin-bottom: 1.2rem;
    color: #4a69bd;
    border-bottom: 2px solid #e1e4e8;
    padding-bottom: 0.5rem;
}

/* Improved button styling */
.primary-button {
    background-color: #4a69bd !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.2rem !important;
    border-radius: 8px !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.primary-button:hover {
    background-color: #3a539b !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
    transform: translateY(-1px) !important;
}

/* 思考プロセスとプロンプト出力のスタイルを改善 */
.thinking-output-label {
    font-weight: 600 !important;
    color: #4285f4 !important;
    background-color: transparent !important;
    margin-bottom: 4px !important;
}

.thinking-output {
    background-color: #f0f7ff !important;
    border-left: 4px solid #4285f4 !important;
    padding: 12px !important;
    border-radius: 6px !important;
    font-size: 0.95rem !important;
    color: #333 !important;
}

.generated-prompt-label {
    font-weight: 600 !important;
    color: #34a853 !important;
    background-color: transparent !important;
    margin-bottom: 4px !important;
    margin-top: 12px !important;
}

.generated-prompt {
    background-color: #f0fff4 !important;
    border-left: 4px solid #34a853 !important;
    padding: 12px !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    color: #333 !important;
}

.text-input-area {
    border: 1px solid #d0d7de;
    border-radius: 8px;
}

/* Add animation for loading states */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}

.loading {
    animation: pulse 1.5s infinite;
}

/* Gallery improvements */
.gallery-item {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease;
}

.gallery-item:hover {
    transform: scale(1.02);
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("<div class='header'><h1 class='title'>FanFic Illustrator <span class='subtitle-inline'>with Animagine XL 4.0 Opt</span></h1><p class='subtitle'>Illustrate your fan stories with beautiful AI-generated art<br>二次創作ファン小説にAIで魅力的な挿絵を</p></div>")
    
    with gr.Column():
        # Text Input Section
        with gr.Group(elem_classes=["section"]):
            gr.HTML("<h3 class='section-title'>1. Your Narrative / あなたの創作した物語</h3>")
            novel_text = gr.Textbox(
                label="",
                placeholder="Enter your fan story or narrative here... / ここにファンストーリーや物語を入力してください...",
                lines=10,
                elem_classes=["text-input-area"],
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    series_selector = gr.Dropdown(
                        choices=display_series_list,
                        value=display_series_list[0] if display_series_list else "",
                        label="Series / シリーズ",
                    )
                with gr.Column(scale=1):
                    character_selector = gr.Dropdown(
                        choices=get_characters_for_series(display_series_list[0] if display_series_list else ""),
                        value=display_character_list[0] if display_character_list else "",
                        label="Character / キャラクター",
                    )
            
            with gr.Row():
                category_selector = gr.Dropdown(
                    choices=display_category_list,
                    value=display_category_list[0] if display_category_list else "",
                    label="Illustration Type / イラストタイプ",
                )
            
            convert_btn = gr.Button("Generate Prompt / プロンプト生成", elem_classes=["primary-button"])
        
        # Thinking Process & Generated Prompt Section
        with gr.Group(elem_classes=["section"]):
            gr.HTML("<h3 class='section-title'>2. AI Interpretation / AIの解釈結果</h3>")
            
            gr.HTML("<div class='thinking-output-label'>AI Thought Process / AIの思考過程</div>")
            thinking_output = gr.Textbox(
                label="",
                lines=6,
                elem_classes=["thinking-output"],
                visible=True
            )
            
            gr.HTML("<div class='generated-prompt-label'>Generated Prompt / 生成されたプロンプト</div>")
            prompt_output = gr.Textbox(
                label="",
                lines=3,
                elem_classes=["generated-prompt"],
            )
            
            use_prompt_btn = gr.Button("Create Illustration with This Prompt / このプロンプトでイラスト作成", elem_classes=["primary-button"])
        
        # Image Generation Section
        with gr.Group(elem_classes=["section"]):
            gr.HTML("<h3 class='section-title'>3. Illustration Generation / イラスト生成</h3>")
            
            # 生成イラストを一番上に配置
            output_gallery = gr.Gallery(label="Generated Illustrations / 生成されたイラスト", show_label=True)
            
            # プロンプト入力欄
            prompt = gr.Textbox(
                label="Prompt / プロンプト",
                placeholder="Enter your prompt here... / ここにプロンプトを入力してください...",
                lines=3,
            )
            
            # 詳細設定のアコーディオン - デフォルトでは閉じている
            with gr.Accordion("Advanced Options / 詳細設定", open=False):

                # スタイルセレクター
                current_style = gr.State("(None)")  # 現在選択されているスタイルを保存
                style_selector = gr.Dropdown(
                    choices=style_names,
                    value="(None)",
                    label="Style / スタイル",
                    info="Select a style to apply to your prompt / プロンプトに適用するスタイルを選択",
                )

                # ネガティブプロンプト
                negative_prompt = gr.Textbox(
                    label="Negative Prompt / ネガティブプロンプト",
                    placeholder="What you don't want to see in the image / 画像に含めたくない要素",
                    value=DEFAULT_NEGATIVE_PROMPT,
                    lines=3,
                )
                
            # 「イラスト生成」を「イラスト再生成」に変更
            generate_btn = gr.Button("Regenerate Illustration / イラスト再生成", elem_classes=["primary-button"])
    
    # Setup event listeners
    # シリーズが変更されたときにキャラクターリストを更新するイベント
    series_selector.change(
        fn=update_character_list,
        inputs=[series_selector],
        outputs=[character_selector],
    )
    
    # スタイルが変更されたときにプロンプトを更新するイベント
    style_selector.change(
        fn=update_prompt_with_style,
        inputs=[prompt, current_style, style_selector],
        outputs=[prompt],
    ).then(
        fn=lambda x: x,
        inputs=[style_selector],
        outputs=[current_style],
    )
    
    # プロンプト生成ボタンのイベント
    convert_btn.click(
        fn=process_text_to_prompt,
        inputs=[
            novel_text,
            series_selector,
            character_selector,
            category_selector,
        ],
        outputs=[thinking_output, prompt_output, gr.JSON(visible=False)],
    )
    
    # プロンプトを画像生成に使用するボタンのイベント
    use_prompt_btn.click(
        fn=copy_prompt_to_generation,
        inputs=[prompt_output],
        outputs=[prompt, gr.Textbox(visible=False)],
    ).then(
        fn=load_image_model,
        inputs=[],
        outputs=[gr.Textbox(visible=False)],
    ).then(
        fn=lambda p, np, style: generate(
            prompt=p,
            negative_prompt=np,
            seed=0,  # デフォルトのseed値
            custom_width=832,  # デフォルトの幅
            custom_height=1216,  # デフォルトの高さ
            guidance_scale=5.0,  # デフォルトのguidance_scale
            num_inference_steps=28,  # デフォルトのnum_inference_steps
            sampler="Euler a",  # デフォルトのsampler
            aspect_ratio_selector=DEFAULT_ASPECT_RATIO,  # デフォルトのアスペクト比
            style_selector=style,  # スタイルセレクターの値を渡す
            use_upscaler=False,  # デフォルトのupscaler設定
            upscaler_strength=0.55,  # デフォルトのupscaler強度
            upscale_by=1.5,  # デフォルトのupscale値
            add_quality_tags=True,  # デフォルトの品質タグ設定
        ),
        inputs=[prompt, negative_prompt, style_selector],
        outputs=[output_gallery, gr.JSON(visible=False)],
    )
    
    # 画像生成ボタンのイベント
    generate_btn.click(
        fn=lambda p, np, style: generate(
            prompt=p,
            negative_prompt=np,
            seed=0,
            custom_width=832,
            custom_height=1216,
            guidance_scale=5.0,
            num_inference_steps=28,
            sampler="Euler a",
            aspect_ratio_selector=DEFAULT_ASPECT_RATIO,
            style_selector=style,
            use_upscaler=False,
            upscaler_strength=0.55,
            upscale_by=1.5,
            add_quality_tags=True,
        ),
        inputs=[prompt, negative_prompt, style_selector],
        outputs=[output_gallery, gr.JSON(visible=False)],
    )


# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=IS_COLAB)
