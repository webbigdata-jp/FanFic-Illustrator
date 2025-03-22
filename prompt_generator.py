import os
import logging
import torch
import utils
from typing import Tuple, Optional, Dict, Any
import gc


# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_prompt_logger():
    """プロンプト生成専用のファイルログハンドラをセットアップします"""
    import os
    import logging
    from logging.handlers import RotatingFileHandler

    # プロンプト記録用のロガー
    prompt_logger = logging.getLogger('prompt_generator.io')
    prompt_logger.setLevel(logging.INFO)

    # ログディレクトリの作成
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # ログファイルのパス
    log_file = os.path.join(log_dir, 'prompt_generation.log')

    # ローテーションするファイルハンドラ（5MBごとに最大5ファイル）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=5,
        encoding='utf-8'
    )

    # フォーマッタの設定
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    # ハンドラをロガーに追加（既存のハンドラがあれば追加しない）
    if not prompt_logger.handlers:
        prompt_logger.addHandler(file_handler)

    return prompt_logger

# ロガーのセットアップ
prompt_io_logger = setup_prompt_logger()


def log_prompt_io(novel_text, series_name, character_name, category, thinking, prompt_text):
    """プロンプト生成の入力と出力をログに記録する関数"""
    # 入力テキストが長すぎる場合は省略
    if len(novel_text) > 500:
        logged_text = novel_text[:500] + "...(truncated)"
    else:
        logged_text = novel_text
        
    log_entry = (
        f"\n{'='*80}\n"
        f"INPUT:\n"
        f"Series: {series_name}\n"
        f"Character: {character_name}\n"
        f"Category: {category}\n"
        f"Text: {logged_text}\n\n"
        f"OUTPUT:\n"
        f"Thinking: {thinking}\n\n"
        f"Prompt: {prompt_text}\n"
        f"{'='*80}\n"
    )
    
    prompt_io_logger.info(log_entry)

# グローバル変数
_model = None
_tokenizer = None


def load_model():
    """モデルをロードする関数"""
    global _model, _tokenizer

    # すでにロードされている場合はスキップ
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading prompt generation model...")

        # 使用するモデル名 - 環境変数から取得するか、デフォルト値を使用
        model_name = os.getenv("PROMPT_MODEL_NAME", "webbigdata/FanFic-Illustrator")

        # HuggingFaceからモデルを直接ロードする
        model_path = model_name

        device_map = "auto" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and hasattr(torch, 'bfloat16') else torch.float16
        logger.info(f"Using device: {device_map} for prompt generation model")

        # モデルの読み込み
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            use_cache=True,
            low_cpu_mem_usage=True,
        )

        # トークナイザーの読み込み
        _tokenizer = AutoTokenizer.from_pretrained(model_path)

        # パッドトークンが設定されていない場合は設定
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        logger.info("Prompt generation model loaded successfully!")
        return _model, _tokenizer

    except Exception as e:
        logger.error(f"Failed to load prompt generation model: {str(e)}")
        raise

def unload_model():
    """メモリからモデルをアンロードする関数"""
    global _model, _tokenizer
    
    if _model is not None:
        del _model
        _model = None
    
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    
    # メモリの解放
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    logger.info("Prompt generation model unloaded")

def generate_prompt(
    novel_text: str,
    series_name: str = "original",
    character_name: str = "original character",
    category: str = "general"
) -> Tuple[str, str]:
    try:
        # モデルとトークナイザーの読み込み
        model, tokenizer = load_model()
        logger.info("Loading model, tokenizer is ok...")
        
        # 長すぎる入力のトリミング（トークン数の制限）
        max_input_length = 5072 #1024
        #if len(novel_text) > max_input_length * 4:  # 大まかな文字数の制限
        #    novel_text = novel_text[:max_input_length * 4]
        #    logger.warning(f"Input text was too long and has been truncated")
        
        # システムプロンプト
        system = "あなたは文章の一説を指定ジャンル・キャラクターが登場するシーンに書き換え、そのシーンに合った挿絵を作成するために画像生成AI用プロンプトを作成する優秀なプロンプトエンジニアです"
        
        # ユーザープロンプト
        prompt = f"""### 小説のコンテキストを補足する情報
content category: {category}
series name: {series_name}
series description: {series_name} series
character name: {character_name}
character description: {character_name} from {series_name}

### 小説データ
{novel_text}

まず<think>内で以下のように思考を整理します。

<think>
concept: イラストのコンセプトを考えます。小説の内容から主題、設定、雰囲気を理解し、どのようなイラストが最も適切か、全体の構成を考えます
- 人数: 挿絵の中に登場させる人数を考えます。作品に登場する人物の数や重要性を考慮し、メインで描くべき人物やサブキャラクターについても検討してください
- キャラクター名/シリーズ名: 既存作品のキャラクター/シリーズか、オリジナル作品かを考えます。既存作品の場合は、原作の設定や特徴を尊重した表現方法も考慮してください
- ポーズ/構図: ポーズ/構図指定に使うタグを考えます。物語の場面において、キャラクターがどのような体勢/状況にあるのか、どのアングルから描くと効果
的かを検討してください
- 背景/環境: 背景/環境指定に使うタグを考えます。物語の舞台設定や時間帯、天候など、雰囲気を表現するために必要な背景要素を詳しく考えてください
- 描画スタイル/テクニック: 描画スタイル/テクニックに使うタグを考えます。物語のジャンルや雰囲気に合わせて、どのような画風や技法が適しているかを検討してください
- 身体的特徴/画面上の物体: 身体的特徴/画面上の物体に関連するタグを考えます。キャラクターの外見的特徴や、シーンに必要な小道具、アイテムなどを詳細に考えてください
</think>

改行の場所も含めて、この順序と書式を厳密に守ってください。
各項目は上記の順序と書式で記述してください。具体的かつ詳細に説明し、十分な長さで考察してください（<think>タグ全体で600-800文字程度が望ましいです）

その後、思考結果に基づき<prompt>内に英単語を18単語ほどカンマで区切って出力してください。キャラクター名/シリーズ名は指定されていたら必ず含めます。
日本語は使用しないでください。 最も重要で適切なタグを選び、有効なプロンプトとなるよう考慮してください

### 使用可能な英単語
出力時には以下のタグを優先して使用し、足りない場合は一般的な英単語で補足します
masterpiece, best quality, highresなどの品質に関連するタグは後工程で付与するのでつけてはいけません

**人数/性別**:
- 風景や動物を中心に描画する時: no_human
- 女性の人数: 1girl, 2girls, 3girls, multiple girls
- 男性の人数: 1boy, 2boys, 3boys, multiple boys
- 1girlや1boy指定時にキャラクター中心の構図にするために追加で指定: solo

**ポーズ/構図**:
- 視点: from above, from behind, from below, looking at viewer, straight-on, looking at another, looking back, out of frame, on back, from side, looking to the side, feet out of frame, sideways, three quarter view, looking up, looking down, looking ahead, dutch angle, high up, from outside, pov, vanishing point
- 姿勢/行動: battle, chasing, fighting, leaning, running, sitting, squatting, standing, walking, arm up, arms up, against wall, against tree, holding, spread legs, lying, straddling, flying, holding weapon, clothes lift, hand on own cheek, scar on cheek, hand on another's cheek, kissing cheek, cheek-to-cheek, bandaid on cheek, finger to cheek, hands on another's cheeks, hand on own hip, hand over face, v, kneeling, arabesque (pose), body roll, indian style, standing on one leg, hugging own legs, seiza, nuzzle, unsheathing, holding weapon, holding sword, holding gun, trembling

**背景/環境**:
- 構図/芸術ジャンル: landscape, portrait, still life, group shot, cowboy shot, upper body, full body, detailed face, depth of field, intricate details, cinematic lighting, detailed background, detailed, extremely detailed, perfect composition, detailed face, solo focus, detailed face and body, character focus, intricate, sharp focus, male focus
- 色彩/装飾: greyscale, sepia, blue theme, flat color, high contrast, limited palette, border, cinematic, scenery, rendered, contrast, rich contrast, volumetric lighting, high contrast, glowing
- 背景/風景: checkered background, simple background, indoors, outdoors, jungle, mountain, beach, forest, city, school, cafe, white background, sky
- 時間帯: day, night, twilight, morning, sunset, dawn, dusk
- 天気: sunny, rain, snow, cloud, storm, wind, fogg

**描画スタイル/テクニック**:
- 技法: 3D, oekaki, pixel art, sketch, watercolor, oil painting, digital art, illustration, photorealistic, anime, monochrome, retro color, source anime, cg, realistic
- 表現手法: animalization, personification, science fiction, cyberpunk, steampunk, fantasy, dark novel style, anime style, realistic style, graphic novel style, comic, concept art
- 媒体/伝統的技法: traditional media, marker (medium), watercolor (medium), graphite (medium), official art, sketch, artbook, cover
- 絵柄の年代(指定された時のみ利用): newest, year 1980, year 2000, year 2010, year 1990, year 2020

**身体的特徴/画面上の物体**:
- キャラクター属性/職業/クラス: student, teacher, soldier, knight, wizard, ninja, doctor, artist, musician, athlete, virtual youtuber, chibi, maid
- 表情: angry, blush stickers, drunk, grin, aroused, happy, sad, smile, laugh, crying, surprised, worried, nervous, serious, drunk, blush, aroused, :d, tongue out, sweatdrop, tongue out, :o, tears, tearing up, scared

- 身体的特徴: {{'髪型/髪色': ['long hair', 'short hair', 'twintails', 'ponytail', 'braid', 'bun', 'curly hair', 'straight hair', 'messy hair', 'blonde hair', 'black hair', 'brown hair', 'red hair', 'blue hair', 'green hair', 'white hair', 'purple hair', 'grey hair', 'ahoge', 'sidelocks', 'side ponytail', 'perfect hair', 'tail', 'multicolored hair', 'wavy hair', 'bangs', 'blunt bangs', 'twintails', 'hair between eyes', 'very long hair', 'braid', 'curly hair', 'braided ponytail', 'hand in own hair', 'hair over one eye', 'hair flower', 'two-tone hair', 'streaked hair', 'two side up'], '目の色': ['blue eyes', 'brown eyes', 'green eyes', 'red eyes', 'black eyes', 'purple eyes', 'yellow eyes', 'heterochromia', 'detailed eyes', 'glowing eyes', 'beatiful eyes', 'closed eyes', 'one eye closed'], '身体部位': ['bare shoulders', 'bare arms', 'bare legs', 'barefoot', 'abs', 'flat chest', 'small breasts', 'medium breasts', 'asymmetrical breasts', 'pointy breasts', 'sagging breasts', 'clenched teeth', 'pointy ears', 'perfect anatomy', 'closed mouth', 'long sleeves', 'open mouth', 'pale skin', 'collarbone', 'midriff', 'perfect anatomy', 'bare arms', 'thighs', 'parted lips', 'tongue', 'tanlines', 'dot nose', 'goggles on head', 'armpits', 'nail polish', 'mole', 'feet', 'lips', 'dark-skinned female', 'zettai ryouiki', 'shiny skin'], '身体部位(獣人、擬人化時のみ使用)': ['animal ears', 'cat ears', 'horse ears', 'horse girl', 'fang', 'teeth', 'horns', 'tail'], '服装/装飾品': ['uniform', 'suit', 'dress', 'casual wear', 'formal wear', 'belt', 'detached sleeves', 'swimsuit', 'kimono', 'armor', 'hat', 'glasses', 'white shirt', 'shirt', 'jewelry', 'necklace', 'earrings', 'bracelet', 'watch', 'ribbon', 'hair ribbon', 'scarf', 'gloves', 'boots', 'high heels', 'hair ornament', 'jacket', 'glasses', 'skirt', 'long sleeves', 'short sleeves', 'thighhighs', 'underwear', 'school uniform', 'swimsuit', 'panties', 'hair bow', 'bikini', 'miniskirt', 'fingerless gloves', 'bowtie', 'serafuku', 'japanese clothes', 'choker', 'pants', 'wings', 'open clothes', 'pantyhose', 'pleated skirt', 'frills', 'necktie', 'shorts', 'collared shirt', 'leather armor', 'hairband', 'shoes', 'sleeveless', 'alternate costume', 'socks', 'fingering', 'denim shorts', 'epaulettes', 'santa costume', 'ribbon-trimmed sleeves', 'black bowtie', 'gym uniform', 'white bra', 'angel wings', 'crossdressing', 'cuffs', 'halo', 'high heels', 'apron', 'red bow', 'vest', 'open jacket', 'white panties', 'leotard', 'coat', 'black jacket', 'high heels', 'black pantyhose', 'see-through', 'miniskirt', 'elbow gloves', 'wide sleeves', 'white thighhighs', 'fur trim', 'plaid', 'one-piece swimsuit', 'maid headdress', 'ascot', 'high-waist skirt']}}
- 体液: blood, saliva, sweat, tears
- 前景/持ち物/操作物: sword, katana, sheath, gun, book, phone, bag, umbrella, instrument, vehicle, food, drink, guitar, piano, violin, drums, flute, car, bicycle, motorcycle, airplane, ship, flower, weapon, heart, speech bubble, carriage, locomotive
- 生物: dog, cat, horse, bird, fish, dragon, unicorn, monster, fox, wolf, bear, tiger, lion, dragon, fairy, ghost, zombie, vampire

**性的表現（sensitive, nsfw, explicitのいずれかを指定した時のみ使用可）**:
- 身体部位女性専用: cleavage, backboob, sideboob, underboob, navel, huge breasts, large breasts
- 身体部位男性専用: topless male, necktie between pectorals, loose necktie, bare pectorals, male underwear, fundoshi
- 身体部位共通: open shirt, unbuttoned shirt, seductive smile, bare back, groin, groin tendon, midriff

### 出力
"""
        
        # メッセージ形式に整形
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        
        # トークナイゼーション
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        logger.info("tokenizer.apply_chat_template is ok...")
        
        # 長すぎる入力のトリミング
        if inputs.shape[1] > max_input_length:
            inputs = inputs[:, :max_input_length]
            logger.warning(f"Input tokens were too many and have been truncated to {max_input_length}")
        
        # 生成
        logger.info("before torch.no_grad")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs,
                num_beams=3,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.5,
                top_p=0.95,
                repetition_penalty=1.0,
                #dry_multiplier=0.5,
                top_k = 40,
                min_p = 0.00,
                pad_token_id=tokenizer.pad_token_id,
            )

        logger.info("after ttorch.no_grad")
        # デコード
        full_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # モデルが生成したメッセージ部分だけを抽出
        model_markers = ["assistant\n", "assistant:\n", "assitant\n"]
        model_response = full_outputs[0]
        
        for marker in model_markers:
            if marker in model_response:
                model_response = model_response.split(marker)[-1].strip()
                break
        
        # 思考過程とプロンプトの抽出
        thinking = ""
        prompt_text = ""
        
        print(model_response)
        if "<think>" in model_response and "</think>" in model_response:
            thinking = model_response.split("<think>")[1].split("</think>")[0].strip()

        def clean_prompt_text(text):
            # 削除するタグのリスト
            tags_to_remove = [
                "masterpiece", "high score", "great score", "absurdres", 
                "highres", "original character", "original series", 
                "general", "sensitive", "nsfw", "explicit"
            ]
            
            # テキストを単語に分割して処理
            words = []
            current_words = text.split(',')
            
            # 各単語をトリムして処理
            for word in current_words:
                word = word.strip()
                # 空の単語はスキップ
                if not word:
                    continue
                # 削除対象のタグかチェック
                if any(tag == word.lower() for tag in tags_to_remove):
                    continue
                # まだ追加されていない単語のみ追加（重複排除）
                if word not in words:
                    words.append(word)
            
            # カンマで結合して返す
            return ', '.join(words)
        
        if "<prompt>" in model_response:
            if "</prompt>" in model_response:
                prompt_text = model_response.split("<prompt>")[1].split("</prompt>")[0].strip()
            else:
                prompt_text = model_response.split("<prompt>")[1].strip()

            prompt_text = clean_prompt_text(prompt_text)
        else:
            prompt_text = f"1girl, {character_name}, {series_name}, anime style, highres"

        prompt_text = prompt_text + f", {category}"
        
        log_prompt_io(novel_text, series_name, character_name, category, thinking, prompt_text)
        logger.info(f"Successfully generated prompt from text")
        return thinking, prompt_text
        
    except Exception as e:
        logger.error(f"Error generating prompt: {str(e)}")
        # エラー時のフォールバック
        return f"エラーが発生しました: {str(e)}", f"1girl, {character_name}, {series_name}, anime style, highres"
