import argparse
import time
import PIL.Image
import google.generativeai as genai
import base64
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)
from vlm.inference.utils import pipeline_inference


LANGUAGES = ["en", "de", "es", "hi", "zh"]
API_KEY_GEMINI = "<Your API Key>"
SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


def input_creator(all_prompts, image_paths, model_path, df_captions, add_caption, unimodal):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Input for model_inference()
    processor = None
    processed_prompts = []
    for image_path in image_paths:
        for raw_prompt in all_prompts:
            prompt_1 = raw_prompt[0]
            prompt_2 = raw_prompt[1]
            if add_caption:
                id_image = image_path.split("/")[-1].split(".jpg")[0]
                caption = df_captions[df_captions["ID"]
                                      == id_image]["Translation"].iloc[0]
                text_prompt_1 = {"type": "text",
                                 "text": prompt_1.format(str(caption))}
                text_prompt_2 = {"type": "text",
                                 "text": prompt_2.format(str(caption))}
            else:
                text_prompt_1 = {"type": "text", "text": prompt_1}
                text_prompt_2 = {"type": "text", "text": prompt_2}

            image_pil = PIL.Image.open(image_path)

            if unimodal:
                text_prompt_1["text"] = text_prompt_1["text"][:-7]
                text_prompt_2["text"] = text_prompt_2["text"].replace(
                    "Caption inside the meme:", "Text:")
                text_prompt_1["text"] = text_prompt_1["text"].replace(
                    "meme", "text")
                text_prompt_2["text"] = text_prompt_2["text"].replace(
                    "meme", "text")
                processed_prompts.append(
                    {"prompt": [text_prompt_1["text"], text_prompt_2["text"]]})
            else:
                processed_prompts.append(
                    {"prompt": [text_prompt_1["text"], image_pil, text_prompt_2["text"]]})

    return processor, processed_prompts


def model_creator(model_path):
    # Model Configuration

    model_config = genai.GenerationConfig(
        max_output_tokens=40,
        temperature=0.0,
    )
    genai.configure(api_key=API_KEY_GEMINI)
    model = genai.GenerativeModel(
        "gemini-1.5-pro-001", generation_config=model_config)
    return model


def model_inference(prompt, model, processor, unimodal):
    time.sleep(0.2)

    response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
    try:
        # Code that might raise the AttributeError
        response_text = prompt[0] + prompt[-1] + \
            "\nAssistant: " + response.text
    except AttributeError as e:
        # Handle the AttributeError here
        print(f"An AttributeError occurred: {e}")
        response_text = prompt[0] + prompt[-1] + "\nAssistant: " + str(e)

    return response_text


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run pipeline inference with specified model path.')

    # Add an argument for MODEL_PATH
    parser.add_argument('--model_path', type=str,
                        required=False, default='gemini_pro/dont/matter')
    parser.add_argument('--caption', action='store_true',
                        help='Enable captioning')
    parser.add_argument('--multilingual', action='store_true',
                        help='Enable captioning')
    parser.add_argument('--country_insertion',
                        action='store_true', help='Enable captioning')
    parser.add_argument('--unimodal', action='store_true',
                        help='Enable captioning')
    args = parser.parse_args()

    pipeline_inference(args.model_path, LANGUAGES, input_creator, model_creator, model_inference,
                       add_caption=args.caption, multilingual=args.multilingual, country_insertion=args.country_insertion,
                       unimodal=args.unimodal)
