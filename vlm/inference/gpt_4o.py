import base64
from openai import OpenAI
import sys
import os
import argparse

current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)
from vlm.inference.utils import pipeline_inference


LANGUAGES = ["en", "de", "es", "hi", "zh"]
# LANGUAGES = ["de"]
API_KEY = "<Your API Key>"


def input_creator(all_prompts, image_paths, model_path, df_captions, add_caption, unimodal):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Input for model_inference()
    processor = None
    processed_prompts = []
    for image_path in image_paths:
        base64_image = encode_image(image_path)
        for index, raw_prompt in enumerate(all_prompts):
            id_image = image_path.split("/")[-1].split(".jpg")[0]
            prompt_1 = raw_prompt[0]
            prompt_2 = raw_prompt[1]
            if add_caption:
                id_image = image_path.split("/")[-1].split(".jpg")[0]
                caption = df_captions[df_captions["ID"]
                                      == id_image]["Translation"].iloc[0]
                text_prompt_1 = {"type": "text", "text": prompt_1.format(str(caption))}
                text_prompt_2 = {"type": "text", "text": prompt_2.format(str(caption))}
            else:
                text_prompt_1 = {"type": "text", "text": prompt_1}
                text_prompt_2 = {"type": "text", "text": prompt_2}

            if unimodal:
                text_prompt_1["text"] = text_prompt_1["text"][:-7]
                messages = [
                    {
                        "role": "user",
                        "content": [
                            text_prompt_1,
                            text_prompt_2,
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            text_prompt_1,
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                            text_prompt_2,
                        ],
                    }
                ]
            processed_prompts.append({"prompt": [messages, id_image + "_" + str(index)]})

    return processor, processed_prompts


def model_creator(model_path):
    api_key = API_KEY
    client = OpenAI(api_key=api_key)
    return client


def model_inference(prompt, model, processor, unimodal):
    response = model.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=prompt,
        max_tokens=40,
        temperature=0,
    )
    response_text = prompt[0]["content"][0]["text"] + "\nAssistant:" + response.choices[0].message.content

    return response_text


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Run pipeline inference with specified model path.')

    # Add an argument for MODEL_PATH
    parser.add_argument('--model_path', type=str, required=False, default='gpt_4o/dont/matter')
    parser.add_argument('--caption', action='store_true', help='Enable captioning')
    parser.add_argument('--multilingual', action='store_true', help='Enable captioning')
    parser.add_argument('--country_insertion', action='store_true', help='Enable captioning')
    parser.add_argument('--unimodal', action='store_true', help='Enable captioning')
    args = parser.parse_args()

    pipeline_inference(args.model_path, LANGUAGES, input_creator, model_creator, model_inference, 
                       add_caption=args.caption, multilingual=args.multilingual, country_insertion=args.country_insertion,
                       unimodal=args.unimodal)