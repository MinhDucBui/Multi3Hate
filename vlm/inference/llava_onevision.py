from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import torch
from PIL import Image
import sys
import os
import argparse

current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)
from vlm.inference.utils import pipeline_inference, create_prompt_for_input


LANGUAGES = ["en", "de", "es", "hi", "zh"]
# LANGUAGES = ["en"]


def input_creator(all_prompts, image_paths, model_path, df_captions, add_caption, unimodal):
    # Input for model_inference()
    processor = AutoProcessor.from_pretrained(model_path)

    processor.patch_size = 14
    processor.vision_feature_select_strategy = "default"

    processed_prompts = []
    for image_path in image_paths:
        for raw_prompt in all_prompts:
            text_prompt_1, text_prompt_2 = create_prompt_for_input(
                raw_prompt, df_captions, image_path, add_caption)

            if unimodal:
                text_prompt_1["text"] = text_prompt_1["text"][:-7]
                text_prompt_2["text"] = text_prompt_2["text"].replace(
                    "Caption inside the meme:", "Text:")
                text_prompt_1["text"] = text_prompt_1["text"].replace(
                    "Answer if this meme", "Answer based solely on this text")
                text_prompt_2["text"] = text_prompt_1["text"].replace(
                    "Given a meme from online", "Given only this text from online")
                text_prompt_2["text"] = text_prompt_1["text"].replace(
                    "Would you consider this meme", "Would you consider only this text")
                conversation = [{
                    "role": "user",
                    "content": [
                        text_prompt_1,
                        text_prompt_2,
                    ],
                },
                ]
            else:
                conversation = [{
                    "role": "user",
                    "content": [
                        text_prompt_1,
                        {"type": "image"},
                        text_prompt_2
                    ],
                },
                ]
            processed_prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True)
            processed_prompts.append(
                {"prompt": processed_prompt, "image_path": image_path})

    return processor, processed_prompts


def model_creator(model_path):
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return model


def model_inference(image_path, prompt, model, processor, unimodal):
    # model.generation_config["pad_token_id"] = processor.pad_token_id
    raw_image = Image.open(image_path)
    if unimodal:
        raw_image = None
    inputs = processor(images=raw_image,
                       text=prompt,
                       return_tensors='pt').to('cuda', torch.float16)
    output = model.generate(**inputs, max_new_tokens=40,
                            do_sample=False, temperature=0.0)
    response_text = processor.decode(output[0][2:], skip_special_tokens=True)
    return response_text


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run pipeline inference with specified model path.')

    # Add an argument for MODEL_PATH
    parser.add_argument('--model_path', type=str, required=False,
                        default='models/sync/models--llava-hf--llava-onevision-qwen2-72b-ov-hf/snapshots/7f872aec22af34da2b31b2b3efb6a6403a5bb6c7')
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
