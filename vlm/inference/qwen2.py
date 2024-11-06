import argparse
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)
from vlm.inference.utils import pipeline_inference, create_prompt_for_input


LANGUAGES = ["en", "de", "es", "hi", "zh"]


def input_creator(all_prompts, image_paths, model_path, df_captions, add_caption, unimodal):
    # Input for model_inference()
    processor = AutoProcessor.from_pretrained(model_path)
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
                    "meme", "text")
                text_prompt_2["text"] = text_prompt_2["text"].replace(
                    "meme", "text")
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
                        {"type": "image", "image": image_path},
                        text_prompt_2
                    ],
                },
                ]

            processed_prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True)
            processed_prompts.append(
                {"prompt": [conversation, processed_prompt]})

    return processor, processed_prompts


def model_creator(model_path):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    # model = None
    return model


def model_inference(prompt, model, processor, unimodal):
    if unimodal:
        image_inputs = None
        video_inputs = None
    else:
        image_inputs, video_inputs = process_vision_info(prompt[0])
    inputs = processor(
        text=[prompt[1]],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    output = model.generate(**inputs, max_new_tokens=40,
                            do_sample=False, top_k=None)
    response_text = processor.decode(output[0][2:], skip_special_tokens=True)
    return response_text


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run pipeline inference with specified model path.')

    # Add an argument for MODEL_PATH
    parser.add_argument('--model_path', type=str, required=False,
                        default='models/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/3ca981c995b0ce691d85d8408216da11ff92f690')
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
