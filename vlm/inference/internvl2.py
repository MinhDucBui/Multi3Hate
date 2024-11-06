import argparse
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import torchvision.transforms as T
import torch
import math
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)
from vlm.inference.utils import pipeline_inference, create_prompt_for_input

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LANGUAGES = ["en", "de", "es", "hi", "zh"]


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def input_creator(all_prompts, image_paths, model_path, df_captions, add_caption, unimodal):
    # Input for model_inference()
    processor = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=40, do_sample=False)
    generation_config["pad_token_id"] = processor.pad_token_id

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
                conversation = text_prompt_1["text"] + \
                    "<image>" + text_prompt_2["text"]
                pixel_values = None
            else:
                conversation = text_prompt_1["text"] + \
                    "<image>" + text_prompt_2["text"]
                pixel_values = load_image(
                    image_path, max_num=12).to(torch.bfloat16).cuda()
            processed_prompts.append(
                {"prompt": [conversation, pixel_values, generation_config]})

    return processor, processed_prompts


def model_creator(model_path):
    device_map = split_model('InternVL2-Llama3-76B')
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map).eval()
    return model


def model_inference(prompt, model, processor, unimodal):
    response_text = model.chat(processor, prompt[1], prompt[0], prompt[2])
    response_text = prompt[0] + "Assistant: " + response_text
    return response_text


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run pipeline inference with specified model path.')

    # Add an argument for MODEL_PATH
    parser.add_argument('--model_path', type=str, required=False,
                        default="models/models--OpenGVLab--InternVL2-Llama3-76B/snapshots/e05e8226c7ddd6e6e48cca5357e6c528f4d16cef")
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
