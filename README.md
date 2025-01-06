# Multi3Hate: Multimodal, Multilingual, and Multicultural Hate Speech Detection with Vision–Language Models

## We published Multi3Hate on Huggingface! Check it out [here](https://huggingface.co/datasets/MinhDucBui/Multi3Hate).


**Multi3Hate** dataset repository, introduced in our paper:

> **Multi3Hate: Advancing Multimodal, Multilingual, and Multicultural Hate Speech Detection with Vision–Language Models**

This repository contains all the resources and code used in the paper.

## 🗂️ Dataset Structure
The dataset is organized in the `data/` folder:

- **Images**: `data/memes/` - Meme images categorized by language in subfolders.
- **Annotations**:
  - `data/final_annotations.csv` - Aggregated annotations.
  - `data/raw_annotations.csv` - Annotations by individual annotators.

## 🚀 Running VLM Inference

### 1. Install Dependencies
To get started, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Model Inference
Use the scripts in `vlm/inference/` to run inference with Vision-Language Models (VLMs). Below are commands for each available model:

```bash
python vlm/inference/llava_onevision.py
python vlm/inference/internvl2.py
python vlm/inference/qwen2.py
python vlm/inference/gpt_4o.py
python vlm/inference/gemini_pro.py
```

> **Important Notes**:
> - For closed-source models, provide a valid API key.
> - Ensure you have the correct version of `transformers` installed for `internvl`:
>   ```bash
>   pip install transformers==4.37.2
>   ```

## 📈 Model Evaluation

To evaluate model predictions, use this command, replacing `<folder>` with the path to your model's prediction folder:

```bash
python vlm/evaluation/eval --model_predictions <folder>
```
