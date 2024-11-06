
# Dataset and Model Inference Guide

### Dataset Structure
- The dataset is located in the `data/` folder:
  - `data/memes/` contains meme images, organized into subfolders by language.
  - `data/final_annotations.csv` holds the aggregated annotations.
  - `data/raw_annotations.csv` contains detailed annotations for each annotator.

### Running VLM Inference
1. **Install Dependencies**  
   First, install the required packages by running:

   ```bash
   pip install -r requirements.txt
   ```

2. **Model Inference**  
   To run inference with Vision-Language Models (VLMs), use the scripts provided in `vlm/inference/`. Here are the commands for each model:

   ```bash
   python vlm/inference/llava_onevision.py
   python vlm/inference/internvl2.py
   python vlm/inference/qwen2.py
   python vlm/inference/gpt_4o.py
   python vlm/inference/gemini_pro.py
   ```

   **Note:**
   - For closed-source models, you will need to provide an API key.
   - For `internvl`, ensure you have the correct version of the `transformers` library installed:

     ```bash
     pip install transformers==4.37.2
     ```

### Model Evaluation
To evaluate model predictions, run the following command, replacing `<folder>` with the path to your model predictions:

```bash
python vlm/evaluation/eval --model_predictions <folder>
```
