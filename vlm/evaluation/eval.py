import argparse
import re
from scipy import stats
from sklearn.metrics import f1_score
from scipy.stats import ranksums
import numpy as np
from tqdm import tqdm
import pandas as pd
import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
two_dirs_up = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
sys.path.append(two_dirs_up)
from vlm.inference.local_paths import ANNOTATION_PATH



np.seterr(divide='ignore', invalid='ignore')


tqdm.pandas()

LANGUAGES = ["de", "en", "es", "hi", "zh"]
MAPPING = {
    "en": "US",
    "de": "DE",
    "es": "MX",
    "hi": "IN",
    "zh": "CN"
}


def significance_test(row):
    n1 = 6
    n2 = 6
    min_max_value = [value for key, value in row.items() if "textbf" in str(
        value[0]) or "underline" in str(value[0]) or "phantom" in str(value[0])]
    for index, value in enumerate(min_max_value):
        match = re.search(r'\\(?:textbf|underline){([\d.]*)}', value[0])
        if match:
            min_max_value[index] = [float(match.group(1)), value[1], value[3]]
        if "phantom" in value[0]:
            value[0] = value[0].replace("\\phantom{}", "")
            min_max_value[index] = [float(value[0]), value[1], value[3]]

    min_max_value = sorted(min_max_value, key=lambda x: x[0])
    if min_max_value and len(min_max_value) == 3:
        # T Test
        # _, p_value = stats.ttest_ind_from_stats(min_max_value[0][0], min_max_value[0][1], n1, min_max_value[1][0], min_max_value[1][1], n2, equal_var=False)

        # Wilcoxon Rank Sum
        p_value = stat_test(min_max_value[0][2], min_max_value[2][2])
        alpha = 0.05
        if p_value < alpha:
            for key, value in row.items():
                if "textbf" in str(value[0]):
                    p_value = stat_test(
                        min_max_value[1][2], min_max_value[2][2])
                    if p_value < alpha:
                        row[key] = [value[0] + "$^{\mathbf{**}}", value[1]]
                    else:
                        row[key] = [value[0] + "$^{\mathbf{*}}", value[1]]

            return row
        else:
            return row
    else:
        return row


def extract_llm_answer(response):
    response = response.lower()
    prefixes = [
        "assistant: ", "assistant:", "assistant\n", "assistant", "the answer is likely ",
        "answer choice:", "answer:", "antwort:", "mi respuesta es:",
        "respuesta:", "respondo:", "选择：", "选项:", "答案:",
        "答:", "选择 ", "答案：", "选择选项:", "the answer is ", "nur ein buchstabe als antwort erwünscht:",
        "respondo con una sola letra del alfabeto:", "por lo que la respuesta correcta sería:", "die antwort ist "
    ]

    for prefix in prefixes:
        if prefix in response:
            response = response.split(prefix)[-1]
    return response


def extract_answer(response):
    response = extract_llm_answer(response)
    response = response.lower()
    response = response.strip()

    b_responses = ["b: hate", "b: non-hate", "answer: b"]
    a_responses = ["a: non-hate", "a: hate", "a) hate", "answer: a"]
    if any(s in response for s in b_responses):
        return "b"
    if any(s in response for s in a_responses):
        return "a"

    response = response.split(" ")[0].split("\n")[0].strip()
    return response


def check_response(response_llm, response, extracted):
    non_hate_phrases = [
        "this meme is non-hate", "would rate this meme as non-hate", "non-hate (b)",
        "non-hate.", "no", "option 'a'", "a) non-hate", "not hate.", "not hate speech!",
        "no, this is not hate speech", "no, the meme is not hate speech", "non-hate",
        "non-hate:", "गैर नफरत", "sin odio", "नहीं\n\n", "不属于仇恨言论", "不一定是仇恨言论",
        "答案：b：非仇恨", "- क (नॉन-द्वेष)", "नफरत नहीं", "गैर-नफरत", "a：非仇恨", "非仇恨", "n. \n\nthis", "n (this meme",
        "这个表情包中的内容并未表现出任何攻击或侮辱的情感", "答案：b，非仇恨", "b: 非仇恨", "b. 非仇恨", "a：非仇恨", "b：非仇恨", "b：非仇恨",
        "n (this is a complex", "n - without further context", "n - it's ", "n (as in"
    ]

    nonhate_phrases_equal = [
        "n \n", "n", "n.", "n\n", "n.\n", "n. \n"
    ]

    hate_phrases = [
        "this meme contains hate speech", "yes, i would consider this meme as hate speech",
        "hate", "yes, I would consider this meme as hate speech", "h8", "घृणा", "घृणा",
        "enthält eine hassrede", "enthält eine form von hassre", "a: odio", "नफरत",
        "इस मीम का उपयोग अपमानजनक", "नफ़रत", "जी हां, यह मीम घृणित भाषण है", "b：仇恨", "b: 仇恨",
        "this falls under attacking", "gendered insult often used to demean and belittle someone",
        "meme targets and ridicules a specific group", "y (it's impossible",
        "contains potentially offensive language and imagery",
        "clearly targets groups based", "y (as in",
        "meme uses dismissive and mocking language", "indicates the language targets a group",
        "clearly targets a specific nationality or ethnicity",
        "meme promotes harmful stereotypes and fear-mongering",
        "meme relies on a harmful stereotype to make a joke",
        "interpreted as perpetuating harmful stereotypes",
        'appear to be targeting an individual based on appearance, which could be considered an "identity factor"',
        "this could be considered discriminatory language",
        "could be considered a form of body shaming", "j\n\n"
    ]

    hate_phrases_equal = [
        "yes",
        "hate",
        "h8", "घृणा", "घृणा", "hass",  "odio", "odio (a)", "odio (b)",
        "y \n", "y", "y\n", "j \n", "j", "j\n", "j.", "j.\n", "j. ", "j. \n"

    ]
    unable_to_analyze_phrases = [
        "i'm sorry but this task involve", "i cannot analyze", "i'd need to have the meme text translated",
        "i am unable to interpret the text", "i am unable to provide the analysis",
        "choose your answer by selecting either 'a' or 'b'", "i am unable to assist",
        "i'm unable to analyze this meme", "c: unrelated", "i cannot make that determination.",
        "i can't provide assistance",
        "please provide the meme",
        "das meme enthält eine frage, die", "meme enthält eine aussage",
        "neues jahr!!!", "h.", "मैं नहीं जानता।",
        "it is impossible to determine the nature of the meme",
        "it's impossible to give a definitive answer",
        "it's impossible to accurately assess the meme",
        "it's impossible to determine with certainty",
        "impossible to determine the intent of the meme",
        "impossible to determine without further context",
        "impossible to determine without translation",
        "meaning is unclear without further context",
        "impossible to determine the intent of the meme without more context",
        "impossible to determine the nature"
    ]
    unable_to_analyze_phrases_equal = [
        "o", "१६", "घ", "घ्रोणी\n\nअत्याचार", "c"
    ]

    # Check for non-hate phrases
    if any(phrase in response_llm for phrase in non_hate_phrases) or any(phrase == response_llm for phrase in nonhate_phrases_equal):
        if "b: Non-hate" in response or "ख: गैर नफरत" in response or "b: Kein Hass" in response or "b: Sin odio" in response:
            return 1
        elif "a: Non-hate" in response or "क: गैर नफरत" in response or "a: Kein Hass" in response or "a: Sin odio" in response:
            return 0

    # Check for hate phrases
    if any(phrase in response_llm for phrase in hate_phrases) or any(phrase == response_llm for phrase in hate_phrases_equal):
        if "b: Hate" in response or "ख: घृणा" in response or "b: Hass" in response or "b: Odio" in response:
            return 1
        elif "a: Hate" in response or "क: घृणा" in response or "a: Hass" in response or "a: Odio" in response:
            return 0

    # Check for unable to analyze phrases
    if any(phrase in response_llm for phrase in unable_to_analyze_phrases) or any(phrase == response_llm for phrase in unable_to_analyze_phrases_equal):
        print(
            f"1: Invalid response encountered: {response} \n Extracted LLM: ---{response_llm}---\n Extracted: ----{extracted}----")
        return -1

    return -2


def mapping_response(response):
    extracted = extract_answer(response)

    a_response = ["a", "a:", "a: hate", "a.", "a:", "a)", "a (**Non-hate**)", "'a'", "-a-",
                  "क", "a：仇恨", "अ", "antwort: 'a'", "क:", "क.", "**a**", "答案: a", "**a**.",
                  "a：非仇恨", "a<|eot_id|>", "का", "答案：a", "答案：a：非仇恨", "回答：a", "答案是：a",
                  "答：a：非仇恨", "回答：a：非仇恨", "a：", "a**", "(a)", "'a'."]
    b_response = ["'b'", "b", "b:", "b:", "b)", "b:", "b.", "b: hate", "b: non-hate", "(b)", "b：hate", "*b",
                  "ख", "b：非仇恨", "ब", "-b:", "ख:", "答案：b", "b：仇恨", "ख.", "回答：b", "回答：b：非仇恨",
                  "b：", "答：b", "选项b：仇恨", "b:非仇恨", "**b**", "**b**.", "答案是b：仇恨", "b:<sep>",
                  "b，非仇恨", "b**", "b.**"]

    invalid_response = ["bustin", "busters.kerry", "a.kerry",
                        "busters.", "bhindiwereview", "bhaving",
                        "bheye.", "africaarabi.", "bilder.", "banner",
                        "banner", "bhaving", "bust", "bubble",
                        "b.19999999", "bietehrt.", "a120.", "ademon", "b20.",
                        "the", "h", "an", "g", "d", "e", "aa", "ग", "n", "keine informationen.",
                        "दोनों में नहीं", "जी-3", "इतिमान से भरा हुआ हो सकता है."]

    # Some more logic:

    response_llm = extract_llm_answer(response)
    complex_logic = check_response(response_llm, response, extracted)
    if complex_logic != -2:
        return complex_logic

    valid_responses_b = ["choice: b", "meme: b", "选项: b", "答案: b"]

    if extracted in a_response or response_llm == "answer (a)":
        return 0
    elif extracted in b_response or response_llm in valid_responses_b:
        return 1
    elif extracted in invalid_response:
        print(
            f"Invalid response encountered: ---{response}--- \n Extracted LLM: ---{response_llm}---\n Extracted: ----{extracted}----")
        return -1
    else:
        print(
            f"Invalid response encountered: ---{response}--- \n Extracted LLM: ---{response_llm}---\n Extracted: ----{extracted}----")
        return -1


def process_response_to_hatespeech(row):
    # Invalid:
    if row['processed_answer'] == -1:
        return -1

    # if response = 0 -> a. if response = 1 -> b.
    if row['prompt'] % 2 != 0:
        # b = hate
        if row['processed_answer'] == 1:
            return 1
        elif row['processed_answer'] == 0:
            return 0
    else:
        # a = hate
        if row['processed_answer'] == 1:
            return 0
        elif row['processed_answer'] == 0:
            return 1


# Function to calculate accuracy with dynamic column names
def calculate_group_accuracy(df, group_col, col1, col2):
    df_group = df.groupby(group_col)
    accuracy = df_group[[col1, col2]].apply(lambda x: (
        x[col1] == x[col2]).mean()).reset_index(name='accuracy')
    return accuracy


def stat_test(df1, df2):
    sample1 = np.array(df1["accuracy"])
    sample2 = np.array(df2["accuracy"])
    p_value = round(ranksums(sample1, sample2).pvalue, 5)
    return p_value


def calc_acc(df, gt_name, predict_name):
    accuracy_by_group = calculate_group_accuracy(
        df, "prompt", gt_name, predict_name)

    mean_accuracy = round(accuracy_by_group['accuracy'].mean() * 100, 2)
    std_accuracy = round(
        np.std(np.array(accuracy_by_group['accuracy']), ddof=1) * 100, 2)

    return mean_accuracy, std_accuracy, accuracy_by_group


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run pipeline inference with specified model path.')

    # Add an argument for MODEL_PATH
    parser.add_argument('--model_predictions', type=str, required=False,
                        default='vlm/results/models--llava-hf--llava-onevision-qwen2-72b-ov-hf')
    args = parser.parse_args()
    model_predictions_folder = args.model_predictions
    df_gt = pd.read_csv(ANNOTATION_PATH)
    df_gt = df_gt.reset_index()
    df_gt["Meme ID"] = df_gt["Meme ID"].astype(str)

    latex_preds = {}
    # Loop over all folders inside the parent folder
    for root, dirs, files in os.walk(model_predictions_folder):
        for file in files:
            for language in LANGUAGES:
                print("----Language: {}-----".format(language))
                folder_path = os.path.join(root, file)
                if os.path.exists(folder_path):
                    df_inference = pd.read_csv(folder_path)
                else:
                    continue
                df_inference.rename(columns={'ID': 'Meme ID'}, inplace=True)
                df_inference['answer'] = df_inference['response'].apply(
                    extract_answer)
                df_inference['processed_answer'] = df_inference['response'].apply(
                    mapping_response)
                df_inference['hate_prediction'] = df_inference.apply(
                    process_response_to_hatespeech, axis=1)
                df_inference['response'] = df_inference["response"].str.replace(
                    "\n", "").str.replace("assistant", "").str[-20:]
                df_inference = df_inference[[
                    "Meme ID", "prompt", "response", "answer", "hate_prediction"]]
                output_path = os.path.join(
                    root, "processed_responses_" + language + ".csv")
                df_inference.to_csv(output_path, index=False)

                # Evaluation
                df_inference["Meme ID"] = df_inference["Meme ID"].astype(str)
                df_inference = pd.merge(df_inference, df_gt, on="Meme ID")
                # N Invalid Responses
                n_invalid = sum(df_inference["hate_prediction"] == -1)
                # Accuracy
                for language_eval in LANGUAGES:
                    print(f"------EVAL LANG {language_eval}------")
                    accuracy, std, df_acc = calc_acc(
                        df_inference, MAPPING[language_eval], "hate_prediction")
                    print("Accuracy: {}\nStandard Deviation: {}".format(
                        accuracy, std))
