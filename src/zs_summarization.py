import argparse
import json
import logging
import os
import pathlib
from collections import defaultdict
from multiprocessing import Pool

import jsonlines
import nltk
import numpy as np
import tqdm
from nltk.tokenize import word_tokenize
from rapidfuzz import fuzz
from rouge_score import rouge_scorer

from bedrock_utils import predict_one_eg_mistral, predict_one_eg_claude_instant
from prompts import (
    ZS_NAIVE_PROMPT_STR_FOR_MISTRAL,
    ZS_NAIVE_PROMPT_STR_FOR_CLAUDE,
    ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL,
    ZS_KEYWORD_PROMPT_STR_FOR_CLAUDE,
)

ZS_NAIVE_PROMPT_STR = {
    "mixtral": ZS_NAIVE_PROMPT_STR_FOR_MISTRAL,
    "claude": ZS_NAIVE_PROMPT_STR_FOR_CLAUDE,
}


ZS_KEYWORD_PROMPT_STR = {
    "mistral": ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL,
    "claude": ZS_KEYWORD_PROMPT_STR_FOR_CLAUDE,
}


def estimate_logits_threshold(dataset_file, percentile_threshold):
    if not os.path.exists(dataset_file):
        logging.warning("validation set not found for logits threshold.")
        return -1

    with jsonlines.open(dataset_file) as f:
        data = list(f)

    if "input_kw_model" not in data[0]:
        logging.warning("input_kw_model not found in the file. Use -1 as threshold.")
        return -1

    logits = []

    for item in data:
        for kw_info_model in item["input_kw_model"]:
            logits.append(kw_info_model["score"])
    return np.percentile(logits, percentile_threshold)


class NaivePrompt(object):
    def __init__(self, model_name, dataset_name, customized_prompt=None):
        self.prompt = customized_prompt or ZS_NAIVE_PROMPT_STR[model_name][dataset_name]

    def __call__(self, example):
        return self.prompt.replace("<text>", example["trunc_input"])


def remove_duplicate_top_k(candidates, top_k, threshold=70):
    ret = []

    for candidate in candidates:
        to_delete = set()
        to_skip = False

        if len(ret) >= top_k:
            break

        for added_kw in ret:
            if fuzz.ratio(added_kw["phrase"].lower(), candidate["phrase"].lower()) >= threshold:
                if len(added_kw["phrase"]) <= len(candidate["phrase"]):
                    to_delete.add(added_kw["phrase"])
                else:
                    to_skip = True

        ret = [item for item in ret if item["phrase"] not in to_delete]

        if not to_skip:
            ret.append(candidate)

    return ret


class SegExtTopK(object):
    def __init__(
        self,
        model_name,
        dataset_name,
        top_k,
        deduplicate=True,
        logits_threshold=-1,
        use_rank=False,
        customized_prompt=None,
    ):
        self.prompt = customized_prompt or ZS_KEYWORD_PROMPT_STR[model_name][dataset_name]
        self.top_k = top_k
        self.deduplicate = deduplicate
        self.logits_threshold = logits_threshold
        self.use_rank = use_rank

    def __call__(self, example):
        if self.use_rank:
            selected_keywords = sorted(example["trunc_input_phrases"], key=lambda x: x["rank"])
            for i in range(len(selected_keywords)):
                selected_keywords[i]["score"] = i
        else:
            selected_keywords = []
            for kw_info in sorted(example["input_kw_model"], key=lambda x: x["score"], reverse=True):
                if kw_info["score"] < self.logits_threshold:
                    break
                selected_keywords.append(example["trunc_input_phrases"][kw_info["kw_index"]])
                selected_keywords[-1]["score"] = kw_info["score"]

        if self.deduplicate:
            selected_keywords = remove_duplicate_top_k(selected_keywords, top_k=self.top_k)
        else:
            selected_keywords = selected_keywords[: self.top_k]

        formatted_keywords = "; ".join([item["phrase"] for item in selected_keywords]) + "."
        return self.prompt.replace("<text>", example["trunc_input"]).replace("<keywords>", formatted_keywords)


def get_prompt_fn(model_name, dataset, kw_strategy, kw_model_top_k, logits_threshold):
    if kw_strategy == "disable":
        return NaivePrompt(model_name, dataset)
    elif kw_strategy == "sigext_topk":
        return SegExtTopK(model_name, dataset, top_k=kw_model_top_k, logits_threshold=logits_threshold)
    else:
        raise RuntimeError("unknown kw strategy.")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_rouge_score(inference_data, preds):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)

    labels = [item["raw_output"] for item in inference_data]

    decoded_preds, decoded_labels = postprocess_text(preds, labels)

    result_element = defaultdict(list)
    for pred, label in zip(decoded_preds, decoded_labels):
        score = scorer.score(target=label, prediction=pred)
        for metric_name, value in score.items():
            result_element[f"{metric_name}p"].append(value.precision)
            result_element[f"{metric_name}r"].append(value.recall)
            result_element[f"{metric_name}f"].append(value.fmeasure)

    result = {}
    for metric_name, values in result_element.items():
        result[metric_name] = np.mean(values)

    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [len(word_tokenize(pred)) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def run_inference(model_name, kw_strategy, kw_model_top_k, dataset, dataset_dir, output_dir, inference_on_split="test"):
    dataset_dir = pathlib.Path(dataset_dir)
    logits_threshold = estimate_logits_threshold(str(dataset_dir.joinpath("validation.jsonl")), 75)
    logging.info(f"logits threshold is {logits_threshold}")

    if model_name == "mistral":
        predict_one_eg_fn = predict_one_eg_mistral
    elif model_name == "claude":
        predict_one_eg_fn = predict_one_eg_claude_instant
    else:
        raise ValueError(f"invalid model name {model_name}")

    prompting_fn = get_prompt_fn(
        model_name,
        dataset,
        kw_strategy,
        kw_model_top_k=kw_model_top_k,
        logits_threshold=logits_threshold,
    )
    assert not isinstance(prompting_fn, dict)
    dataset_dir = pathlib.Path(dataset_dir)

    dataset_filename = str(dataset_dir.joinpath(f"{inference_on_split}.jsonl"))
    with jsonlines.open(dataset_filename) as f:
        inference_data = list(f)

    with Pool(4) as p:
        all_prompt = list(tqdm.tqdm(p.imap(prompting_fn, inference_data), total=len(inference_data)))

    for i in range(len(inference_data)):
        if isinstance(all_prompt[i], str):
            inference_data[i]["prompt_input"] = all_prompt[i]
        else:
            inference_data[i]["prompt_input"] = all_prompt[i][0]
            inference_data[i]["other_info"] = all_prompt[i][1]

    with Pool(4) as p:
        all_res = list(tqdm.tqdm(p.imap(predict_one_eg_fn, inference_data), total=len(inference_data)))
    all_res = [item for item in all_res]

    output_path = str(pathlib.Path(output_dir).expanduser())
    os.makedirs(output_path, exist_ok=True)

    with jsonlines.open(output_path + f"/{inference_on_split}_dataset.jsonl", "w") as f:
        f.write_all(inference_data)

    with open(output_path + f"/{inference_on_split}_predictions.json", "w") as f:
        json.dump(all_res, f, indent=2)

    test_metrics = compute_rouge_score(inference_data, all_res)
    with open(str(pathlib.Path(output_dir).joinpath(f"{inference_on_split}_metrics.json")), "w") as f:
        json.dump(test_metrics, f, indent=2)


def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("transformers.generation").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="mistral", choices=["claude", "mistral"], help="llm name")
    parser.add_argument("--kw_strategy", choices=["disable", "sigext_topk"], help="keyword strategy.")
    parser.add_argument("--kw_model_top_k", default=20, type=int, help="keyword strategy.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["arxiv", "pubmed", "cnn", "samsum", "meetingbank"],
        help="Select from supported datasets.",
    )
    parser.add_argument("--dataset_dir", required=True, type=str, help="directory of train and validation data.")
    parser.add_argument("--output_dir", required=True, type=str, help="directory to save experiment.")
    parser.add_argument("--inference_on_split", default="test", type=str, help="split_to_run_inference")

    args = parser.parse_args()

    run_inference(**vars(args))


if __name__ == "__main__":
    main()
