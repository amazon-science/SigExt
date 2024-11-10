import argparse
import json
import os
import pathlib
import re
from functools import partial

import jsonlines
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from datasets import load_dataset
from nltk.corpus import stopwords
from rake_nltk import Rake
from transformers import AutoTokenizer

DATASET_NAME_MAPPER = {
    "cnn": ("cnn_dailymail", "3.0.0"),
    "arxiv": ("scientific_papers", "arxiv"),
    "pubmed": ("scientific_papers", "pubmed"),
    "xsum": ("EdinburghNLP/xsum",),
    "multi_news": ("multi_news",),
    "gigaword": ("gigaword",),
    "big_patent": ("big_patent",),
    "aeslc": ("aeslc",),
    "billsum": ("billsum",),
    "wikihow": ("wikihow", "all"),
}

SKIP_TEXT_MARK = "......"

en_stopwords = set(list(stopwords.words("english")) + ["@cite", "@math"])


def just_phrase_extractor(text):
    tweet_tokenizer = nltk.tokenize.TweetTokenizer()
    rake = Rake(word_tokenizer=tweet_tokenizer.tokenize, stopwords=en_stopwords)
    all_phrases = rake._generate_phrases(rake._tokenize_text_to_sentences(text))

    offset = 0
    existing_phrases = set()
    text_trim = text

    phrase_list = []
    for phrase in all_phrases:
        pattern = r"(\s*)".join([re.escape(x) for x in phrase])
        try:
            span = re.search(pattern, text_trim.lower()).span(0)
        except:
            continue
        phrase_str = text_trim[span[0]:span[1]]

        if phrase_str.lower() not in existing_phrases:
            phrase_list.append({
                "phrase": phrase_str,
                "index": offset + span[0]
            })
            existing_phrases.add(phrase_str.lower())

        offset = offset + span[1]
        text_trim = text_trim[span[1]:]

    for item in phrase_list:
        assert text[item["index"]:item["index"] + len(item["phrase"])] == item["phrase"]

    return phrase_list


def normalized_special_mark_in_scientific_papers(text):
    return re.sub(r"@x([a-z]+)\d*", r"@\1", text)


def get_token_length(text, tokenizer):
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def concat_lines_and_remove_spaces(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def truncate_text(text, tokenizer, max_len):
    tokens = tokenizer(text, add_special_tokens=False).input_ids
    return tokenizer.decode(tokens[:max_len])


def scientific_papers_input_processor(example, tokenizer, max_input_len):
    raw_text = example["article"]
    text = normalized_special_mark_in_scientific_papers(raw_text)
    text = concat_lines_and_remove_spaces(text)

    trunc_text = truncate_text(text, tokenizer, max_len=max_input_len)

    length_info = {
        "input_raw_text": get_token_length(raw_text, tokenizer),
        "input_processed_text": get_token_length(text, tokenizer),
        "input_truncated_text": get_token_length(trunc_text, tokenizer),
    }
    return raw_text, trunc_text, length_info


def scientific_papers_output_processor(example, tokenizer, max_output_len):
    raw_text = example["abstract"]
    text = normalized_special_mark_in_scientific_papers(raw_text)
    text = concat_lines_and_remove_spaces(text)

    trunc_text = truncate_text(text, tokenizer, max_len=max_output_len)

    length_info = {
        "output_raw_text": get_token_length(raw_text, tokenizer),
        "output_processed_text": get_token_length(text, tokenizer),
        "output_truncated_text": get_token_length(trunc_text, tokenizer),
    }
    return raw_text, trunc_text, length_info


def general_input_processor(example, tokenizer, max_input_len, field):
    raw_text = example[field]
    trunc_text = truncate_text(raw_text, tokenizer, max_len=max_input_len)
    length_info = {
        "input_raw_text": get_token_length(raw_text, tokenizer),
        "input_truncated_text": get_token_length(trunc_text, tokenizer),
    }
    return raw_text, trunc_text, length_info


def general_output_processor(example, tokenizer, max_output_len, field):
    raw_text = example[field]
    trunc_text = truncate_text(raw_text, tokenizer, max_len=max_output_len)
    length_info = {
        "output_raw_text": get_token_length(raw_text, tokenizer),
        "output_truncated_text": get_token_length(trunc_text, tokenizer),
    }
    return raw_text, trunc_text, length_info


def plot_length_info(length_info, output_filename):
    length_info = pd.DataFrame(length_info)

    length_info = length_info.melt(var_name="text_type", value_name="num_tokens")

    plt.figure(figsize=(6, 4))
    plt.xlim(1, 30000)
    sns.histplot(
        length_info, x="num_tokens", hue="text_type", log_scale=True, element="step", fill=False, cumulative=True
    )
    plt.savefig(output_filename, bbox_inches="tight")


def process_split(data, num_sample, input_processor, output_processor, tokenizer, max_input_len, max_output_len):
    shuffled_idx = np.arange(len(data))
    np.random.shuffle(shuffled_idx)

    if num_sample == -1:
        num_sample = len(data)

    processed_data = []
    pbar = tqdm.tqdm(total=num_sample)

    for idx in shuffled_idx:
        item = dict(data[int(idx)])
        item["index"] = int(idx)
        raw_input, trunc_input, input_length_info = input_processor(item, tokenizer, max_input_len)
        raw_output, trunc_output, output_length_info = output_processor(item, tokenizer, max_output_len)

        if not trunc_input or not trunc_output:
            continue

        example = {
            "raw_input": raw_input,
            "raw_output": raw_output,
            "trunc_input": trunc_input,
            "trunc_output": trunc_output,
            "input_length_info": input_length_info,
            "output_length_info": output_length_info,
            "trunc_input_phrases": just_phrase_extractor(trunc_input),
            "trunc_output_phrases": just_phrase_extractor(trunc_output),
        }

        processed_data.append(example)
        pbar.update(1)
        if len(processed_data) == num_sample:
            break

    return processed_data


def create_dataset(
        dataset,
        tokenizer,
        output_dir,
        seed,
        sample_train,
        sample_val,
        sample_test,
        max_input_len,
        max_output_len,
        append_mode=False,
):
    np.random.seed(seed)
    dataset_name = dataset

    if dataset in ["arxiv", "pubmed"]:
        input_processor, output_processor = scientific_papers_input_processor, scientific_papers_output_processor
    elif dataset in ["cnn"]:
        input_processor, output_processor = (
            partial(general_input_processor, field="article"),
            partial(general_output_processor, field="highlights"),
        )
    elif dataset in ["xsum", "multi_news", "gigaword"]:
        input_processor, output_processor = (
            partial(general_input_processor, field="document"),
            partial(general_output_processor, field="summary"),
        )
    elif dataset in ["big_patent"]:
        input_processor, output_processor = (
            partial(general_input_processor, field="description"),
            partial(general_output_processor, field="abstract"),
        )
    elif dataset in ["aeslc"]:
        input_processor, output_processor = (
            partial(general_input_processor, field="email_body"),
            partial(general_output_processor, field="subject_line"),
        )
    elif dataset in ["billsum"]:
        input_processor, output_processor = (
            partial(general_input_processor, field="text"),
            partial(general_output_processor, field="summary"),
        )
    elif dataset in ["wikihow"]:
        input_processor, output_processor = (
            partial(general_input_processor, field="text"),
            partial(general_output_processor, field="headline"),
        )
    else:
        assert 0, dataset

    if dataset == "wikihow":
        dataset = load_dataset(*DATASET_NAME_MAPPER[dataset], data_dir="./wikihow_data/")
    else:
        dataset = load_dataset(*DATASET_NAME_MAPPER[dataset])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    for split_name, num_sample in [("validation", sample_val), ("train", sample_train), ("test", sample_test)]:
        if split_name not in dataset:
            continue
        processed_split = process_split(
            dataset[split_name],
            num_sample,
            input_processor=input_processor,
            output_processor=output_processor,
            tokenizer=tokenizer,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
        )

        with jsonlines.open(pathlib.Path(output_dir).joinpath(f"{split_name}.jsonl"), "a" if append_mode else "w") as f:
            f.write_all(processed_split)

        plot_length_info(
            [item["input_length_info"] for item in processed_split],
            str(pathlib.Path(output_dir).joinpath(f"{dataset_name}_{split_name}_input_length.png")),
        )
        plot_length_info(
            [item["output_length_info"] for item in processed_split],
            str(pathlib.Path(output_dir).joinpath(f"{dataset_name}_{split_name}_output_length.png")),
        )


def main():
    parser = argparse.ArgumentParser("Preprocess data and extract keywords.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_NAME_MAPPER.keys()) + ["all", "all_other"],
        help="Select from supported datasets.",
    )
    parser.add_argument(
        "--tokenizer",
        default="allenai/longformer-large-4096",
        type=str,
        help="Specify the HF tokenizer for truncation.",
    )
    parser.add_argument("--max_input_len", default=6000, type=int, help="Tokens to keep at beginning of article.")
    parser.add_argument("--max_output_len", default=510, type=int, help="Tokens to keep in output.")
    parser.add_argument("--output_dir", required=True, type=str, help="Output dir for the dataset.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed to sample dataset.")
    parser.add_argument("--sample_train", default=1000, type=int, help="Number of training examples to sample.")
    parser.add_argument("--sample_val", default=200, type=int, help="Number of validation examples to sample.")
    parser.add_argument("--sample_test", default=500, type=int, help="Number of test examples to sample.")

    args = vars(parser.parse_args())
    os.makedirs(args["output_dir"], exist_ok=True)
    with open(pathlib.Path(args["output_dir"]).joinpath("data_config.json"), "w") as f:
        json.dump(args, f)

    if args["dataset"] in ["all", "all_other"]:
        mode = args["dataset"]
        del args["dataset"]
        for dataset in DATASET_NAME_MAPPER.keys():
            if mode == "all_other" and dataset in ["cnn", "arxiv", "pubmed"]:
                continue
            create_dataset(dataset=dataset, **args, append_mode=True)
    else:
        create_dataset(**args, append_mode=False)


if __name__ == "__main__":
    main()
