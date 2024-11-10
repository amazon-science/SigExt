import argparse
import copy
import glob
import logging
import pathlib

import jsonlines
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from train_longformer_extractor_context import KWDatasetContext, KeywordExtractorClf


def find_best_checkpoint(checkpoint_dir):
    candidates = sorted(glob.glob(f"{checkpoint_dir}/lightning_logs/*/checkpoints/*.ckpt"))

    if not len(candidates):
        raise RuntimeError("Candidates not found.")

    best_score = 0
    best_checkpoint = None
    for item in candidates:
        tokens = pathlib.Path(item).stem.split("-")
        info = {}
        for tok in tokens:
            key, value = tok.split("_")
            info[key] = value

        if float(info["recall20"]) >= best_score:
            best_checkpoint = item
            best_score = float(info["recall20"])
    return best_checkpoint


def parse_result(raw_dataset, predicts):
    raw_dataset = copy.deepcopy(raw_dataset)
    for pred_info in predicts:
        example_info = raw_dataset[pred_info["id"]]
        score = pred_info["score"]
        if len(score) > len(example_info["trunc_input_phrases"]):
            raise RuntimeError("model prediction does not match with number of phrases.")
        example_info["input_kw_model"] = score

    return raw_dataset


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser("Run longformer inference")
    parser.add_argument("--dataset_dir", required=True, type=str, help="directory of train and validation data.")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="directory of checkpoints.")
    parser.add_argument("--output_dir", required=True, type=str, help="directory to save predictions.")
    parser.add_argument(
        "--base_model", default="allenai/longformer-base-4096", type=str, help="Backbone pre-trained model."
    )

    args = parser.parse_args()

    best_checkpoint = find_best_checkpoint(str(pathlib.Path(args.checkpoint_dir).expanduser()))
    logging.info(f"Using f{best_checkpoint}")
    model = KeywordExtractorClf.load_from_checkpoint(best_checkpoint, base_model="allenai/longformer-base-4096")

    trainer = pl.Trainer(devices=1, accelerator="gpu")

    dataset_dir = pathlib.Path(args.dataset_dir).expanduser()
    output_dir = pathlib.Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_class = KWDatasetContext

    for split in ["validation", "test"]:
        dataset = dataset_class(
            dataset_filename=str(dataset_dir.joinpath(f"{split}.jsonl")),
            base_model=args.base_model,
            example_kw_hit_threshold=0,
            hide_gt=True,
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        predicts = trainer.predict(model, dataloaders=dataloader)

        dataset_with_prediction = parse_result(dataset.raw_dataset, predicts)

        with jsonlines.open(str(output_dir.joinpath(f"{split}.jsonl")), "w") as f:
            f.write_all(dataset_with_prediction)


if __name__ == "__main__":
    main()
