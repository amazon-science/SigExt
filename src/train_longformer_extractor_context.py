import argparse
import logging
import pathlib

import jsonlines
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import tqdm
import transformers
from rouge_score import rouge_scorer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer


class KWDatasetContext(Dataset):
    def __init__(
            self,
            dataset_filename,
            base_model,
            example_kw_hit_threshold=3,
            base_model_max_length=4096,
            hide_gt=False,
    ):
        super().__init__()
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

        if hide_gt:
            assert example_kw_hit_threshold == 0

        with jsonlines.open(dataset_filename) as f:
            self.raw_dataset = list(f)

        pos_cc = 0
        neg_cc = 0
        for idx, item in tqdm.tqdm(enumerate(self.raw_dataset), total=len(self.raw_dataset),
                                   desc="process data"):
            x = [0]
            y = [-100]
            example_kw_hit_cc = 0

            selected_keyword_strs = set()
            if not hide_gt:
                for kw in item["trunc_input_phrases"]:
                    best_rouge_f = 0
                    best_rouge_p = 0
                    best_rouge_r = 0
                    for ent in item["trunc_output_phrases"]:
                        score = scorer.score(target=ent["phrase"], prediction=kw["phrase"])
                        best_rouge_f = max(best_rouge_f, score["rouge1"].fmeasure)
                        best_rouge_p = max(best_rouge_p, score["rouge1"].precision)
                        best_rouge_r = max(best_rouge_r, score["rouge1"].recall)
                    if best_rouge_f >= 0.6 or best_rouge_p >= 0.8 or best_rouge_r >= 0.8:
                        selected_keyword_strs.add(kw["phrase"])

            current_text_index = 0
            for kw_info in item["trunc_input_phrases"]:
                if current_text_index < kw_info["index"]:
                    content = item["trunc_input"][current_text_index: kw_info["index"]]
                    content_tokens = self.tokenizer(content)["input_ids"][1:-1]
                    x.extend(content_tokens)
                    y.extend([-100] * len(content_tokens))
                    assert len(content_tokens)
                else:
                    if current_text_index != 0:
                        content_tokens = self.tokenizer(" ")["input_ids"][1:-1]
                        x.extend(content_tokens)
                        y.extend([-100] * len(content_tokens))
                        assert len(content_tokens)

                format_kw = f"{kw_info['phrase']}"
                input_ids = self.tokenizer(format_kw)["input_ids"][1:-1]

                if not hide_gt and kw_info["phrase"] in selected_keyword_strs:
                    labels = [1] * len(input_ids)
                else:
                    labels = [0] * len(input_ids)

                if labels[-1] == 1:
                    example_kw_hit_cc += 1

                assert y[-1] == -100
                x.extend(input_ids)
                y.extend(labels)
                assert len(labels)
                current_text_index = kw_info["index"] + len(kw_info["phrase"])

            if current_text_index < len(item["trunc_input"]):
                content = item["trunc_input"][current_text_index:]
                content_tokens = self.tokenizer(content)["input_ids"][1:-1]
                x.extend(content_tokens)
                y.extend([-100] * len(content_tokens))

            x = x[: base_model_max_length - 1]
            y = y[: base_model_max_length - 1]
            x.extend([2])
            y.extend([-100])

            x = torch.tensor(x).long()
            y = torch.tensor(y).long()

            if example_kw_hit_cc >= example_kw_hit_threshold:
                self.data.append((x, y, idx))
                pos_cc += torch.sum(y == 1).numpy()
                neg_cc += torch.sum(y == 0).numpy()

        logging.info(f"keyword ratio {pos_cc / (pos_cc + neg_cc)}")
        logging.info(f"Dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class KeywordExtractorClf(pl.LightningModule):
    def __init__(self, base_model):
        super().__init__()
        self.clf = AutoModelForTokenClassification.from_pretrained(base_model, num_labels=2)
        self.validation_step_outputs = []

    def predict_step(self, batch, batch_idx):
        x, y, idx = batch  # here y is just an indicator of whether it is the end of a phrase.
        assert x.size(0) == 1
        logits = F.log_softmax(self.clf(x)[0], dim=-1)

        kw_count = 0
        score_and_label = []

        current_logits = 0
        current_len = 0
        current_in_phrase = False

        for i in range(x.size(1)):
            if y[0][i] != -100:
                if not current_in_phrase:
                    current_in_phrase = True
                current_logits += logits[0][i][1]
                current_len += 1
            else:
                if current_in_phrase:
                    current_in_phrase = False
                    score_and_label.append({"kw_index": kw_count, "score": float(current_logits / current_len)})
                    kw_count += 1
                    current_logits = 0
                    current_len = 0

        return {"id": int(idx[0]), "score": score_and_label}

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        assert x.size(0) == 1
        logits = self.clf(x)[0]
        loss = F.cross_entropy(logits[0], y[0], reduction="sum", weight=torch.Tensor([0.1, 1]).float().to(self.device))
        self.log("train/loss", loss)
        self.log("train/std", logits[0, :, 1].std())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        assert x.size(0) == 1
        logits = self.clf(x)[0]
        loss = (
            F.cross_entropy(logits[0], y[0], reduction="sum", weight=torch.Tensor([0.1, 1]).float().to(self.device))
            .cpu()
            .numpy()
        )
        logits = F.log_softmax(logits, dim=-1)

        n = x.size(1)

        score_and_label = []

        logits = logits.cpu().numpy()
        y = y.cpu().numpy()

        current_logits = 0
        current_len = 0
        current_in_phrase = False

        for i in range(x.size(1)):
            if y[0][i] != -100:
                if not current_in_phrase:
                    current_in_phrase = True
                current_logits += logits[0][i][1]
                current_len += 1
            else:
                if current_in_phrase:
                    current_in_phrase = False
                    score_and_label.append(
                        {"index": i, "logits": float(current_logits / current_len), "label": y[0][i - 1]}
                    )
                    current_logits = 0
                    current_len = 0

        score_and_label = sorted(score_and_label, key=lambda item: (item["logits"], -item["index"]), reverse=True)

        step_output = {
            "loss": loss,
            "logits_std": np.std([item["logits"] for item in score_and_label]),
            "logits_std_all": np.std(logits[0, :, 1]),
        }

        for top_k in [5, 10, 20]:
            step_output[f"precision_{top_k}"] = np.mean([item["label"] for item in score_and_label[:top_k]])
            step_output[f"recall_{top_k}"] = np.sum([item["label"] for item in score_and_label[:top_k]]) / np.sum(
                [item["label"] for item in score_and_label]
            )
        self.validation_step_outputs.append(step_output)

    def on_validation_epoch_end(self):
        for top_k in [5, 10, 20]:
            self.log(
                f"val/precision_{top_k}",
                float(np.mean([item[f"precision_{top_k}"] for item in self.validation_step_outputs])),
                sync_dist=True,
            )
            self.log(
                f"val/recall_{top_k}",
                float(np.mean([item[f"recall_{top_k}"] for item in self.validation_step_outputs])),
                sync_dist=True,
            )
        self.log("val/loss", float(np.mean([item["loss"] for item in self.validation_step_outputs])), sync_dist=True)
        self.log(
            "val/logits_std",
            float(np.mean([item["logits_std"] for item in self.validation_step_outputs])),
            sync_dist=True,
        )
        self.log(
            "val/logits_std_all",
            float(np.mean([item["logits_std_all"] for item in self.validation_step_outputs])),
            sync_dist=True,
        )

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.estimated_stepping_batches, num_warmup_steps=100
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser("Train longformer keyword extractor.")
    parser.add_argument("--dataset_dir", required=True, type=str, help="directory of train and validation data.")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="directory to save checkpoints.")
    parser.add_argument("--seed", default=42, type=int, help="random seed for trainer.")
    parser.add_argument(
        "--base_model", default="allenai/longformer-base-4096", type=str, help="Backbone pre-trained model."
    )
    parser.add_argument("--load_ckpt", default=None, type=str, help="Pretrained ckpt.")

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    dataset_dir = pathlib.Path(args.dataset_dir).expanduser()
    train_set = KWDatasetContext(
        dataset_filename=str(dataset_dir.joinpath("train.jsonl")),
        base_model=args.base_model,
        example_kw_hit_threshold=1,
    )
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_set = KWDatasetContext(
        dataset_filename=str(dataset_dir.joinpath("validation.jsonl")), base_model=args.base_model
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

    if args.load_ckpt:
        model = KeywordExtractorClf.load_from_checkpoint(args.load_ckpt, base_model=args.base_model)
    else:
        model = KeywordExtractorClf(base_model=args.base_model)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=2,
        monitor="val/recall_20",
        mode="max",
        every_n_epochs=1,
        filename="epoch_{epoch:02d}-step_{step:06d}-recall20_{val/recall_20:.3f}",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        default_root_dir=args.checkpoint_dir,
        strategy="ddp_find_unused_parameters_true",
        log_every_n_steps=1,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
