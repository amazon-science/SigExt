## SigExt: Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization

This is the implementation of the EMNLP'24 paper.

Title: [Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization](https://www.amazon.science/publications/salient-information-prompting-to-steer-content-in-prompt-based-abstractive-summarization)

Authors: [Lei Xu](https://leixx.io/), 
[Asad Karim](https://www.amazon.science/author/asad-karim), 
[Saket Dingliwal](https://www.amazon.science/author/saket-dingliwal), 
[Aparna Elangovan](https://scholar.google.com/citations?user=eaow7uAAAAAJ&hl=en)

## Introduction
Large language models (LLMs) are highly effective at generating summaries across various domains through prompting 
techniques, reducing the need for dedicated training in summarization applications. However, designing prompts that 
guide LLMs to generate summaries with an appropriate level of detail and a coherent writing style can be challenging. 
Keyphrase Signal Extractor (SigExt) addresses this by leveraging salient information directly from the source document 
to improve summarization prompts. By integrating extracted keyphrases, SigExt enhances ROUGE F1 and recall, making 
generated summaries more aligned with reference texts and more complete. Additionally, the number of keyphrases 
provides a precision-recall trade-off, allowing for tailored summarization outputs. 

## Run Experiments
Here is an example on running SigExt on CNN dataset.

```
# Prepare datasets in jsonl format
python3 src/prepare_data.py --dataset cnn --output_dir experiments/cnn_dataset/

# Train the longformer keyphrase extractor
python3 src/train_longformer_extractor_context.py \
  --dataset_dir experiments/cnn_dataset/ \
  --checkpoint_dir experiments/cnn_extractor_model/

# Inference the longformer keyphrase extractor
python3 src/inference_longformer_extractor.py \
  --dataset_dir experiments/cnn_dataset/ \
  --checkpoint_dir experiments/cnn_extractor_model/ \
  --output_dir experiments/cnn_dataset_with_keyphrase/

# Run summarization
python3 src/zs_summarization.py \
  --model_name claude \
  --kw_strategy sigext_topk \
  --kw_model_top_k 15 \
  --dataset cnn \
  --dataset_dir experiments/cnn_dataset_with_keyphrase/ \
  --output_dir experiments/cnn_extsig_predictions/
```

## Citation
```text
@inproceedings{xu2024salient,
  title={Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization},
  author={Xu, Lei and Karim, Mohammed Asad and Dingliwal, Saket and Elangovan, Aparna},
  booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track",
  year={2024}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

