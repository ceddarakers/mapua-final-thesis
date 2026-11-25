# Fine-Tuning of Sentiment Analysis in Taglish E-Commerce Reviews

BS Computer Science thesis (CSS200-1 Thesis 2), School of Information Technology, Mapúa University.

**Authors:** Bautista, Candado & Monteon

## Overview

This study fine-tunes Gemma-2 models to classify Taglish (Tagalog-English) Shopee product reviews as **Positive**, **Neutral**, or **Negative**. The revised experiment compares **Gemma-2-2B QLoRA** against **Gemma-2-9B QLoRA**, with 4-bit NF4 quantization and class-weighted loss.

Each review is labeled from its star rating and a rule-based text labeler built from two experts' criteria. The text label is used when a rule fires; otherwise the star label is used.

## Dataset

[Shopee Reviews (Tagalog) dataset](https://huggingface.co/datasets/scaredmeow/shopee-reviews-tl-stars), using its predefined 70/15/15 train / validation / test split.

## Method

- Text cleaning with PII masking (Tagalog affixes like *nag-download* are kept)
- Expert-criteria labeling (star label + criteria label → final label)
- QLoRA fine-tuning: rank 16, alpha 32, attention projections, 3 epochs
- 3 seeds per model (42, 1337, 2024)
- Zero-shot baselines for both models

## Evaluation

- Accuracy, Macro F1, Weighted F1 (mean ± SD over seeds)
- Per-class F1, precision, and recall
- Challenger Set (244 hard Taglish reviews)
- McNemar's test with Bonferroni correction (2B vs 9B)

## Files

| File | Description |
|---|---|
| `FINAL_THESIS_PROJECT.ipynb` | Full notebook: data, labeling, training, evaluation, figures, and Gradio demo |
| `final_thesis_project.py` | Python export of the notebook |

## Running

Open the notebook in Google Colab with an **A100 GPU**. Add your Hugging Face token as a Colab secret named `HF_TOKEN`, and accept the Gemma-2 license on Hugging Face. Outputs are saved to Google Drive.
