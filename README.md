# 🎬 Sentiment Analysis on IMDB Movie Reviews

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vishal-7003/sentiment-analysis-nlp/blob/main/sentiment_analysis_distilBERT.ipynb)

A complete NLP pipeline comparing classical ML against a fine-tuned Transformer for binary sentiment classification.

## Results

| Model | Accuracy | F1 Score | Training Time |
|-------|----------|----------|---------------|
| TF-IDF + Logistic Regression | 89.46% | 89.46% | ~30.6 seconds |
| DistilBERT (fine-tuned) | **91.32%** | **91.32%** | ~10 minutes |

---

## Overview

This project builds an end-to-end sentiment analysis system on the [IMDB dataset](https://huggingface.co/datasets/imdb) (50,000 movie reviews), benchmarking two fundamentally different NLP paradigms:

- **Classical ML Baseline** — TF-IDF vectorization + Logistic Regression
- **Transformer Model** — DistilBERT (`distilbert-base-uncased`) fine-tuned for sequence classification

The goal was to understand *when* the added complexity of a Transformer is worth it, and *why* contextual embeddings outperform bag-of-words approaches on this task.

---

## Dataset

- **Source:** IMDB Movie Reviews (via HuggingFace `datasets`)
- **Size:** 50,000 reviews — 25k train / 25k test
- **Classes:** Positive / Negative (perfectly balanced — 12,500 each per split)
- **Median review length:** ~230 words; 29% exceed 256 tokens

---

## Approach

### 1. TF-IDF + Logistic Regression (Baseline)

- Bigram TF-IDF with `max_features=50,000`
- Captures phrases like *"not good"* and *"highly recommended"*
- Trains in **30.6 seconds** on CPU
- Achieves **89.46% accuracy** — a surprisingly strong baseline

### 2. DistilBERT Fine-Tuning

- `distilbert-base-uncased` — 66M parameters, 6 Transformer encoder blocks
- WordPiece tokenization, `max_length=256`, `[CLS]` token for classification
- Fine-tuned for 3 epochs on T4 GPU with:
  - Learning rate: `2e-5` (AdamW)
  - Warmup steps: 200
  - Weight decay: 0.01
  - Mixed precision (`fp16=True`)
- Achieves **91.32% accuracy** — ~1.86% gain over baseline

---

## Key Findings

- A TF-IDF baseline gets you 89.5% — sentiment is largely predictable from word choice alone
- DistilBERT gains ~1.8% by understanding **context, negation, and word relationships**
- Validation loss rose after epoch 1, suggesting mild overfitting — early stopping would help
- **Error analysis** reveals the model struggles most with:
  - Sarcasm (`"Oh great, another predictable ending"`)
  - Mixed sentiment (positive story, negative acting)
  - Ambiguous reviews with hedged language
- Truncating at 256 tokens costs only ~0.5% accuracy since sentiment signal is front-loaded in reviews

---

## Project Structure

```
sentiment-analysis-nlp/
│
├── sentiment_analysis_distilBERT.ipynb   # Main notebook (baseline + fine-tuning + evaluation)
└── README.md
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Classical ML | Scikit-learn |
| Data | HuggingFace Datasets |
| Visualization | Matplotlib, Seaborn |
| Hardware | Google Colab (Tesla T4 GPU) |

---

## How to Run

### Option 1 — Google Colab (Recommended)
Open the notebook directly in Colab. All dependencies install via `!pip install` in the first cell.

### Option 2 — Local

```bash
git clone https://github.com/Vishal-7003/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
pip install transformers datasets scikit-learn matplotlib seaborn torch
jupyter notebook sentiment_analysis_distilBERT.ipynb
```

> **Note:** Fine-tuning DistilBERT requires a GPU. For CPU-only environments, skip to Section 4 (baseline) or load the saved model directly.

---

## Tradeoffs & When to Use Which

| Scenario | Recommended Model |
|----------|------------------|
| Real-time inference on CPU | TF-IDF + Logistic Regression |
| Accuracy-critical production system | DistilBERT |
| Low-resource / edge deployment | TF-IDF + Logistic Regression |
| Data with heavy negation / sarcasm | DistilBERT |

---

## Future Improvements

- [ ] Try `roberta-base` — expected ~94% accuracy
- [ ] Add early stopping to prevent overfitting
- [ ] SHAP explainability to visualize word-level attribution
- [ ] Deploy on HuggingFace Spaces for live inference demo

---

## About

Built as part of my ML portfolio while pursuing MCA (AI/ML specialization) at Chandigarh University.

**Connect:** [LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/Vishal-7003)
