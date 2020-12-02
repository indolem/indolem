# :indonesia: Welcome to IndoLEM and IndoBERT! 👋 

## Paper
Fajri Koto, Afshin, Rahimi, Jey Han Lau, and Timothy Baldwin. [_IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP_](https://www.aclweb.org/anthology/2020.coling-main.66.pdf). 
In Proceedings of the 28th COLING, December 2020.

## 1. About IndoBERT

[IndoBERT](https://huggingface.co/indolem/indobert-base-uncased) is the Indonesian version of BERT model. We train the model using over 220M words, aggregated from three main sources: 
* Indonesian Wikipedia (74M words)
* news articles from Kompas, Tempo (Tala et al., 2003), and Liputan6 (55M words in total)
* an Indonesian Web Corpus (Medved and Suchomel, 2017) (90M words).

We trained the model for 2.4M steps (180 epochs) with the <b>final perplexity over the development set being 3.97 (similar to English BERT-base)</b>.

### How to use

Load model and tokenizer (tested with transformers==3.5.1)
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
```

## 2. About IndoLEM

IndoLEM (“Indonesian Language Evaluation Montage”) is a comprehensive Indonesian benchmark that comprises of seven tasks for the Indonesian language.
This benchmark is categorized into three pillars of NLP tasks: morpho-syntax, semantics, and discourse. 

We provide README file for each task. To find further information regarding each task, please click the related repository.

Experimental result over IndoLEM using mBERT, malayBERT and our IndoBERT:

| Task | Metric | Bi-LSTM | mBERT | MalayBERT | IndoBERT |
| ---- | ---- | ---- | ---- | ---- | ---- |
| POS Tagging | Acc | 95.4 | <b>96.8</b> | <b>96.8</b> | <b>96.8</b> |
| NER UGM | F1| 70.9 | 71.6 | 73.2 | <b>74.9</b> |
| NER UI | F1 | 82.2 | 82.2 | 87.4 | <b>90.1</b> |
| Dep. Parsing (GSD) | UAS/LAS | 85.25/80.35 | 86.85/81.78 | 86.99/81.87 | <b>87.12<b/>/<b>82.32</b> |
| Dep. Parsing (PUD) | UAS/LAS | 84.04/79.01 | <b>90.58</b>/<b>85.44</b> | 88.91/83.56 | 89.23/83.95 |
| Sentiment Analysis | F1 | 71.62 | 76.58 | 82.02 | <b>84.13</b> |
| IndoSum | R1/RL | 67.96/67.24 | 68.40/67.67 | 68.44/67.71 | <b>69.93</b>/<b>69.21</b> |
| Liputan6 (Sum) | R1/RL | 36.10/33.56 | 39.81/37.02 | --/-- | <b>41.08</b>/<b>38.01</b> |
| Next Tweet Prediction | Acc | 73.6 | 92.4 | 93.1 | <b>93.7</b> |
| Tweet Ordering | Corr (ρ) | 0.45 | 0.53 | 0.51 | <b>0.59</b> |
