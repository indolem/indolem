## Sentiment Analysis

This dataset is based on binary classification (positive and negative), with distribution:
* Train: 3638 sentences
* Development: 399 sentences
* Test: 1011 sentences

The data is sourced from 1) Twitter [(Koto and Rahmaningtyas, 2017)](https://www.researchgate.net/publication/321757985_InSet_Lexicon_Evaluation_of_a_Word_List_for_Indonesian_Sentiment_Analysis_in_Microblogs)
and 2) [hotel reviews](https://github.com/annisanurulazhar/absa-playground/).

The experiment is based on 5-fold cross validation. The splits are provided in `data/`
## Running baselines
#### Naive Bayes and Logistic Regression:
```
python baseline_NB_LR.py
```
#### BiLSTM
Please download indonesian fasttext embedding ([cc.id.300.vec](https://fasttext.cc/docs/en/crawl-vectors.html)).
```
CUDA_VISIBLE_DEVICES=0 python baseline_bilstm.py --data_path data/ --fasttext_path path_to_fast_text_embedding
```

## Running BERT-based model
#### IndoBERT
```
CUDA_VISIBLE_DEVICES=0 python bert.py --data_path data/ --lang id
```
#### MalayBERT
```
CUDA_VISIBLE_DEVICES=0 python bert.py --data_path data/ --lang my
```
#### mBERT
```
CUDA_VISIBLE_DEVICES=0 python bert.py --data_path data/ --lang multi
```

Please refer to the code, if you want to adjust another parameters.
