## Tweet Ordering

This task is based on the sentence ordering task of Barzilay and Lapata (2008) to assess text relatedness. 
We construct the data by shuffling Twitter threads (containing 3–5 tweets), and assessing the predicted
ordering in terms of rank correlation (ρ) with the original.

The experiment is based on 5-fold cross validation. The splits are provided in `data/`, with distribution:
* Train: 4327 threads
* Development: 760 threads 
* Test: 1521 threads


## Running baselines

Please download indonesian fasttext embedding ([cc.id.300.vec](https://fasttext.cc/docs/en/crawl-vectors.html)).
#### BiLSTM
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
