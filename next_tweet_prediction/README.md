## Next Tweet Prediction

To evaluate model coherence, we design a next tweet prediction (NTP) task
that is similar to the next sentence prediction (NSP) task used to train BERT (Devlin et al., 2019). In
NTP, each instance consists of a Twitter thread (2–4 tweets) that we call the premise, and four possible
options for the next tweet, one of which is the actual response from the original thread.

The experiment is not based on 5-fold cross validation as we ensure that there is no overlap between the next tweet
candidates in the training and test sets.

* Train: 5681 threads
* Development: 811 threads 
* Test: 1890 threads


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
