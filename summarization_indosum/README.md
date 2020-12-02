## Summarization

IndoLEM uses [IndoSum](https://arxiv.org/abs/1810.05334) for extractive summarization.
Our experiment is based on [Liu and Lapata (2018)](https://arxiv.org/abs/1908.08345) framework with three BERT models: IndoBERT, malayBERT, and mBERT.

## Experiment


1. First, download the data [here](https://drive.google.com/file/d/1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco/view) and put the them (all folds) in folder `data/`
3. For code dependencies, please take a look to the original implementation of [PreSumm](https://github.com/nlpyang/PreSumm).
2. Run three scripts for data preprocessing:
```
python make_datafiles_presum_indobert.py
python make_datafiles_presum_malaybert.py
python make_datafiles_presum_mbert.py
```
3. Now you can run the experiment by using the script below:

IndoBERT
```
cd scripts
chmod +x *
./train_indobert.sh
./eval_indobert.sh
```
MalayBERT
```
cd scripts
./train_malaybert.sh
./eval_malaybert.sh
```
mBERT
```
cd scripts
./train_mbert.sh
./eval_mbert.sh
```

## Evaluation

Please install [pyrouge](https://github.com/bheinzerling/pyrouge) for evaluating the summary.
