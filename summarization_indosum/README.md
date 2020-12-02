## Summarization

IndoLEM uses [IndoSum](https://arxiv.org/abs/1810.05334) for extractive summarization.
Our experiment is based on [Liu and Lapata (2018)](https://arxiv.org/abs/1908.08345) framework with three BERT models: IndoBERT, malayBERT, and mBERT.

## Requirements
Tested with below configuration. Higher torch version is not suitable for [PreSumm](https://github.com/nlpyang/PreSumm).
```
python==3.7.6
torch==1.1.0
torchvision==0.8.1
transformers==3.0.0
pyrouge==0.1.3
tensorboardX==2.1
```

## Experiment

1. First, download the data [here](https://drive.google.com/file/d/1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco/view) and put the them (all folds) in folder `data/`
2. Original implementation can be found [here](https://github.com/nlpyang/PreSumm).
3. Run three scripts for data preprocessing:
```
python make_datafiles_presum_indobert.py
python make_datafiles_presum_malaybert.py
python make_datafiles_presum_mbert.py
```
3. Now you can run the experiment by using the script below:

IndoBERT
```
cd scripts
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
In `scripts/` run `chmod +x *` to enable bash execution. The training requires 3 GPUs (V100 16GB). If you have lower GPU size, please reduce the batch size.

## Evaluation

Please install [pyrouge](https://github.com/bheinzerling/pyrouge) for evaluating the summary.
