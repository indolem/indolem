# Dependency Parsing

Our data is from [Universal Dependencies](https://universaldependencies.org/).
#### 1. UD-Indo-PUD (5 folds)
1,000 sentences of UD-Indo-PUD ([Zeman et al., 2018](https://www.aclweb.org/anthology/K17-3001/)), and we use the corrected version by  [Alfina et al. (2019)](https://github.com/ialfina/revised-id-pud)
#### 2. UD-Indo-GSD (1 fold)
5,593 sentences of UD-Indo-GSD ([McDonald et al., 2013](https://www.aclweb.org/anthology/P13-2017/))

## Experiment

We modify [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2) and use [_Deep Biaffine Attention for Neural Dependency Parsing_](https://arxiv.org/abs/1611.01734). By Timothy Dozat, Christopher D. Manning. In ICLR 2017.

#### Requirements for NeuroNLP2
Tested with:
```
python==3.6
torch==1.3.1
gensim==0.12.0
overrides==3.1.0
conllu==4.2.1
scipy==1.1.0
transformers==3.1.0
```
#### Experiment with BERT embedding

For each *UD-Indo-PUD* and *UD-Indo-GSD*, you can run jupyter notebook `extract.ipynb` to extract BERT embedding for Indonesian, Malaysian, and Multilingual BERT.
This will take a while, especially for *UD-Indo-GSD*. A folder `data/` will be created in each repository *UD-Indo-PUD* and *UD-Indo-GSD.*

You may skip the extraction step by downloading them [here](https://drive.google.com/drive/folders/1dG2nxtvxRbzKLsFTSvwlxrZBvK71mtti?usp=sharing).

Now you are ready to run the experiment. We have provided script file in `experiments/scripts/`

For IndoBERT:
```
./experiments/scripts/run_deepbiaf_indobert0.sh
./experiments/scripts/run_deepbiaf_indobert1.sh
./experiments/scripts/run_deepbiaf_indobert2.sh
./experiments/scripts/run_deepbiaf_indobert3.sh
./experiments/scripts/run_deepbiaf_indobert4.sh
./experiments/scripts/run_deepbiaf_indobertGSD.sh
```
For MalayBERT:
```
./experiments/scripts/run_deepbiaf_malaybert0.sh
./experiments/scripts/run_deepbiaf_malaybert1.sh
./experiments/scripts/run_deepbiaf_malaybert2.sh
./experiments/scripts/run_deepbiaf_malaybert3.sh
./experiments/scripts/run_deepbiaf_malaybert4.sh
./experiments/scripts/run_deepbiaf_malaybertGSD.sh
```
For mBERT:
```
./experiments/scripts/run_deepbiaf_mbert0.sh
./experiments/scripts/run_deepbiaf_mbert1.sh
./experiments/scripts/run_deepbiaf_mbert2.sh
./experiments/scripts/run_deepbiaf_mbert3.sh
./experiments/scripts/run_deepbiaf_mbert4.sh
./experiments/scripts/run_deepbiaf_mbertGSD.sh
```
