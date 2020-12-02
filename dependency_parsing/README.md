# Dependency Parsing

## Data

#### 1. UD-Indo-PUD 
1,000 sentences of UD-Indo-PUD (Zeman et al., 2018), and we use the corrected version by  [Alfina et al. (2019)](https://github.com/ialfina/revised-id-pud)
#### 2. UD-Indo-GSD
5,593 sentences of UD-Indo-GSD (McDonald et al., 2013)

## Experiment

We modify [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2) and use Stanford Deep BiAffine Parser [_Deep Biaffine Attention for Neural Dependency Parsing_](https://arxiv.org/abs/1611.01734). By Timothy Dozat, Christopher D. Manning. In ICLR 2017.

## Requirements for NeuroNLP2

Python 3.6, PyTorch >=1.3.1, Gensim >= 0.12.0

