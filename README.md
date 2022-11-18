# MMTr

### [Improving multimodal fusion with Main Modal Transformer for emotion recognition in conversation](https://www.sciencedirect.com/science/article/pii/S0950705122010711)

## Required Packages:

* pytorch==1.11.0
* transformers==4.14.1
* numpy
* pickle
* tqdm
* sklearn

## Run on GPU:

Model runs on one GPU by default, and we didn't try it on CPU.

> We recommend using GPU with memory more than 24G , otherwise you may need to adjust the hyperparameters and the results may vary significantly.

## Quick Start

To run the model on test sets of two datasets

1.  for IEMOCAP dataset

   train.py    --dataset IEMOCAP

2. for MELD dataset

   train.py    --dataset MELD

## Citation

If you find this work useful, please cite our work:

```
@article{ZOU2022109978,
title = {Improving multimodal fusion with Main Modal Transformer for emotion recognition in conversation},
journal = {Knowledge-Based Systems},
volume = {258},
pages = {109978},
year = {2022},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2022.109978},
url = {https://www.sciencedirect.com/science/article/pii/S0950705122010711},
author = {ShiHao Zou and Xianying Huang and XuDong Shen and Hankai Liu}
}
```

## Acknowledgement

Some code of this project are referenced from [MMGCN](https://github.com/hujingwen6666/MMGCN).
We thank their open source materials for contribution to this task.

