# Latent Thought Models with Variational Bayes Inference-Time Computation
Deqian Kong*, Minglu Zhao*, Dehong Xu*, Bo Pang, Shu Wang, Edouardo Honig, Zhangzhang Si, Chuan Li, Jianwen Xie^, Sirui Xie^, Ying Nian Wu^

*Equal contribution, ^Equal advising

[[Paper link](https://arxiv.org/abs/2502.01567)], [[Project Blog](https://deqiankong.github.io/blogs/ltm/)]

**ICML 2025**

## Overview
We propose a novel class of language models, Latent Thought Models (LTMs), which incorporate explicit latent thought vectors that follow an explicit prior model in latent space. These latent thought vectors guide the autoregressive generation of ground tokens through a Transformer decoder. Training employs a dual-rate optimization process within the classical variational Bayes framework: fast learning of local variational parameters for the posterior distribution of latent vectors (inference-time computation), and slow learning of global decoder parameters. 

## Installation

```bash
git clone [address]
cd Latent-Thought-LM
conda env create -f env.yml
conda activate ltm
```

## Training

```bash
python train_ltm.py
```
