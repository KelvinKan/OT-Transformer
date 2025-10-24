## OT-Transformer (Text Generation Experiments)

This repository contains the text generation implementation of the paper:

> **"Optimal Control for Transformer Architectures: Enhancing Generalization, Robustness and Efficiency"**  
> Kelvin Kan, Xingjian Li, Benjamin J. Zhang, Tuhin Sahai, Stanley Osher, and Markos A. Katsoulakis, 
> 
> *Neural Information Processing Systems (NeurIPS), 2025*  
> [Paper Link](https://arxiv.org/pdf/2505.13499?)

The implementation is directly modified from [NanoGPT](https://github.com/karpathy/nanoGPT)

## How to easily turn your standard Transformer into OT-Transformer (pseudocode)

This plug-and-play modification requires minimal changes to a standard Transformer.

```
Require: Input x, standard Transformer model f, 
regularization parameter lmbda, terminal time T, number of integration steps M

def OT_Transformer(x, f, lmbda, T, M):
  # Initialize: 
  reg = 0 
  dt = T/M
  
  # OT-Transformer numerical integration
  for _ in range(M):
    velocity = f(x)
    x = x + dt*velocity
    reg = reg + dt * (torch.norm(velocity) ** 2 / (torch.numel(velocity))) # normalize by num. elements
    
  # Return: OT-Transformer output x, regularization term lmbda*reg
  return x, lmbda*reg
```


## Commands for the experiments

To run the text generation experiments from our paper, use the following commands.

- To run the NanoGPT experiments on Shakespeare (character-level) with a GPU, 
  - use the command for a standard Transformer
```sh
python train.py config/train_shakespeare_char.py --cts=False --lmbda=0.0 --random_seed=1
```

  - use the command for OT-Transformer (reduced model size)
```sh
python train.py config/train_shakespeare_char.py --cts=True --steps=5 --lmbda=1.0 --n_layer=5 --n_head=5 --n_embd=320 --random_seed=1
```

- To run the GPT-2 experiments on OpenWebText with 4 GPUs,
  - use the command for a standard Transformer
```sh
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py --cts=False --lmbda=0.0 --random_seed=1
```

  - use the command for OT-Transformer
```sh
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py --cts=True --steps=10 --lmbda=0.1 --random_seed=1
```

## Evaluation

To evaluate the trained nanoGPT models, use the command
```sh
python evaluate.py config/train_shakespeare_char.py
```

To evaluate the trained GPT-2 models, use the command
```sh
python evaluate_gpt2.py config/train_gpt2.py
```

## Hyperparameter

`cts = True or False`, specify whether the transformer is continuous-time or not

`steps` = number of time steps in forward Euler integration (for continuous-time only)

`lmbda` = regularization hyperparameter (for continuous-time only)

`random_seed` sets random seed for the experiment. We used random_seed = 1, 2, 3 for the three random trials on NanoGPT

`lnf_on = True or False` This control whether to apply layer normalization to the final output:

```sh
for block in self.transformer.h:


    x = block(x)


x = self.transformer.ln_f(x)

```

`ln1_on = True or False` `ln2_on = True or False` These two controls whether to apply layer normalization to the transformer blocks:

```sh
x = x + self.attn(self.ln_1(x))


x = x + self.mlp(self.ln_2(x))
```


## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm torchtext bert-score nltk sacremoses rouge-score
```

Dependencies:

- [pytorch](https://pytorch.org) <3 (version=2.1.0)
- [numpy](https://numpy.org/install/) <3 (version=1.26.4)
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3
-  `torchtext` for computing Bleu score (version=0.16.0)
-  `bert-score` for computing bert score
-  `nltk` for tokenization
-  `sacremoses` for tokenization
-  `rouge-score` for computing rouge score

## Data preparation

Use the command to get the Shakespeare (character-level) data
```sh
python data/shakespeare_char/prepare.py
```

Use the command to get the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/) data
```sh
python data/openwebtext/prepare.py
```