
# Bidirectional Helmholtz Machines
This repo is an implementation of [Bidirectional Helmholtz Machines](https://proceedings.mlr.press/v48/bornschein16.html) in PyTorch Lightning, unlike the [original implementation](https://github.com/jbornschein/bihm) in Theano.

> [!CAUTION]
> Unfortunately, I was unable to reproduce the results from the original paper with this codebase. The cause is likely hidden in the details of the original implementation.

All sampling is done in parallel, resulting in a swift implementation that strongly benefits from GPU acceleration. On my GTX-1080Ti, training speeds reach up to 40it/s or 12s/epoch. Validation is a bit slower, because of the Gibbs sampling procedure.

## Why this repo?
My interest in this paper comes from its connection to [Predictive Coding](https://arxiv.org/abs/2207.12316), which is also a probabilistic model. In fact, trade out the Bernoulli's for Gaussians and the sampling for variational inference, and it's the same model.

I was hoping these two adjustments would be straightforward to implement, to end up with a bidirectional formulation of Predictive Coding. Unfortunately, this proved more challenging than I thought, starting from the fact that I couldn't even reproduce the original paper's scores.

Anyway, I hope this repo can still help people who are interested in BiHMs and are looking for a fast implementation in PyTorch.

## Repo structure
- *binarizedmnist*: contains the implementation for Binarized MNIST (Murray & Salakhutdinov, 2009). Includes a PyTorch `Dataset`, a `LightningDataModule` and a Lightning `Callback` for visualization.
- `bihm.py`: full implementation of the BiHM in a `LightningModule` with many explanatory comments. For functionality, see below.
- `BiHM_playground.ipynb`: **a plug&play notebook to quickly get started** with BiHMs on Binarized MNIST. Uses `wandb` for external logging.


## How it works
The BiHM is a VAE-like model that learns to model the input data in an unsupervised fashion.

It internally consists of two independent models:
* $q_{enc}(z | x)$: the bottom-up 'encoder', going from image to hidden states
* $p_{dec}(x | z)$: the top-down 'decoder', generating an image from the hidden states

Unlike VAEs, these two models share internal representations. In other words, the weights are independent, but the activations are shared!

During training, we sample hidden states $z$ from the input image $x$ via $q_{enc}$ and compare how well they are modelled under $p_{dec}$. The loss aims to minimize the discrepancy between $q_{enc}$ and $p_{dec}$ on these samples.

At inference, one can easily generate images using just $p_{dec}$. However, the paper shows that you can improve the generation quality by using $p^* = 1/Z \sqrt{p_{dec} * q_{enc}}$ (or equivalently, the average of the log likelihoods). Of course, since the hidden states are shared, sampling from this bidirectional model is difficult and requires Gibbs sampling.

## How it's implemented
For stability, I implemented everything using the negative log-likelihoods (aka energy function $E$).

Below is the energy formulation of the original paper's Algorithm 1:
1.  Sample a batch $z’$ from $q_{enc}(z|x)$ using a bottom-up sweep
2.  Calculate $E_p(x, z’)$ and $E_q(x, z’)$, but ignore $p(z)$ and $q(x)$ (pretend they're uniform)
3.  Calculate $w = \text{SoftMax}_{z’}(\frac{1}{2} E_q(x, z’) - \frac{1}{2} E_p(x, z’))$
4.  Perform $w$-weighted gradient descent on $E_p(x, z’) + E_q(x, z’)$
	-   :exclamation: No gradient flow through $w$!
	-   :exclamation: All nodes are observed: no need to backprop through layers!

> [!NOTE]
> The formula for $w$ prefers samples that are less likely in $q_{enc}$ than in $p_{dec}$. You might think that this will not happen very often, as we are sampling from $q_{enc}$, so most samples should be quite likely in $q_{enc}$. However, this is not true, because typical samples of high-dimensional distributions have _low_ probability (see [Gaussian Soap Bubble](https://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/)).

## Installation
To install the repo with the exact dependencies from my setup, run this in the command line:
```
git clone https://github.com/cgoemaere/BiHM
cd BiHM/
conda create --name bihm_test_env --file requirements.txt -c conda-forge -c pytorch
conda activate bihm_test_env 
python -c "from bihm import BiHM; print(BiHM)" #just a little test; should print "<class 'bihm.BiHM'>"
```

## Possible improvements
* Speed
	* Calculate the energy over all layers in parallel instead of sequentially looping over them
	* Use `torch.compile` (my GPU is too old for this...)
* Model
	* Find out why this code doesn't reproduce the paper's results
	* Go beyond Bernoulli distributions
	* Add a classification node $\hat y$
