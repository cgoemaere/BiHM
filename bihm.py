import math
from functools import partial
from itertools import chain
from typing import Literal

import torch
import torch.distributions as distr
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn

MASK_VALUE = -1.0


class BiHM(LightningModule):
    """
    Code for the Bidirectional Helmholtz Machine, based on https://proceedings.mlr.press/v48/bornschein16.html

    This implementation is hardcoded to work on MNIST with the following architecture:
    img <-> z01 <-> ... <-> z12

    All distributions are Bernoulli

    Steps:
    1) Sample a batch z' from q_enc(z|x) using a bottom-up sweep
    2) Calculate E_p(x, z') and E_q(x, z') as if p(z) and q(x) are uniform
    3) Calculate w = SoftMax_z'(½ E_q(x, z') - ½ E_p(x, z'))
    4) Perform w-weighted gradient descent on E_p(x, z') + E_q(x, z')

    All nodes have the following shape: (batch_size, nr_samples, *node_shape)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.node_keys = ["img"] + [f"z{i:02}" for i in range(1, 13)]
        self.node_sizes = [28 * 28, 300, 200, 100, 75, 50, 35, 30, 25, 20, 15, 10, 10]

        self.node_dict = {key: None for key in self.node_keys}

        def distr_generator(loc, event_dims=1):
            # Bernoulli
            return distr.Independent(distr.Bernoulli(logits=loc), event_dims)

        # Priors and NLL for predictors of these nodes
        self.node_distr = {key: distr_generator for key in self.node_keys}
        self.node_distr["img"] = partial(distr_generator, event_dims=3)

        # "from_node->to_node": predictor_network
        self.q_enc_layers = nn.ModuleDict(
            {
                f"{from_key}->{to_key}": nn.Linear(from_size, to_size)
                for from_key, to_key, from_size, to_size in zip(
                    self.node_keys[:-1],
                    self.node_keys[1:],
                    self.node_sizes[:-1],
                    self.node_sizes[1:],
                )
            }
        )
        first_key = f"img->{self.node_keys[1]}"
        self.q_enc_layers.update(
            {
                first_key: nn.Sequential(
                    nn.Flatten(start_dim=-3),
                    self.q_enc_layers[first_key],
                )
            }
        )

        # "from_node->to_node": predictor_network
        self.p_dec_layers = nn.ModuleDict(
            {
                f"{from_key}->{to_key}": nn.Linear(from_size, to_size)
                for from_key, to_key, from_size, to_size in zip(
                    self.node_keys[1:],
                    self.node_keys[:-1],
                    self.node_sizes[1:],
                    self.node_sizes[:-1],
                )
            }
        )

        class Prior(nn.Module):
            def forward(self, x):
                try:
                    return self.prior
                except AttributeError:
                    # Just set the correct shape once, and store it
                    self.prior = torch.full_like(x, 0.5)  # Bernoulli prior

                    return self.prior

        first_key = f"{self.node_keys[1]}->img"
        self.p_dec_layers.update(
            {
                f"{self.node_keys[-1]}->{self.node_keys[-1]}": Prior(),
                first_key: nn.Sequential(
                    self.p_dec_layers[first_key], nn.Unflatten(-1, (1, 28, 28))
                ),
            }
        )

        self.nr_train_samples = 10
        self.gibbs_iters = 4
        self.nr_gibbs_samples = 8

    def on_fit_start(self):
        # Store batch_size for easy access
        self.batch_size = self.trainer.datamodule.batch_size

        # Pre-allocate (needed for advanced indexing in Gibbs sampling code)
        self.torch_arange_batch_size = torch.arange(self.batch_size, device=self.device)

        # Initialize layers as in BiHM paper
        def paper_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, -1.0)

        self.apply(paper_init)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        from itertools import chain

        param_optimizer = torch.optim.Adam(
            chain(self.p_dec_layers.parameters(), self.q_enc_layers.parameters()),
            lr=1e-3,
            # fused=True, # uncomment on CUDA for faster training
        )
        return param_optimizer

    def on_train_epoch_end(self):
        # Adjust learning rate and K at epoch 60 (where learning plateaus)
        if self.current_epoch == 60:
            # Modify the learning rate
            new_lr = 4e-3
            for param_group in self.optimizers().param_groups:
                param_group["lr"] = new_lr

            # Modify K
            self.nr_train_samples = 100

            # Log the changes
            print("Reached epoch 60, switching to lr=4e-3 & K=100")

    def E(self, model: Literal["q_enc", "p_dec"]):
        """Returns the negative log-likelihood of the states under the given model"""
        E = 0.0  # sum of NLLs

        for edge, layer in getattr(self, model + "_layers").items():
            (n0, n1) = edge.split("->")

            # Create the distribution with the predicted variational parameters
            distr_pred = self.node_distr[n1](layer(self.node_dict[n0]))

            # Add the NLL of the actual node value according to the prediction
            E -= distr_pred.log_prob(self.node_dict[n1])

        return E

    @torch.no_grad()
    def q_enc_sample(self, img=None, K: int = 1):
        if img is not None:
            # Add nr_samples dim in position 1
            self.node_dict["img"] = img.unsqueeze(1)
        # else:
        #     Work with the existing self.node_dict["img"]

        if K > 1:
            self.node_dict["img"] = self.node_dict["img"].expand(-1, K, -1, -1, -1)

        z_prev = self.node_dict["img"]
        prev_node = "img"

        # Iteratively sample a value for the next node, given the value of the previous node
        for next_node in self.node_keys[1:]:
            edge = f"{prev_node}->{next_node}"
            z_next = self.q_enc_layers[edge](z_prev)
            z_prev = self.node_dict[next_node] = self.node_distr[next_node](z_next).sample()
            prev_node = next_node

    @torch.no_grad()
    def p_dec_sample(self):
        # We start from the prior for the last node
        # Note that this prior has the shape (batch_size, self.nr_train_samples, *node_shape)
        z_prev = None

        # Our node naming allows key sorting for the right order of the model
        for edge in sorted(self.p_dec_layers.keys(), reverse=True):
            z_next = self.p_dec_layers[edge](z_prev)
            next_node = edge.split("->")[1]
            z_prev = self.node_dict[next_node] = self.node_distr[next_node](z_next).sample()

        return z_next  # = Pr[img], without sampling (cfr. Fig. 1.A of BiHM paper)

    def training_step(self, batch, batch_idx):
        # Draw the samples and calculate their energies
        self.q_enc_sample(batch["img"], K=self.nr_train_samples)
        energies_p = self.E("p_dec")  # = (batch_size, K)
        energies_q = self.E("q_enc")  # = (batch_size, K)

        # Calculate K importance weights that sum up to 1 (per input)
        # w = SoftMax_z'(½ E_q(x, z') - ½ E_p(x, z'))
        w = F.softmax(0.5 * (energies_q - energies_p), dim=1).detach()  # = (batch_size, K)

        # Per batch item, take the weighted sum of the energies.
        # Then, average across the batch elements.
        # So: loss = (w*(energies_p+energies_q)).sum(dim=1).mean()
        # Or more efficiently:
        loss = torch.dot(w.flatten(), (energies_p + energies_q).flatten()) / self.batch_size

        self.log("loss", loss, prog_bar=True)

        # L1 regularization: sum(|w|) for all parameters
        l1_norm = sum(
            torch._foreach_norm(
                [p for n, p in self.named_parameters() if n.endswith("weight")], ord=1
            )
        )

        # Add L1 regularization to the loss
        loss += 0.001 * l1_norm

        return loss

    @torch.no_grad()
    def gibbs_sample(self, node: str, mask=None):
        node_distr = self.node_distr[node]
        nr_samples = self.nr_gibbs_samples

        ###############################################
        ### Step 1: draw samples from 0.5*p + 0.5*q ###
        ###############################################
        # Draw samples from the predictive layer in p_dec
        for edge, layer in self.p_dec_layers.items():
            n0, n1 = edge.split("->")
            if n1 == node:
                p_samples = node_distr(layer(self.node_dict[n0])).sample((nr_samples,))
                break

        # Draw samples from the predictive layer in q_enc
        if node != "img":
            for edge, layer in self.q_enc_layers.items():
                n0, n1 = edge.split("->")
                if n1 == node:
                    q_samples = node_distr(layer(self.node_dict[n0])).sample((nr_samples,))
                    break
        else:  # there is no predictive layer for img in q_enc, so we just sample from p
            q_samples = p_samples

        # Pick samples from p or q with 50-50 chance
        samples = torch.where(
            torch.randint(
                0,
                2,
                (nr_samples,) + (1,) * self.node_dict[node].dim(),
                dtype=bool,
                device=self.device,
            ),
            p_samples,
            q_samples,
        )

        # When drawing multiple samples from torch.distrutions, the returned shape
        # always adds a dimension in position 0 for nr_samples. For consistency, we
        # make sure sampling dim is the second dim, after batch dim.
        samples = samples.transpose(0, 1).squeeze(2)

        # Only sample the image where it was masked, keep the rest fixed
        if node == "img" and mask is not None:
            img_repeated = self.node_dict[node].expand(self.batch_size, nr_samples, 1, 28, 28)
            mask_repeated = mask.expand(self.batch_size, nr_samples, 1, 28, 28)
            samples = torch.where(mask_repeated, samples, img_repeated)

        ###############################################################
        ### Step 2: calculate the importance weights of the samples ###
        ###############################################################
        ## Formula for w (see p.5 of BiHM paper):
        # num_w = √(Π Pr[node | other_node] * Π Pr[other_node | node])
        # denom_w = Σ Pr[node | other_node]
        # =>
        # log_num_w   = 0.5 *  (Σ LogPr[node | other_node] + Σ LogPr[other_node | node])
        # log_denom_w = LogSumExp(LogPr[node | other_node])
        log_num_w = 0.0
        log_denom_w = []

        for edge, layer in chain(self.p_dec_layers.items(), self.q_enc_layers.items()):
            n0, n1 = edge.split("->")
            local_distr = self.node_distr[n1]

            if n1 == node:  # predictive layer (used both in numerator and denominator)
                log_prob = local_distr(layer(self.node_dict[n0])).log_prob(samples)
                log_num_w += 0.5 * log_prob
                log_denom_w.append(log_prob)

            elif n0 == node:  # discriminative layer (used only in numerator)
                log_prob = local_distr(layer(samples)).log_prob(self.node_dict[n1])
                log_num_w += 0.5 * log_prob

        # Calculate final weights as log_num - log_denom
        # We use stack to create a new dim over which we can logsumexp-reduce.
        log_w = log_num_w - torch.logsumexp(torch.stack(log_denom_w, dim=0), dim=0)

        ###################################################################
        ### Step 3: use w-weighted random selection of the final sample ###
        ###################################################################
        # Select (indices of) final samples, based on w
        sample_indx = torch.multinomial(F.softmax(log_w, dim=1), num_samples=1).squeeze(-1)

        # Extract the samples based on the indices, and re-insert the sample dim in position 1
        self.node_dict[node] = samples[self.torch_arange_batch_size, sample_indx].unsqueeze(1)

    def forward(self, img):
        # Do a first node init by sampling from q_enc
        self.q_enc_sample(img)

        # Improve results through block Gibbs sampling
        even_layers = self.node_keys[::2]
        odd_layers = self.node_keys[1::2]
        mask = (img == MASK_VALUE).unsqueeze(1)
        for _ in range(self.gibbs_iters):
            # First, update the even layers, then the odd layers
            for layer in chain(even_layers, odd_layers):
                self.gibbs_sample(layer, mask=mask)

        # We don't need to return anything during training.
        # At inference, we can easily access the models node values through node_dict.

    @torch.no_grad()
    def estimate_2logZ(self, K_outer_factor: int = 1):
        """
        Provides a biased estimator of 2 log Z (the normalization constant)
        One can obtain an unbiased estimator of Z² through exponentiation.

        See Eq. 7 in the BiHM paper.

        For ease of implementation, we set K_outer  = k * batch_size * nr_train_samples

        Interestingly, 2 log Z = -2 D_B(p, q) (= Bhattacharyya distance),
        so we want 2 log Z to be as high as possible.
        """
        log_E_outer = []  # log of outer expected value

        K_outer = K_outer_factor * self.batch_size * self.nr_train_samples
        K_inner = 1
        for _ in range(K_outer_factor):
            # Outer sampling
            self.p_dec_sample()
            log_sqrt_q_over_p = 0.5 * (self.E("p_dec") - self.E("q_enc"))

            # Inner sampling, starting from x ~ p_dec
            self.q_enc_sample(K=K_inner)
            log_sqrt_p_over_q = 0.5 * (self.E("q_enc") - self.E("p_dec"))

            # Sum over all samples in the batch (later, we'll divide by the necessary constant for averaging)
            log_E_outer.append(torch.logsumexp(log_sqrt_q_over_p + log_sqrt_p_over_q, dim=(0, 1)))

        # Finally, average over the number of samples to get the expected value
        # E_outer = 1/K sum(...) ==> log_E_outer = logsumexp(...) - log(K)
        # When split up in parts, logsumexp(...) = logsumexp(logsumexp(part1), logsumexp(part2), ...)
        return torch.logsumexp(torch.stack(log_E_outer, dim=0), dim=0) - math.log(K_outer)

    def _log_pow_p(self, pow: float, img=None, K: int = 64):
        """Helper function to compute log (E[p_over_q ^ 1/pow]) ^ pow"""
        self.q_enc_sample(img, K)
        log_invpow_p_over_q = (self.E("q_enc") - self.E("p_dec")) / pow  # = (batch_size, K)

        # log (E[p_over_q ^ 1/pow]) ^ pow
        # = pow * log E[p_over_q ^ 1/pow]
        # = pow * (log sum p_over_q ^ 1/pow -  log K)
        # = pow * (log sum exp log_invpow_p_over_q - log K)
        #
        # To end up with a single scalar, we take the mean across the batch size.
        return pow * (torch.logsumexp(log_invpow_p_over_q, dim=1).mean() - math.log(K))

    def _log_p(self, img=None, K=64):
        """Return an estimate for log p(x) (see Eq. 5 of BiHM paper)"""
        # log p(x)  = log E[p_over_q]
        return self._log_pow_p(1.0, img, K)

    def _log_p_tilde_star(self, img=None, K=64):
        """Return an estimate for log p̃*(x) (see Eq. 4 of BiHM paper)"""
        # log p*(x) = log (E[sqrt p_over_q])²
        return self._log_pow_p(2.0, img, K)

    def _log_p_star(self, img=None, K=64):
        """Return an estimate for log p*(x) (see Eq. 1 of BiHM paper)"""
        return self._log_p_tilde_star(img, K) - self.estimate_2logZ()

    def prob_metrics_dict(self, prefix: str = ""):
        est_2logZ = self.estimate_2logZ()
        log_p = self._log_p()
        log_p_star = self._log_p_tilde_star() - est_2logZ

        return {
            prefix + "2logZ": est_2logZ,
            prefix + "log_p": log_p,
            prefix + "log_p_star": log_p_star,
        }

    def log_prob_metrics(self, batch, prefix: str = ""):
        self.forward(img=batch["img"])
        self.log_dict(self.prob_metrics_dict(prefix=prefix), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.log_prob_metrics(batch, prefix="val_")

    def test_step(self, batch, batch_idx):
        self.log_prob_metrics(batch, prefix="test_")

    def masked_reconstruction_loss(self, batch):
        # Hardcoded: mask bottom half of image (in effort to reconstruct it with Gibbs sampling)
        img = batch["img"].clone()
        img[..., 14:] = MASK_VALUE

        self.forward(img=img)

        # Check how good the pixel-wise reconstruction is on average
        # But of course, don't take the clamped half of the img into account (=> loss *= 2)
        loss = F.mse_loss(self.node_dict["img"].squeeze(1), batch["img"], reduction="mean") * 2

        return loss

    def predict_step(self, batch, batch_idx, use_p_star=True):
        if use_p_star:
            loss = self.masked_reconstruction_loss(batch)
            print("masked_reconstruction_loss =", loss.item())
            img = self.node_dict["img"]
        else:
            img = self.p_dec_sample()[:, 0]  # select the first of self.nr_train_samples

        print("Energy =", 0.5 * (self.E("p_dec") + self.E("q_enc")).mean())

        return {"img": img}
