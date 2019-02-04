from typing import Tuple

import torch
from ignite.engine import Engine
from torch import nn
from torch.nn import functional as F

from tvae.nn import (PositionalEmbedding, TransformerDecoderLayer,
                     TransformerEncoderLayer)
from tvae.data import TVAEDataset


class TransformerVAE(nn.Module):
    """Transformer Variational Autoencoder.

    Args:
        max_seq_len (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embedding_weights (torch.Tensor, optional): Defaults to None. Tensor with embedding weights.
        pool_context (bool, optional): Defaults to False. Whether to pool context vector or use special context token.
        num_layers (int, optional): Defaults to 6. Number of transformer layers.
        emb_size (int, optional): Defaults to 600. Embedding size.
        latent_size (int, optional): Defaults to 100. Latent size.
        dim_m (int, optional): Defaults to 512. Model dimension.
        dim_i (int, optional): Defaults to 2048. Inner model dimension.
        n_heads (int, optional): Defaults to 8. Number of heads.
        initial_token_idx (int, optional): Defaults to 2. Initial token index.
        dropout (float, optional): Defaults to 0.1. Dropout probability.

    Input:
        - **input** (torch.Tensor): Input sequence tensor of shape ``(batch, seq_len)``.

    Outputs: seq_vocab_distr, mu, logvars
        - **first**: tensor of shape ``(batch, seq_len, :attr:`vocab_size`), containing output sequence distribution;
        - **second**: mu variable, **third**: logvar variable (both of shape ``(batch, :attr:`latent_size`)``).

    Notes:
        - If :attr: `embedding_weights` passed, : attr: `vocab_size` and: attr: `emb_size` will be inherited from
          embedding tensor shape.
    """

    def __init__(self,
                 max_seq_len: int,
                 vocab_size: int,
                 embedding_weights=None,
                 pool_context=False,
                 num_layers=6,
                 emb_size=600,
                 latent_size=100,
                 dim_m=512,
                 dim_i=2048,
                 n_heads=8,
                 initial_token_idx=2,
                 dropout=0.1):

        super(TransformerVAE, self).__init__()

        self.initial_token_idx = initial_token_idx

        if embedding_weights is not None:
            assert isinstance(embedding_weights, torch.Tensor), "embedding must be a torch.Tensor"
            vocab_size, emb_size = embedding_weights.shape
        self.vocab_size = vocab_size
        self.pool_context = pool_context

        message = 'Model `dim_m` must be divisible by `n_heads` without a remainder.'
        assert dim_m % n_heads == 0, message
        dim_proj = dim_m // n_heads

        encoder_decoder_args = {
            'dim_m': dim_m,
            'dim_q_k': dim_proj,
            'dim_v': dim_proj,
            'n_heads': n_heads,
            'dim_i': dim_i,
            'dropout': dropout
        }

        # Transformer
        self.embedding = PositionalEmbedding(max_seq_len, dim_m, vocab_size, emb_size, embedding_weights)
        # Encoder
        self.repr_token = nn.Embedding(1, dim_m)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(**encoder_decoder_args) for _ in range(num_layers)
        ])
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(**encoder_decoder_args) for _ in range(num_layers)
        ])
        self.out = nn.Sequential(
            nn.Linear(dim_m, vocab_size)
        )

        # VAE
        self.to_mu = nn.Linear(dim_m, latent_size)
        self.to_logvar = nn.Linear(dim_m, latent_size)
        self.decode_latent = nn.Linear(latent_size, dim_m)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = input.shape[0]
        input_embedded = self.embedding(input)

        if not self.pool_context:
            # Append representation token
            repr_embedded_token = self.repr_token(torch.zeros(batch_size, 1, dtype=torch.long, device=input.device))
            encoder_state = torch.cat((input_embedded, repr_embedded_token), dim=1)
        else:
            encoder_state = input_embedded

        for encoder_layer in self.encoder_layers:
            encoder_state = encoder_layer(encoder_state)

        if not self.pool_context:
            # Use last hidden state as sequence context vector:
            seq_repr = encoder_state[:, -1, :].view(batch_size, -1)
        else:
            seq_repr = encoder_state.mean(dim=1).view(batch_size, -1)

        # Reparameterize
        mu = self.to_mu(seq_repr)
        logvar = self.to_logvar(seq_repr)
        z = self.reparameterize(mu, logvar)

        encoder_state = self.decode_latent(z).view(batch_size, 1, -1)

        # Decode
        mask = self.autoregressive_mask(input)
        decoder_state = input_embedded
        for decoder_layer in self.decoder_layers:
            decoder_state = decoder_layer(decoder_state, encoder_state, mask)

        output = self.out(decoder_state)[:, :-1, :]
        return output.contiguous(), mu, logvar

    def inference(self, sequence: torch.Tensor = None, z: torch.Tensor = None, limit=50)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (sequence is not None) or (z is not None), "`sequence` or `z` must be specified"

        if sequence is not None:
            device = sequence.device
            batch_size = sequence.shape[0]
            sequence_embedded = self.embedding(sequence)

            if not self.pool_context:
                # Append representation token
                repr_embedded_token = self.repr_token(torch.zeros(batch_size, 1, dtype=torch.long,
                                                                  device=sequence.device))
                encoder_state = torch.cat((sequence_embedded, repr_embedded_token), dim=1)
            else:
                encoder_state = sequence_embedded

            for encoder_layer in self.encoder_layers:
                encoder_state = encoder_layer(encoder_state)

            if not self.pool_context:
                # Use last hidden state as sequence context vector:
                seq_repr = encoder_state[:, -1, :].view(batch_size, -1)
            else:
                seq_repr = encoder_state.mean(dim=1).view(batch_size, -1)

            # Reparameterize
            mu = self.to_mu(seq_repr)
            logvar = self.to_logvar(seq_repr)
            z = self.reparameterize(mu, logvar)
        else:
            device = z.device
            batch_size = z.shape[0]

        encoder_state = self.decode_latent(z).view(batch_size, 1, -1)

        generated_seq = torch.full((batch_size, 1),
                                   self.initial_token_idx,
                                   dtype=torch.long,
                                   device=device)

        # Decode:
        for _ in range(limit):
            generated_embedded = self.embedding(generated_seq)
            mask = self.autoregressive_mask(generated_seq)
            decoder_state = generated_embedded
            for decoder_layer in self.decoder_layers:
                decoder_state = decoder_layer(decoder_state, encoder_state, mask)

            output_distr = self.out(decoder_state)
            last_generated_token_idx = output_distr[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_seq = torch.cat((generated_seq, last_generated_token_idx), dim=-1)
        return generated_seq[:, 1:].contiguous(), output_distr, z

    def create_trainer(self, optimizer: torch.optim.Optimizer, device: torch.device) -> Engine:
        def _update(engine: Engine, batch: dict):
            # TODO: Add scheduled KLD
            batch["src"] = batch["src"].to(device)
            self.train()

            optimizer.zero_grad()
            sequence_distr, mu, logvar = self.forward(batch["src"])

            ce = F.cross_entropy(sequence_distr.view(-1, self.vocab_size), batch["src"][:, 1:].contiguous().view(-1))
            kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()

            loss = ce + kld
            loss.backward()
            optimizer.step()

            engine.state.kld = kld.item()
            engine.state.ce = ce.item()
            engine.state.loss = loss.item()
            return loss.item()

        return Engine(_update)

    def create_evaluator(self, device: torch.device) -> Engine:
        def _evaluate(engine: Engine, batch: dict):
            batch["src"] = batch["src"].to(device)
            self.eval()

            generated_seq, output_distr, z = self.inference(batch["src"], limit=batch["src"].shape[1] - 1)
            engine.state.generated_seq = generated_seq

            return output_distr, batch["src"][:, 1:].contiguous()
        return Engine(_evaluate)

    def learnable_parameters(self):
        """Get all learnable parameters of the model.
        Returns: Generator of parameters.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    @staticmethod
    def autoregressive_mask(tensor):
        """Generate auto - regressive mask for tensor. It's used to preserving the auto - regressive property.
        Args:
            tensor(torch.Tensor): of shape ``(batch, seq_len)``.
        Returns:
            torch.Tensor: a byte mask tensor of shape ``(batch, seq_len, seq_len)`` containing mask for
            illegal attention connections between decoder sequence tokens.
        """
        batch_size, seq_len = tensor.shape
        x = torch.ones(
            seq_len, seq_len, device=tensor.device).tril(-1).transpose(0, 1)

        return x.repeat(batch_size, 1, 1).byte()

    @staticmethod
    def create(dataset: TVAEDataset, margs: dict):
        embedding_dump = dataset.get_embeddings()
        if embedding_dump is not None:
            margs["embedding_weights"] = torch.from_numpy(embedding_dump).float()
        # Choose max sequence length
        margs["max_seq_len"] = dataset.max_sequence_length
        margs["vocab_size"] = len(dataset.spm_model)
        model = TransformerVAE(**margs)
        return model, margs
