import unittest

import torch
from torch import optim

from tvae.models import TransformerVAE


class TestTransformerVAEMethods(unittest.TestCase):
    def setUp(self):
        self.max_seq_len = 100
        self.max_vocab_size = 100
        self.latent_size = 100
        self.model = TransformerVAE(self.max_seq_len, self.max_vocab_size, latent_size=self.latent_size,
                                    num_layers=1, dim_m=32, dim_i=64)
        self.optimizer = optim.Adam(self.model.learnable_parameters())

    def test_init(self):
        self.assertIsInstance(self.model, TransformerVAE)

    def test_forward(self):
        input_sequence = torch.randint(0, 100, (8, 42))
        output_seq_distr, mu, logvar = self.model(input_sequence)

        self.assertTupleEqual(output_seq_distr.shape, (8, 41, self.max_vocab_size))
        self.assertTupleEqual(mu.shape, (8, self.latent_size))
        self.assertTupleEqual(logvar.shape, (8, self.latent_size))

    def test_inference_sequence(self):
        input_sequence = torch.randint(0, 100, (8, 42))
        generated_seq, generated_seq_distr, z = self.model.inference(input_sequence, limit=50)

        self.assertTupleEqual(generated_seq.shape, (8, 50))
        self.assertTupleEqual(generated_seq_distr.shape, (8, 50, self.max_vocab_size))
        self.assertTupleEqual(z.shape, (8, self.latent_size))

    def test_inference_z(self):
        z = torch.randn((8, self.latent_size))
        generated_seq, generated_seq_distr, z = self.model.inference(z=z, limit=50)

        self.assertTupleEqual(generated_seq.shape, (8, 50))
        self.assertTupleEqual(generated_seq_distr.shape, (8, 50, self.max_vocab_size))
        self.assertTupleEqual(z.shape, (8, self.latent_size))

    def test_inference_invalid_input(self):
        with self.assertRaises(AssertionError):
            self.model.inference()

    def test_trainer(self):
        data = [{"src": torch.randint(0, 100, (8, 42))}]
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.model.to(device)
        else:
            device = torch.device("cpu")
        trainer = self.model.create_trainer(self.optimizer, device)
        self.assertIsNotNone(trainer)

        state = trainer.run(data)
        kld, ce, loss = state.kld, state.ce, state.loss
        self.assertIsInstance(kld, float)
        self.assertIsInstance(ce, float)
        self.assertIsInstance(loss, float)

    def test_evaluator(self):
        data = [{"src": torch.randint(0, 100, (8, 42))}]
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.model.to(device)
        else:
            device = torch.device("cpu")

        data = [{"src": torch.randint(0, 100, (8, 42))}]
        evaluator = self.model.create_evaluator(device)

        state = evaluator.run(data)
        generated, original = state.output
        self.assertTupleEqual(generated.shape, (8, 41, self.max_vocab_size))
        self.assertTupleEqual(original.shape, (8, 41))
