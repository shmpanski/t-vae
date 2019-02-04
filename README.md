# Transformer Variational Autoencoder

- [Transformer Variational Autoencoder](#transformer-variational-autoencoder)
  - [Description](#description)
  - [Training launch](#training-launch)

## Description

This is a little experiment with VAE.
I want to try connect self-attention model with variational autoencoder.
But there are some problems with this kind of transformer application.

First of this: transformer don't encode sequence representation into single continuous vector.
It encodes sequence into seuqence of hidden representations:
`(t1, t2, ..., tn) -> (h1, h2, ..., hn)`.
I suggest using a special token `<repr>`, that will presumably be trained to store the hidden state of the entire sequence:
`(t1, t2, ..., tn, <repr>) -> (h1, h2, ..., hn, h_repr)`.

Also, it's possible to use pooled attention as a context vector of encoded sequence.
Check `pool_context` attribute of `TransformerVAE`.

*All of this approaches work bad. Need more research in this direction*


## Training launch

Use `*.yml` config to describe runs.

```
python train.py workbench/run.yml
```
