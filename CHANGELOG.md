# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0]

### Changed
- `BetaTCVAE` now optimizes a **genuine β-TC-VAE objective**: the KL term is
  decomposed into index-code mutual information, **total correlation**, and the
  dimension-wise KL (Chen et al., 2018), estimated with minibatch stratified
  sampling. `beta` now scales the real total-correlation term `KL[q(z) ||
  prod_j q(z_j)]` instead of the earlier closed-form proxy that was a function
  of the posterior variance only. `forward` returns `(logits, mu, log_var, z)`;
  `vae-train` gains an `--alpha` flag (defaults: `alpha=1`, `beta=4`, `gamma=1`).
- `BetaTCVAE` is now a **sentence VAE**: a molecule is encoded to a single
  fixed-size latent vector (masked mean-pool of the encoder states) rather than
  a per-position latent, so each molecule is one point in latent space — the
  natural setting for nearby sampling and interpolation. `latent_dim` is now
  independent of `embedding_dim` (the latent is projected back up before
  decoding), so it can act as a genuine bottleneck; `vae-train` gains a
  `--latent-dim` flag. This also removes the decoder's cross-attention memory,
  so generation and interpolation no longer differ in how they mask it.
- The VAE latent-space workflow now uses the shared toolkit tokenizers instead
  of a private 4-character toy tokenizer, so it works on arbitrary molecules
  (not just the synthetic C/O/ring dataset). `SelfiesTokenizer` is the default,
  making every decoded sample a syntactically valid molecule.
- `generate`/`interpolate` now take an in-memory `(model, tokenizer)` pair and
  reuse the standard checkpoint format, and are exposed as first-class CLI
  commands: `molgen vae-train`, `molgen vae-sample`, `molgen vae-interpolate`.

### Fixed
- Latent-space generation and interpolation previously returned **no
  molecules**: decoded token ids were clamped to `min(idx, 4)` (collapsing
  every output onto the pad/first tokens) and the toy tokenizer could only
  represent `C`, `O`, `(`, `)`. Both paths now produce valid, distinct
  molecules (covered by end-to-end tests in `tests/test_generate.py`).

## [0.1.0]

A ground-up modernization that turns the original VAE-only collection of scripts
into an installable, tested molecular-generation toolkit.

### Added
- Installable `molgen` package (`pip install -e .`) with `pyproject.toml`, ruff,
  pre-commit, a pytest suite, and GitHub Actions CI (Python 3.10–3.12).
- Atom-level regex SMILES tokenizer and a SELFIES tokenizer (guaranteed-valid decoding).
- RDKit chemistry helpers: validity, canonicalization, and SMILES enumeration.
- `SmilesDataset` with dynamic-padding collation and optional SMILES-enumeration
  augmentation, a bundled sample dataset, and a fragment-based diverse synthetic generator.
- Models: `CharRNN` (GRU/LSTM) and `MolGPT` (decoder-only Transformer), alongside
  the existing Transformer VAE.
- Training loop with AdamW, gradient clipping, validation, and mixed precision (AMP) on CUDA.
- Autoregressive sampling with temperature, top-k, and top-p (nucleus) filtering.
- MOSES-style metrics: validity, uniqueness, novelty, internal diversity, unique
  scaffolds, SNN, and QED/logP/MW/SA property summaries, plus an aggregate report.
- Model + tokenizer checkpointing and a `molgen` command-line interface (`train`/`sample`/`eval`).
- Reproducibility/logging utilities, a PEP 561 `py.typed` marker, and a runnable quickstart example.

### Fixed
- `generate`/`interpolate` ignored their `device` argument.
- `torch.load` now passes `map_location` and `weights_only`.
- Reconstruction loss and accuracy now ignore padding positions.
- The Transformer used `batch_first=False` on `(batch, seq, …)` inputs — so attention
  ran across the batch dimension — and lacked padding masks and positional encoding; all fixed.
- Vocabulary size is derived from the tokenizer instead of a hardcoded, off-by-one value.

### Notes
- The original VAE latent-space workflow is retained:
  `python -m molgen.{synthetic,vae,generate,interpolate}`.
