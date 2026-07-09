"""molgen: a lightweight toolkit for de novo molecular generation.

Provides SMILES and SELFIES tokenizers, CharRNN / MolGPT / VAE models, a
mixed-precision training loop, configurable sampling, and a MOSES-style metric
suite. See https://github.com/DaoyuanLi2816/molgen for documentation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("molgen")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "0.2.0"

__all__ = ["__version__"]
