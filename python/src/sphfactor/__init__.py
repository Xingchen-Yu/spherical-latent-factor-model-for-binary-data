"""Spherical latent factor model for binary data (Yu & Rodriguez, arXiv:2008.05109) in JAX."""

import os

# XLA's default 75% GPU preallocation fails on WSL.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
