# Contributing

## Setup

[UV](https://docs.astral.sh/uv/) Python package manager.

```bash
git clone https://github.com/vhrabar/local-vs-global-skin-lesions.git
cd local-vs-global-skin-lesions

uv sync --group dev --group cpu   # CPU (+ Apple MPS)
uv sync --group dev --group cuda  # NVIDIA CUDA 13.2
uv sync --group dev --group rocm  # AMD ROCm 7.14

# auto git hook config
uv run pre-commit install
```

## Dataset

Download HAM10000 into `data/`

```bash
uv run scripts/download_ham10k.py data/ --with-test --with-segmentations
```
