import os
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch

DATA_ROOT = "kaggle/input/blablabla/" if Path("kaggle/input/blablabla/").exists() else "data/"


def set_seeds(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train")
    subparsers.add_parser("validate")
    subparsers.add_parser("predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")
    print(f"data: {DATA_ROOT}")


if __name__ == "__main__":
    main()
