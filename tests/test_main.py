import torch

from lvg.main import resolve_device


def test_resolve_device_explicit() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto() -> None:
    assert resolve_device(None).type in {"cuda", "mps", "cpu"}
