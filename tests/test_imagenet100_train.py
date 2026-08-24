import pytest
import torch

from experiments.imagenet100.train_vit import _validate_cuda_architecture


def test_rejects_gpu_older_than_compiled_torch_arches(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (6, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_70", "sm_75", "sm_100"])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Tesla P100")
    with pytest.raises(RuntimeError, match="select a T4/L4"):
        _validate_cuda_architecture(torch.device("cuda"))


def test_accepts_supported_gpu_architecture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_70", "sm_75", "sm_100"])
    _validate_cuda_architecture(torch.device("cuda"))
