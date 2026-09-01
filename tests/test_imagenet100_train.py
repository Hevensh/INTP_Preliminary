import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.imagenet100.train_vit import (
    DistributedContext,
    _run_epoch,
    _validate_cuda_architecture,
)


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


def test_gradient_accumulation_matches_one_large_batch_update():
    torch.manual_seed(0)
    images = torch.randn(4, 5)
    targets = torch.tensor([0, 1, 2, 1])
    full_loader = DataLoader(TensorDataset(images, targets), batch_size=4)
    micro_loader = DataLoader(TensorDataset(images, targets), batch_size=2)

    full_model = nn.Linear(5, 3)
    micro_model = nn.Linear(5, 3)
    micro_model.load_state_dict(full_model.state_dict())
    full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.1)
    micro_optimizer = torch.optim.SGD(micro_model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    context = DistributedContext(0, 0, 1, torch.device("cpu"))

    common = {
        "criterion": criterion,
        "device": torch.device("cpu"),
        "amp": False,
        "scheduler": None,
        "scaler": torch.amp.GradScaler("cuda", enabled=False),
        "grad_clip_norm": 0.0,
        "max_steps": None,
        "epoch": 1,
        "epochs": 1,
        "phase": "train",
        "progress_interval_seconds": 0.0,
        "distributed": context,
    }
    _run_epoch(
        model=full_model,
        loader=full_loader,
        optimizer=full_optimizer,
        gradient_accumulation_steps=1,
        **common,
    )
    _run_epoch(
        model=micro_model,
        loader=micro_loader,
        optimizer=micro_optimizer,
        gradient_accumulation_steps=2,
        **common,
    )

    for full_parameter, micro_parameter in zip(
        full_model.parameters(), micro_model.parameters()
    ):
        torch.testing.assert_close(full_parameter, micro_parameter)
