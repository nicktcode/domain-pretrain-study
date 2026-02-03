import torch
import numpy as np
import pytest


def test_dataset_shapes():
    """Dataset should return correct input/label shapes."""
    from train.dataset import PretrainDataset

    tokens = np.arange(1000, dtype=np.uint16)
    ds = PretrainDataset(tokens, context_length=64)

    assert len(ds) > 0
    sample = ds[0]
    assert sample["input_ids"].shape == (64,)
    assert sample["labels"].shape == (64,)
    assert sample["labels"][0].item() == sample["input_ids"][0].item() + 1


def test_scheduler_warmup():
    """LR should increase during warmup."""
    from train.scheduler import WarmupCosineScheduler

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=100, max_steps=1000, max_lr=6e-4, min_lr=6e-5)

    lrs = []
    for _ in range(100):
        lr = scheduler.step()
        lrs.append(lr)

    assert lrs[-1] > lrs[0]
    assert abs(lrs[-1] - 6e-4) < 1e-5


def test_scheduler_cosine_decay():
    """LR should decrease after warmup toward min_lr."""
    from train.scheduler import WarmupCosineScheduler

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=10, max_steps=100, max_lr=6e-4, min_lr=6e-5)

    for _ in range(10):
        scheduler.step()

    lrs = []
    for _ in range(90):
        lr = scheduler.step()
        lrs.append(lr)

    assert lrs[-1] < lrs[0]
    assert lrs[-1] < 1e-4


def test_one_training_step():
    """A single training step should reduce loss."""
    from model.config import ModelConfig
    from model.transformer import Transformer
    from train.dataset import PretrainDataset

    config = ModelConfig(n_layers=2, d_model=64, n_heads=2, d_ff=256, vocab_size=100)
    model = Transformer(config)

    tokens = np.random.randint(0, 100, size=500, dtype=np.uint16)
    ds = PretrainDataset(tokens, context_length=32)
    batch = ds[0]

    input_ids = batch["input_ids"].unsqueeze(0)
    labels = batch["labels"].unsqueeze(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logits = model(input_ids)
    loss1 = torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
    loss1.backward()
    optimizer.step()
    optimizer.zero_grad()

    logits = model(input_ids)
    loss2 = torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

    assert loss2.item() < loss1.item(), "Loss should decrease after one step"
