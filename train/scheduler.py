"""Learning rate scheduler: linear warmup + cosine decay."""

import math


class WarmupCosineScheduler:
    """Warmup for warmup_steps, then cosine decay to min_lr."""

    def __init__(self, optimizer, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.max_lr * self.current_step / self.warmup_steps
        elif self.current_step >= self.max_steps:
            return self.min_lr
        else:
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1
        return lr
