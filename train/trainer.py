"""Training loop for pre-training with wandb logging and checkpointing."""

import os
import time

import torch
import torch.nn.functional as F
import yaml

import wandb

from model.config import ModelConfig
from model.transformer import Transformer
from train.scheduler import WarmupCosineScheduler


class Trainer:
    def __init__(self, config_path: str, mixture_name: str, output_dir: str):
        with open(config_path) as f:
            self.train_config = yaml.safe_load(f)

        self.mixture_name = mixture_name
        self.output_dir = os.path.join(output_dir, mixture_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = self._get_device()
        dtype_str = self.train_config.get("dtype", "float32")
        self.dtype = getattr(torch, dtype_str)
        # Only use autocast when dtype is a reduced precision format
        self.use_autocast = self.dtype in (torch.bfloat16, torch.float16)

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def train(self, model: Transformer, dataloader, val_dataloader=None):
        model = model.to(self.device)
        config = self.train_config

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            betas=tuple(config["betas"]),
            weight_decay=config["weight_decay"],
        )

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=config["warmup_steps"],
            max_steps=config["max_steps"],
            max_lr=config["learning_rate"],
            min_lr=config["min_lr"],
        )

        wandb.init(
            project="domain-pretrain-study",
            name=self.mixture_name,
            config={
                "mixture": self.mixture_name,
                "model_params": sum(p.numel() for p in model.parameters()),
                **config,
            },
        )

        grad_accum = config["gradient_accumulation_steps"]
        max_grad_norm = config["max_grad_norm"]

        global_step = 0
        total_tokens = 0
        best_val_loss = float("inf")

        model.train()
        start_time = time.time()

        while global_step < config["max_steps"]:
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_autocast):
                    logits = model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, model.config.vocab_size),
                        labels.view(-1),
                    )
                    loss = loss / grad_accum

                loss.backward()
                total_tokens += input_ids.numel()

                if (global_step + 1) % grad_accum == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr = scheduler.step()

                    if global_step % config["log_every"] == 0:
                        elapsed = time.time() - start_time
                        tokens_per_sec = total_tokens / elapsed
                        wandb.log({
                            "train/loss": loss.item() * grad_accum,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm.item(),
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/total_tokens": total_tokens,
                        }, step=global_step)

                    if val_dataloader and global_step % config["eval_every"] == 0:
                        val_loss = self._evaluate(model, val_dataloader)
                        wandb.log({"val/loss": val_loss}, step=global_step)
                        model.train()

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self._save_checkpoint(model, optimizer, global_step, "best")

                    if global_step % config["checkpoint_every"] == 0 and global_step > 0:
                        self._save_checkpoint(model, optimizer, global_step, f"step_{global_step}")

                global_step += 1
                if global_step >= config["max_steps"]:
                    break

        self._save_checkpoint(model, optimizer, global_step, "final")
        wandb.finish()

        return model

    @torch.no_grad()
    def _evaluate(self, model: Transformer, dataloader, max_batches: int = 50) -> float:
        model.eval()
        total_loss = 0
        count = 0

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_autocast):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, model.config.vocab_size), labels.view(-1))
            total_loss += loss.item()
            count += 1

        return total_loss / max(count, 1)

    def _save_checkpoint(self, model, optimizer, step, name):
        path = os.path.join(self.output_dir, f"checkpoint_{name}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "mixture": self.mixture_name,
        }, path)
        print(f"  Saved checkpoint: {path}")
