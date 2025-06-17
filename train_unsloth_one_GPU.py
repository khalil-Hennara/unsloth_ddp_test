from unsloth import FastLanguageModel

import torch
import torch.nn as nn  # noqa: F401  # kept for possible future use
from torch.utils.data import DataLoader
import logging
import time
import random 
import numpy as np


def set_seed():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    
def setup_logger():
    """Return a process‑agnostic logger for single‑GPU runs."""
    logger = logging.getLogger("train")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class MyDataset(torch.utils.data.Dataset):
    """Dummy dataset that always returns the same 4K‑token sample."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return 1000

    def __getitem__(self, idx):  # noqa: D401
        enc = self.tokenizer(
            "Hello world!",
            return_tensors="pt",
            padding="max_length",
            padding_side="right",
            max_length=4096,
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0),
        }


def train(args):
    """Main training loop – *single GPU* (cuda:0) only."""
    set_seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = setup_logger()
    logger.info(f"Using device {device}")

    # ────────────────────────────────────────────────────────────────── model ────
    logger.info("Loading model with FastLanguageModel.from_pretrained …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=not args.use_lora,
        use_gradient_checkpointing=True,
        device_map={"": device},  # put *everything* on cuda:0
        disable_log_stats=False,
    )
    logger.info("Model & tokenizer ready.")

    if args.use_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        # quick param count
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Trainable parameters: {total_trainable:,} / {total_params:,} "
            f"({100 * total_trainable / total_params:.2f}% trainable)"
        )

    model.to(device)

    # ──────────────────────────────────────────────────────────────── data ────
    dataset = MyDataset(tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # ──────────────────────────────────────────────────────────── optimisation ────
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    torch.cuda.reset_peak_memory_stats(device)
    model.train()

    start_time = time.time()

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            if step == 0:
                logger.info(f"Input batch shape: {batch['input_ids'].shape}")

            batch = {k: v.to(device) for k, v in batch.items()}

            # torch.cuda.synchronize(device)
            t0 = time.time()
            outputs = model(**batch)
            loss = outputs.loss
            # torch.cuda.synchronize(device)
            t1 = time.time()

            optimizer.zero_grad()
            loss.backward()
            # torch.cuda.synchronize(device)
            t2 = time.time()

            optimizer.step()
            # torch.cuda.synchronize(device)
            t3 = time.time()

            if step % 10 == 0:
                logger.info(
                    f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f} | "
                    f"fwd {t1 - t0:.3f}s | bwd {t2 - t1:.3f}s | opt {t3 - t2:.3f}s | "
                    f"step {t3 - t0:.3f}s"
                )

    # ────────────────────────────────────────────────────────────── metrics ────
    elapsed = time.time() - start_time
    max_alloc = torch.cuda.max_memory_allocated(device) / 1024 ** 2  # MB
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 2
    logger.info(
        f"Finished – total wall‑clock {elapsed:.1f}s | "
        f"GPU peak alloc {max_alloc:.1f} MB | reserved {max_reserved:.1f} MB"
    )
    
    logger.info("We've done from training \U0001f604 \U0001F389 \U0001F389 \U0001F389 \U0001F389 \U0001F389 \U0001F389")
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--use_lora", action="store_true", default=False)
    args = parser.parse_args()

    train(args)
