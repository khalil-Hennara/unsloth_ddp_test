from unsloth import FastLanguageModel

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import shutil
import time
import random 
import numpy as np


def set_seed():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)


def setup_logger(rank):
    """Set up a logger"""
    logger = logging.getLogger(f"Rank-{rank}")  # Make logger name unique per rank
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f'[%(asctime)s][%(levelname)s][Rank {rank}] %(message)s',  # Use rank from param
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def verify_initial_sync(model, logger):
    """
    Verify that, right after creating the DistributedDataParallel (DDP) wrapper,
    every rank holds exactly the same parameters.

    •  Works with any world-size (early return for single-GPU runs)
    •  Puts tensors on **CPU** before the comparison ⇒ no device mismatch
    •  Uses `torch.no_grad()` and `detach()` to avoid autograd overhead
    •  Logs a single SUCCESS / detailed FAILURE message from rank 0
    """
    # ── Skip if not running DDP ──────────────────────────────────────────────────
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return

    rank        = dist.get_rank()
    world_size  = dist.get_world_size()

    # ── Move the local state-dict to CPU so devices match ───────────────────────
    with torch.no_grad():
        cpu_state_dict = {
            k: v.detach().cpu()           # ❶ cut the autograd graph
            for k, v in model.module.state_dict().items()
        }

    # ── Gather every rank’s state-dict onto *every* rank (cheap: only CPU mem) ──
    all_state_dicts = [None] * world_size
    dist.all_gather_object(all_state_dicts, cpu_state_dict)

    # ── Rank 0 does the heavy comparison and prints a verdict ──────────────────
    if rank == 0:
        logger.info("Verifying initial model synchronisation across all ranks …")
        ref_state = all_state_dicts[0]

        for r in range(1, world_size):
            for key, ref_tensor in ref_state.items():
                other_tensor = all_state_dicts[r][key]
                if not torch.equal(ref_tensor, other_tensor):
                    logger.error(
                        f"Sync FAILED – rank {r} differs on parameter “{key}”."
                    )
                    raise RuntimeError("Initial model weights are NOT synchronised!")

        logger.info("SUCCESS – initial model weights are identical on all ranks.")

    # ── Make sure everyone waits for rank 0 to finish logging before continuing ─
    dist.barrier()
        
        
def setup():
    # torchrun sets these environment variable
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    # Check if distributed is already initialized (e.g. by deepspeed or other launcher)
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    # Log device setup
    # We need a basic logger here before the full one is setup if we want to log this part
    if rank == 0:
        print(
            f"[Rank {rank}] Setup: local_rank={local_rank}, world_size={world_size}, rank={rank}, cuda_device_set_to={local_rank}")

    return local_rank, world_size, rank


def cleanup():
    dist.destroy_process_group()


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        enc = self.tokenizer("Hello world!", return_tensors="pt", padding='max_length', padding_side='right',
                             max_length=4096)
        return {

            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0),
        }


def train(args):
    
    set_seed()
    
    local_rank, world_size, rank = setup()

    current_device = torch.device(f"cuda:{local_rank}")

    logger = setup_logger(rank)
    logger.info("Starting training function.")

    current_device = torch.device(f"cuda:{local_rank}")
    logger.info(f"Current device for model loading: {current_device}")

    # Barrier before model loading
    if world_size > 1:
        logger.info("Barrier before model loading / Unsloth compilation.")
        dist.barrier()

    logger.info("Loading model with FastLanguageModel.from_pretrained...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False if args.use_lora else True,
        use_gradient_checkpointing=True,
        device_map={"": "cuda"},
        disable_log_stats=False,
        # Enable compilation but make it distributed-safe
        # trust_remote_code=False,
    )
    logger.info("Model and tokenizer loaded.")

    if args.use_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", ],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing=True,  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
    if rank == 0 and args.use_lora:
        # log on the main process only
        total_trainable = 0
        total_frozen = 0
        for p in model.parameters():
            if p.requires_grad:
                total_trainable += p.numel()
            else:
                total_frozen += p.numel()

        logger.info(
            f"Model parameter count ⇒ "
            f"trainable: {total_trainable:,}  |  frozen: {total_frozen:,}  "
            f"({100 * total_trainable / (total_trainable + total_frozen):.2f}% trainable)"
        )

    # Barrier after model loading / Unsloth compilation
    if world_size > 1:
        logger.info("Barrier after model loading / Unsloth compilation.")
        dist.barrier()
    if next(model.parameters()).device != current_device:
        model.to(current_device)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # model.to(local_rank)
    model.to(current_device)
    logger.info(
        f"Model explicitly moved to {current_device}. Parameter devices: {set(p.device for p in model.parameters())}")

    logger.info("Wrapping model with DistributedDataParallel.")

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    logger.info(f"Model dtype {model.type}")
    logger.info(f"Model device: {model.device}")

    dataset = MyDataset(tokenizer)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    torch.cuda.reset_peak_memory_stats(current_device)

    model.train()
    if rank == 0:
        start = time.time()

    for epoch in range(args.epochs):

        sampler.set_epoch(epoch)

        for index, batch in enumerate(loader):
            if index == 0:
                logger.info(f"The input length is {batch['input_ids'].shape}")

            batch = {key: batch[key].to(rank) for key in batch.keys()}

            torch.cuda.synchronize(local_rank)
            t0 = time.time()

            outputs = model(**batch)

            loss = outputs.loss

            torch.cuda.synchronize(local_rank)
            t1 = time.time()

            optimizer.zero_grad()

            loss.backward()
            torch.cuda.synchronize(local_rank)
            t2 = time.time()

            optimizer.step()
            torch.cuda.synchronize(local_rank)
            t3 = time.time()

            if rank == 0 and index % 10 == 0:
                logger.info(f"Epoch {epoch} | Batch {index} | Loss: {loss.item():.4f}")
                logger.info(
                    f"Bt {index:04d} | "
                    f"fwd {t1 - t0:.3f}s  bwd {t2 - t1:.3f}s  opt {t3 - t2:.3f}s  "
                    f"step {t3 - t0:.3f}s"
                )

    if rank == 0:
        end = time.time() - start
        logger.info(f"The estimated time on one GPU is {end:.4} s")

        elapsed = time.time() - start
        max_alloc = torch.cuda.max_memory_allocated(current_device) / 1024 ** 2  # MB
        max_reserved = torch.cuda.max_memory_reserved(current_device) / 1024 ** 2

        logger.info(f"Total wall-clock time : {elapsed:.1f} s")
        logger.info(f"GPU peak allocated   : {max_alloc:.1f} MB")
        logger.info(f"GPU peak reserved    : {max_reserved:.1f} MB")

    logger.info("We've done from training \U0001f604 \U0001F389 \U0001F389 \U0001F389 \U0001F389 \U0001F389 \U0001F389")
    cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")

    parser.add_argument('--batch_size', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--use_lora', action='store_true', default=False)

    args = parser.parse_args()

    train(args)