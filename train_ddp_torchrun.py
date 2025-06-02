
import os, torch, torch.distributed as dist
if dist.is_available() and not dist.is_initialized():
    dist.init_process_group(backend="nccl", init_method="env://")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

from unsloth import FastLanguageModel
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os, torch, torch.distributed as dist     # import torch first


def setup_logger(rank):
    """Set up a pretty logger that only logs from the main process."""
    logger = logging.getLogger()
    logger.handlers.clear()  # Remove possible duplicate handlers
    # if rank == 0:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s][%(process)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # else:
    #     logger.addHandler(logging.NullHandler())  # No output for non-main ranks
    return logger

def setup():
    # torchrun sets these environment variables
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    # dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, world_size, rank

def cleanup():
    dist.destroy_process_group()

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __len__(self):
        return 1000
    def __getitem__(self, idx):
        enc = self.tokenizer("Hello world!", return_tensors="pt", padding='max_length', padding_side='right', max_length=100)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0),
        }

def train(args):
    local_rank, world_size, rank = setup()
    logger = setup_logger(rank)
    logger.info(f"Starting process rank={rank}, local_rank={local_rank}, world_size={world_size}")
    
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # if dist.is_initialized():
    #     dist.barrier()
    if rank == 0:
        model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = 100,
        load_in_4bit = False,
        load_in_8bit=False,
        full_finetuning=True
        )
    dist.barrier()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=100,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=True,
        skip_compilation=True  # Skip compilation on all ranks
    )
    model.to(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    dataset = MyDataset(tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for index, batch in enumerate(loader):
            batch = {key:batch[key].to(rank) for key in batch.keys()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if rank == 0 and index % 10 ==0:
                logger.info(f"Epoch {epoch} | Batch {index} | Loss: {loss.item():.4f}")
    
    logger.info("We've done from training \U0001f604 \U0001F389 \U0001F389 \U0001F389 \U0001F389 \U0001F389 \U0001F389")
    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    train(args)
