import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class MyDataset(torch.utils.data.Dataset):
    # Replace this with your real dataset
    def __init__(self, tokenizer, length=1024):
        self.tokenizer = tokenizer
        self.length = length
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        # Dummy data for demonstration
        enc = self.tokenizer("Hello world!", return_tensors="pt", padding=True,padding_side='right', max_length=256)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0),
        }

def train(rank, world_size, args):
    setup(rank, world_size)
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Dataset & DataLoader
    dataset = MyDataset(tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = {key:batch[key].to(rank) for key in batch.keys()}
            # input_ids = batch["input_ids"].to(rank)
            # attention_mask = batch["attention_mask"].to(rank)
            # labels = batch["labels"].to(rank)
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if rank == 0:
                print(f"Epoch {epoch} | Loss: {loss.item()}")
    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    assert 2 <= world_size <= 4, "This script is designed for 2 to 4 GPUs."
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
