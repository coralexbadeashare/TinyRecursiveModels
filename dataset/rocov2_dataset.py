import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable
import torchvision.transforms as transforms
from pydantic import BaseModel

class ROCOv2DatasetConfig(BaseModel):
    dataset_path: str
    split: str = "train"
    image_size: int = 224
    max_seq_len: int = 64
    rank: int = 0
    num_replicas: int = 1

class ROCOv2DatasetMetadata(BaseModel):
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int = 0
    total_groups: int = 1
    mean_puzzle_examples: int = 1
    sets: list[str] = ["all"]
    pad_id: int = 0
    ignore_label_id: int = -100

class ROCOv2Dataset(Dataset):
    def __init__(self, config: ROCOv2DatasetConfig, tokenizer: Optional[Callable] = None):
        self.config = config
        self.split_path = os.path.join(config.dataset_path, config.split)
        
        # Find all images
        self.image_files = [f for f in os.listdir(self.split_path) if f.endswith('.jpg')]
        self.image_files.sort() # Ensure deterministic order
        
        # Handle distributed training (sharding)
        if config.num_replicas > 1:
            total_size = len(self.image_files)
            per_replica = total_size // config.num_replicas
            start = config.rank * per_replica
            end = start + per_replica if config.rank < config.num_replicas - 1 else total_size
            self.image_files = self.image_files[start:end]

        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Simple tokenizer if none provided (character level for simplicity in this demo, 
        # but ideally should use a proper tokenizer like BPE or WordPiece)
        # For this demo, let's assume a simple character-level tokenizer or a provided one.
        # If we want to be compatible with HRM's expected inputs, we need integer tokens.
        
        # Let's build a simple vocab from the data if possible, or use a fixed one.
        # For simplicity, let's use ASCII + some specials.
        self.vocab_size = 256 # ASCII
        self.pad_id = 0
        self.eos_id = 1
        self.bos_id = 2
        
        self.metadata = ROCOv2DatasetMetadata(
            vocab_size=self.vocab_size,
            seq_len=config.max_seq_len,
            sets=[config.split]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.split_path, img_filename)
        txt_path = img_path + ".txt"

        # Load Image
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image
            image = torch.zeros((3, self.config.image_size, self.config.image_size))

        # Load Caption
        caption = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()

        # Tokenize
        # Simple ASCII encoding for demo, filtering non-ASCII
        tokens = [self.bos_id] + [ord(c) for c in caption[:self.config.max_seq_len-2] if ord(c) < 256] + [self.eos_id]
        
        # Pad
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        padding_len = self.config.max_seq_len - len(input_ids)
        if padding_len > 0:
            input_ids = input_ids + [self.pad_id] * padding_len
            labels = labels + [-100] * padding_len # Ignore loss for padding
        
        return {
            "inputs": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "images": image,
            # HRM expects these keys often
            "puzzle_identifiers": torch.tensor([0], dtype=torch.long), 
        }

def create_rocov2_dataloader(config: ROCOv2DatasetConfig, **kwargs):
    dataset = ROCOv2Dataset(config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None, # Dataset handles batching? No, standard PyTorch dataset returns single item.
        # Wait, HRM's PuzzleDataset returns batches? 
        # Looking at PuzzleDataset in pretrain.py: "batch_size=None" in DataLoader suggests the dataset might return batches or it's using a custom sampler/collate.
        # Actually, PuzzleDataset usually loads pre-batched npy files.
        # Here we are loading individual files. So we should let DataLoader handle batching.
        # But pretrain.py expects `batch` to be yielded directly.
        # Let's check `pretrain.py` loop: `for set_name, batch, global_batch_size in train_loader:`
        # It seems the loader yields (set_name, batch, size).
        # We need to wrap our standard DataLoader to match this signature.
        **kwargs
    )
    return dataloader, dataset.metadata

class ROCOv2DataLoaderWrapper:
    def __init__(self, dataloader, set_name, global_batch_size):
        self.dataloader = dataloader
        self.set_name = set_name
        self.global_batch_size = global_batch_size

    def __iter__(self):
        for batch in self.dataloader:
            # Batch is a dict of stacked tensors
            yield self.set_name, batch, self.global_batch_size

    def __len__(self):
        return len(self.dataloader)
