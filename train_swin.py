from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass
import os
import math
import shutil
import json
import tqdm
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
import pydantic
from adam_atan2 import AdamATan2

# Import new dataset and model
from dataset.rocov2_dataset import ROCOv2Dataset, ROCOv2DatasetConfig, ROCOv2DatasetMetadata, ROCOv2DataLoaderWrapper
from models.recursive_reasoning.trm_image import TRMImage
from models.losses import ACTLossHead

# Metric imports
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def compute_metrics(refs: Dict[str, List[str]], preds: Dict[str, List[str]]):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    val_res = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, preds)
        if isinstance(method, list):
            for m, s in zip(method, score):
                val_res[m] = s
        else:
            val_res[method] = score
    return val_res

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str = "models.recursive_reasoning.trm_image.TRMImage"
    loss: LossConfig
    model_extra: dict = {}

class TrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str = "/workspace/data/ROCOv2"

    # Hyperparams
    global_batch_size: int = 32
    epochs: int = 50

    lr: float = 1e-4
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 1000

    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95

    # Names
    checkpoint_path: Optional[str] = "checkpoints/trm_swin"

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int

def cosine_schedule_with_warmup_lr_lambda(current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))

def compute_lr(base_lr: float, config: TrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )

def create_dataloader(config: TrainConfig, split: str, rank: int, world_size: int):
    ds_config = ROCOv2DatasetConfig(
        dataset_path=config.data_path,
        split=split,
        rank=rank,
        num_replicas=world_size,
        image_size=224, 
        max_seq_len=512
    )
    dataset = ROCOv2Dataset(ds_config)
    
    loader = DataLoader(
        dataset,
        batch_size=config.global_batch_size // world_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    wrapper = ROCOv2DataLoaderWrapper(loader, split, config.global_batch_size)
    return wrapper, dataset.metadata

def create_model(config: TrainConfig, train_metadata: ROCOv2DatasetMetadata, world_size: int):
    model_cfg = dict(
        batch_size=config.global_batch_size // world_size,
        seq_len=train_metadata.seq_len,
        vocab_size=train_metadata.vocab_size,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        
        # TRM specific
        H_cycles=1,
        L_cycles=1,
        H_layers=2,
        L_layers=2,
        hidden_size=512,
        expansion=4.0,
        num_heads=8,
        pos_encodings="rope",
        halt_max_steps=4,
        halt_exploration_prob=0.05,
        
        vision_backbone="swin:microsoft/swin-tiny-patch4-window7-224",
        freeze_vision=True
    )
    
    model = TRMImage(model_cfg)
    model = ACTLossHead(model, loss_type="softmax_cross_entropy")
    model = model.cuda()
    
    optimizers = [
        AdamATan2(
            model.parameters(),
            lr=0, 
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    ]
    optimizer_lrs = [config.lr]

    return model, optimizers, optimizer_lrs

def init_train_state(config: TrainConfig, train_metadata: ROCOv2DatasetMetadata, world_size: int):
    # Temp ds for len
    temp_ds = ROCOv2Dataset(ROCOv2DatasetConfig(dataset_path=config.data_path, split="train"))
    num_samples = len(temp_ds)
    steps_per_epoch = num_samples // config.global_batch_size
    total_steps = config.epochs * steps_per_epoch

    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )

def train_batch(config: TrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    batch = {k: v.cuda() for k, v in batch.items()}

    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])
    ((1 / global_batch_size) * loss).backward()

    # Clip Gradients
    torch.nn.utils.clip_grad_norm_(train_state.model.parameters(), 1.0)

    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()

    if len(metrics):
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if rank == 0:
            metric_values = metric_values.cpu().detach().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            count = max(reduced_metrics.get("count", 1), 1)
            reduced_metrics = {f"train/{k}": float(v / (global_batch_size if k.endswith("loss") else count)) for k, v in reduced_metrics.items()}
            reduced_metrics["train/lr"] = float(lr_this_step)
            return reduced_metrics

def main():
    RANK = 0
    WORLD_SIZE = 1
    
    config = TrainConfig(
        arch=ArchConfig(loss=LossConfig(name="models.losses.ACTLossHead"))
    )
    
    train_loader, train_metadata = create_dataloader(config, "train", RANK, WORLD_SIZE)
    train_state = init_train_state(config, train_metadata, WORLD_SIZE)

    print("Starting TRM Image training (Swin)...")
    all_metrics = []
    
    for epoch in range(config.epochs):
        train_state.model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        epoch_metrics = {}
        batch_count = 0
        
        for set_name, batch, global_batch_size in pbar:
            metrics = train_batch(config, train_state, batch, global_batch_size, RANK, WORLD_SIZE)
            
            if metrics:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                batch_count += 1
                pbar.set_postfix({"loss": f"{metrics.get('train/lm_loss', 0):.4f}"})

        if batch_count > 0:
            avg_metrics = {k: v / batch_count for k, v in epoch_metrics.items()}
            avg_metrics['step'] = train_state.step
            avg_metrics['epoch'] = epoch
            
            # --- Evaluation Step ---
            if RANK == 0:
                print(f"Running evaluation for Epoch {epoch}...")
                train_state.model.eval()
                refs = {}
                preds = {}
                eval_count = 0
                max_eval = 20
                with torch.no_grad():
                   for _, batch, _ in train_loader: 
                       if eval_count >= max_eval: break
                       images = batch["images"].cuda()
                       # Check wrapper nesting: ACTLossHead -> TRMImage -> TRMImage_Inner
                       # train_state.model is ACT, .model is TRMImage
                       if hasattr(train_state.model.model.inner, "vision_model"):
                            vd = next(train_state.model.model.inner.vision_model.parameters()).device
                            images = images.to(vd)
                       
                       curr_batch_size = images.shape[0]
                       for b in range(curr_batch_size):
                           if eval_count >= max_eval: break
                           img_tensor = images[b].unsqueeze(0)
                           bos_id = 2
                           eos_id = 1
                           pad_id = 0
                           gen_toks = [bos_id]
                           for _ in range(64):
                               seq_len = len(gen_toks)
                               padded_inp = torch.full((1, 512), pad_id, dtype=torch.long).cuda()
                               padded_inp[0, :seq_len] = torch.tensor(gen_toks).cuda()
                               bat = {
                                   "inputs": padded_inp,
                                   "images": img_tensor,
                                   "puzzle_identifiers": torch.tensor([0]).cuda(),
                                   "labels": padded_inp.clone()
                               }
                               carry = train_state.model.initial_carry(bat)
                               while True:
                                   carry, _, _, outputs, fin = train_state.model(carry=carry, batch=bat, return_keys=["logits"])
                                   if fin: break
                               next_tok = torch.argmax(outputs["logits"][0, seq_len-1]).item()
                               if next_tok == eos_id: break
                               gen_toks.append(next_tok)
                           pred_txt = "".join([chr(t) for t in gen_toks if t not in [bos_id, eos_id, pad_id]])
                           preds[str(eval_count)] = [pred_txt]
                           ref_ids = batch["inputs"][b].cpu().tolist()
                           ref_txt = "".join([chr(t) for t in ref_ids if t > 2])
                           refs[str(eval_count)] = [ref_txt]
                           eval_count += 1
                try:
                    scores = compute_metrics(refs, preds)
                    for m, s in scores.items():
                        avg_metrics[f"val/{m}"] = s
                except Exception as e:
                    print(f"Error computing metrics: {e}")

            all_metrics.append(avg_metrics)
            print(f"Epoch {epoch+1} Average: {avg_metrics}")

        # Save
        os.makedirs(config.checkpoint_path, exist_ok=True)
        torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"epoch_{epoch}.pt"))
        torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, "final_checkpoint.pt"))
        with open(os.path.join(config.checkpoint_path, "metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)

if __name__ == "__main__":
    main()
