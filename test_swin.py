from typing import Optional, Any, Sequence, List, Dict
import os
import json
import torch
import tqdm
from torch.utils.data import DataLoader

from dataset.rocov2_dataset import ROCOv2Dataset, ROCOv2DatasetConfig, ROCOv2DatasetMetadata
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

def load_model(checkpoint_path: str):
    return None

def main():
    checkpoint_dir = "checkpoints/trm_swin"
    checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        # check epoch
        latest_epoch = -1
        for f in os.listdir(checkpoint_dir) if os.path.exists(checkpoint_dir) else []:
            if f.startswith("epoch_") and f.endswith(".pt"):
                try:
                    e = int(f.split("_")[1].split(".")[0])
                    if e > latest_epoch: latest_epoch = e
                except: pass
        if latest_epoch >= 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{latest_epoch}.pt")
        else:
            print("No checkpoint found.")
            return

    # Dataset
    data_path = "/workspace/data/ROCOv2"
    ds_config = ROCOv2DatasetConfig(
        dataset_path=data_path,
        split="test",
        rank=0,
        num_replicas=1,
        image_size=224, 
        max_seq_len=512
    )
    dataset = ROCOv2Dataset(ds_config)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Model
    model_cfg = dict(
        batch_size=1,
        seq_len=dataset.metadata.seq_len,
        vocab_size=dataset.metadata.vocab_size,
        num_puzzle_identifiers=dataset.metadata.num_puzzle_identifiers,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        
        H_cycles=1,
        L_cycles=1,
        H_layers=2,
        L_layers=2,
        hidden_size=512,
        expansion=4.0,
        num_heads=8,
        pos_encodings="rope",
        halt_max_steps=4,
        halt_exploration_prob=0.0,
        
        vision_backbone="swin:microsoft/swin-tiny-patch4-window7-224",
        freeze_vision=True
    )
    
    model = TRMImage(model_cfg)
    model = ACTLossHead(model, loss_type="softmax_cross_entropy") 
    
    sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(sd)
    model = model.cuda()
    model.eval()
    
    print("Generating captions...")
    refs = {}
    preds = {}
    eval_count = 0
    max_eval = 200 # Limit for speed
    
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(loader), total=min(len(loader), max_eval)):
            if eval_count >= max_eval: break
            
            images = batch["images"].cuda()
            
            if hasattr(model.model.inner, "vision_model"):
                vd = next(model.model.inner.vision_model.parameters()).device
                images = images.to(vd)
            
            img_tensor = images
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
                
                carry = model.initial_carry(bat) # implicit device?
                
                carry, _, _, outputs, fin = model(carry=carry, batch=bat, return_keys=["logits"])
                
                next_tok_logits = outputs["logits"][0, seq_len-1]
                next_tok = torch.argmax(next_tok_logits).item()
                
                if next_tok == eos_id: break
                gen_toks.append(next_tok)

            pred_txt = "".join([chr(t) for t in gen_toks if t not in [bos_id, eos_id, pad_id]])
            preds[str(eval_count)] = [pred_txt]
            
            ref_ids = batch["inputs"][0].tolist()
            ref_txt = "".join([chr(t) for t in ref_ids if t > 2]) 
            refs[str(eval_count)] = [ref_txt]
            
            eval_count += 1
            
    scores = compute_metrics(refs, preds)
    print("Evaluation Results:", json.dumps(scores, indent=2))
    
    with open(os.path.join(checkpoint_dir, "eval_results.json"), "w") as f:
        json.dump(scores, f, indent=2)

if __name__ == "__main__":
    main()
