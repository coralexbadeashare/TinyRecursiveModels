
import torch
import os
from torch.utils.data import DataLoader
from dataset.rocov2_dataset import ROCOv2Dataset, ROCOv2DatasetConfig
from models.recursive_reasoning.trm_image import TRMImage
from models.losses import ACTLossHead


print("Script starting...")
def main():
    checkpoint_path = "checkpoints/trm_fuselip/final_checkpoint.pt"
    
    # Dataset
    ds_config = ROCOv2DatasetConfig(
        dataset_path="/workspace/data/ROCOv2",
        split="test",
        rank=0,
        num_replicas=1,
        image_size=256, 
        max_seq_len=512
    )
    dataset = ROCOv2Dataset(ds_config)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Model Config (Validating same config as training)
    model_cfg = dict(
        batch_size=1,
        seq_len=512,
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
        vision_backbone="fuselip:chs20/FuseLIP-B-CC3M-MM",
        freeze_vision=True
    )
    
    model = TRMImage(model_cfg)
    model = ACTLossHead(model, loss_type="softmax_cross_entropy")
    sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(sd, strict=False) # Allow missing keys if wrapper
    model = model.cuda()
    model.eval()
    
    print("--- DEBUG GENERATION ---")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5: break
            
            images = batch["images"].cuda()
            # Access inner model through wrapper
            if hasattr(model.model.inner, "vision_model"):
                vd = next(model.model.inner.vision_model.parameters()).device
                images = images.to(vd)
            
            # Simple greedy generation
            gen_toks = [2] # BOS
            for _ in range(50):
                seq_len = len(gen_toks)
                padded_inp = torch.full((1, 512), 0, dtype=torch.long).cuda()
                padded_inp[0, :seq_len] = torch.tensor(gen_toks).cuda()
                
                bat = {
                    "inputs": padded_inp,
                    "images": images,
                    "puzzle_identifiers": torch.tensor([0]).cuda(),
                    "labels": padded_inp,
                }
                
                # Reset carry for each step properly? 
                # Model usually handles state, but for AR we need fresh carry or re-use?
                # TRM is stateless between supervision steps usually, but here we generate token by token.
                # initial_carry creates fresh zero state.
                carry = model.initial_carry(bat) 
                
                carry, _, _, outputs, _ = model(carry=carry, batch=bat, return_keys=["logits"])
                next_tok = torch.argmax(outputs["logits"][0, seq_len-1]).item()
                if next_tok == 1: break # EOS
                gen_toks.append(next_tok)
            
            pred_txt = "".join([chr(t) for t in gen_toks if t > 2])
            
            # Reference
            ref_ids = batch["inputs"][0].tolist()
            ref_txt = "".join([chr(t) for t in ref_ids if t > 2])
            
            print(f"Example {i}:")
            print(f"  Ref:  {ref_txt[:100]}...")
            print(f"  Pred: {pred_txt}")
            print(f"  Pred IDs: {gen_toks}")
if __name__ == "__main__":
    main()
