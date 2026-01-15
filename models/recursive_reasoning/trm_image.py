from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import torchvision.models as models
import copy

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1Carry
)

class TRMImageConfig(TinyRecursiveReasoningModel_ACTV1Config):
    vision_backbone: str = "resnet18"
    freeze_vision: bool = True
    causal: bool = True

@dataclass
class TRMImageInnerCarry(TinyRecursiveReasoningModel_ACTV1InnerCarry):
    pass # Inherit existing structure

class TRMImage_Inner(nn.Module):
    def __init__(self, config: TRMImageConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Vision Encoder
        if self.config.vision_backbone == "resnet18":
            self.vision_model = models.resnet18(pretrained=True)
            self.vision_dim = self.vision_model.fc.in_features
            self.vision_model.fc = nn.Identity()
        elif self.config.vision_backbone.startswith("swin"):
            from transformers import AutoModel
            model_id = self.config.vision_backbone.split(":", 1)[1] if ":" in self.config.vision_backbone else "microsoft/swin-tiny-patch4-window7-224"
            print(f"Loading Swin model: {model_id}")
            self.vision_model = AutoModel.from_pretrained(model_id)
            self.vision_dim = self.vision_model.config.hidden_size
        elif self.config.vision_backbone.startswith("fuselip"):
            # Dynamic import for FuseLIP
            import sys
            FUSELIP_SRC = "/workspace/fuselip/src"
            if FUSELIP_SRC not in sys.path:
                sys.path.insert(0, FUSELIP_SRC)
            try:
                from fuse_clip.fuse_clip_utils import load_model
            except ImportError:
                # Try relative path
                FUSELIP_SRC_ALT = "/mnt/QNAP/bacor/fuselip/src"
                if FUSELIP_SRC_ALT not in sys.path:
                    sys.path.insert(0, FUSELIP_SRC_ALT)
                from fuse_clip.fuse_clip_utils import load_model

            model_id = self.config.vision_backbone.split(":", 1)[1] if ":" in self.config.vision_backbone else "chs20/FuseLIP-B-CC3M-MM"
            print(f"Loading FuseLIP model: {model_id}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.vision_model, self.vision_processor, _ = load_model(model_id, device=device)
            
            # Infer dim
            if hasattr(self.vision_model, "visual"):
                 self.vision_dim = self.vision_model.visual.output_dim
            elif hasattr(self.vision_model, "vision_model"):
                self.vision_dim = self.vision_model.vision_model.config.hidden_size
            else:
                self.vision_dim = 512
        else:
            raise ValueError(f"Unknown vision backbone: {self.config.vision_backbone}")

        if self.config.freeze_vision:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        self.vision_proj = CastedLinear(self.vision_dim, self.config.hidden_size, bias=False)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5) 

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor], images: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Vision embedding
        if hasattr(self.vision_model, "encode_image"):
             # FuseLIP / CLIP
             vision_device = next(self.vision_model.parameters()).device
             images = images.to(vision_device)
             vision_features = self.vision_model.encode_image(images)
             if vision_features.ndim > 2:
                 vision_features = vision_features.mean(dim=1)
        elif self.config.vision_backbone.startswith("swin"):
             vision_device = next(self.vision_model.parameters()).device
             images = images.to(vision_device)
             outputs = self.vision_model(images)
             if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                 vision_features = outputs.pooler_output
             else:
                 vision_features = outputs.last_hidden_state.mean(dim=1)
        else:
             vision_features = self.vision_model(images)
             
        vision_emb = self.vision_proj(vision_features)
        
        # Add vision embedding to all tokens
        embedding = embedding + vision_emb.unsqueeze(1).to(embedding.dtype)

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0 and puzzle_identifiers is not None:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: torch.device = torch.device('cpu')):
        return TRMImageInnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMImageInnerCarry):
        return TRMImageInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TRMImageInnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TRMImageInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch.get("puzzle_identifiers"), batch["images"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TRMImageInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TRMImage(nn.Module):
    """TRM Image Captioning wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRMImageConfig(**config_dict)
        self.inner = TRMImage_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size, device), 
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Need to ensure inner_carry is on device if empty?
        # Typically PyTorch handles this if operations are mixed with on-device tensors.
        
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            # Training ACT logic
            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)[2]
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
