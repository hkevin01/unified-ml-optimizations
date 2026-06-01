# =============================================================================
# FILE: src/models/base_model.py
# PURPOSE: Transformer classifier unifying all three optimization categories.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.low_rank_layer import LowRankAttention, LowRankLinear
from src.models.fused_kernel import FusedLinearGELU, QuantizedLinear
from src.utils.config import ModelConfig


class TransformerBlock(nn.Module):
    """
    ID: MODEL-TB-001
    Requirement: Single transformer block with low-rank attention and fused FFN.
    Purpose: Compose structural (low-rank) + systems (fused kernel) optimizations
             into one reusable block.
    Inputs:
      - d_model (int): model dimension.
      - n_heads (int): attention heads.
      - rank (int): low-rank bottleneck for Q/K/V.
      - dropout (float): dropout rate.
    Outputs: Tensor of shape (B, L, d_model) - same shape as input.
    Preconditions: d_model % n_heads == 0; rank >= 1.
    Postconditions: Output shape == input shape; residual connections applied.
    Assumptions: Layer norm applied pre-residual (Pre-LN) for training stability.
    Side Effects: None.
    Failure Modes: OOM on large B*L with full attention; use gradient checkpointing.
    Error Handling: None; caller manages memory budget.
    Constraints: None.
    Verification: Shape test; gradient norm test.
    References: Pre-LN Transformer (Xiong et al., 2020).
    """

    def __init__(self, d_model: int, n_heads: int, rank: int, dropout: float = 0.1) -> None:
        super().__init__()
        # --- Structural optimization: Low-rank attention (Category A) ---
        self.attn = LowRankAttention(d_model, n_heads, rank, dropout)
        # --- Systems optimization: Fused FFN with Triton kernel (Category B) ---
        ffn_dim = d_model * 4
        self.ffn_fc1 = FusedLinearGELU(d_model, ffn_dim)
        self.ffn_fc2 = nn.Linear(ffn_dim, d_model)
        # Pre-LN normalization layers
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ID: MODEL-TB-FWD-001
        Requirement: Pre-LN residual block: attn + ffn with residual connections.
        Purpose: Standard transformer forward with structural optimizations baked in.
        Inputs: x - (B, L, d_model).
        Outputs: (B, L, d_model).
        Preconditions: x.shape[-1] == d_model.
        Postconditions: Output shape == input shape.
        Assumptions: None.
        Side Effects: None.
        Failure Modes: NaN propagation if LayerNorm receives NaN input.
        Error Handling: Caller should clip gradients to prevent NaN.
        Constraints: None.
        Verification: Single step loss decreases; attention outputs non-NaN.
        References: None.
        """
        # Attention sub-layer with Pre-LN
        x = x + self.dropout(self.attn(self.ln1(x)))
        # FFN sub-layer with Pre-LN (fused GELU inside ffn_fc1)
        x = x + self.dropout(self.ffn_fc2(self.ffn_fc1(self.ln2(x))))
        return x


class OptimizedTransformer(nn.Module):
    """
    ID: MODEL-OT-001
    Requirement: Full sequence classifier combining all three optimization categories:
                 (A) low-rank attention projections,
                 (B) Triton-fused FFN + quantized output head,
                 (C) compatible with AdamW + cosine LR schedule + grad clipping.
    Purpose: Unified demonstration model for the optimization project.
    Inputs (constructor):
      - cfg (ModelConfig): model hyperparameters.
    Forward inputs:
      - input_ids: (B, L) long tensor of token indices.
    Outputs: (B, num_classes) logit tensor.
    Preconditions: input_ids values in [0, vocab_size).
    Postconditions: Output shape is (B, num_classes).
    Assumptions: CLS-token pooling via mean over sequence dimension.
    Side Effects: None.
    Failure Modes: Out-of-vocab indices -> embedding lookup OOB error.
    Error Handling: Clip or clamp input_ids if needed.
    Constraints: None.
    Verification: Single-step overfit test on batch of 2 samples.
    References: BERT (Devlin et al., 2019) for CLS pooling approach.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Token embedding + sinusoidal positional encoding
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        # Transformer layers (each with low-rank attn + fused FFN)
        self.layers = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.rank, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        self.ln_final = nn.LayerNorm(cfg.d_model)

        # --- Category B: QuantizedLinear classifier head ---
        self.classifier = QuantizedLinear(cfg.d_model, cfg.num_classes, bits=8)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        ID: MODEL-OT-INIT-001
        Requirement: Initialize embedding and projection weights consistently.
        Purpose: Stable training start; avoid initial saturation.
        Inputs: None.
        Outputs: None (modifies parameters in-place).
        Preconditions: All sub-modules allocated.
        Postconditions: Embeddings ~ N(0, 0.02); linear biases = 0.
        Assumptions: None.
        Side Effects: Overwrites default PyTorch initialization.
        Failure Modes: None.
        Error Handling: None.
        Constraints: None.
        Verification: Check parameter stats after construction.
        References: GPT-2 weight initialization (Radford et al., 2019).
        """
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        ID: MODEL-OT-FWD-001
        Requirement: Full forward pass: embed -> transformer blocks -> classify.
        Purpose: End-to-end inference path for training and evaluation.
        Inputs: input_ids - (B, L) long tensor, values in [0, vocab_size).
        Outputs: (B, num_classes) logit tensor.
        Preconditions: input_ids.shape[1] <= cfg.seq_len.
        Postconditions: Output is a real-valued tensor; no softmax applied.
        Assumptions: Positional embeddings indexed 0..L-1.
        Side Effects: None.
        Failure Modes: L > cfg.seq_len -> pos_embedding out of range.
        Error Handling: Truncate sequence at cfg.seq_len before calling.
        Constraints: None.
        Verification: Loss decreases on overfit test; output finite.
        References: None.
        """
        B, L = input_ids.shape
        # Clamp sequence to max configured length
        L = min(L, self.cfg.seq_len)
        input_ids = input_ids[:, :L]

        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.emb_dropout(self.embedding(input_ids) + self.pos_embedding(positions))

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        # Mean pooling over sequence dimension (CLS approximation)
        x = x.mean(dim=1)
        return self.classifier(x)
