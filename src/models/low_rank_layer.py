# =============================================================================
# FILE: src/models/low_rank_layer.py
# PURPOSE: Low-rank linear layer implementing tensor-algebra optimization.
#          Category: Mathematical & Structural Optimization (Requirement A).
# =============================================================================

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankLinear(nn.Module):
    """
    ID: MODEL-LRL-001
    Requirement: Approximate a full-rank (d_in x d_out) linear projection with
                 two smaller matrices A (d_in x rank) and B (rank x d_out),
                 reducing parameters from d_in*d_out to rank*(d_in+d_out).
    Purpose: Structural compression that reduces FLOPs proportionally to rank/d.
    Inputs:
      - in_features (int): input dimension d_in.
      - out_features (int): output dimension d_out.
      - rank (int): bottleneck rank r << min(d_in, d_out).
      - bias (bool): whether to add a bias term.
      - init_scale (float): scaling factor applied to matrix B at init
                            (keeps initial output near zero, similar to LoRA).
    Outputs: Tensor of shape (..., out_features).
    Preconditions: rank <= min(in_features, out_features).
    Postconditions: Output shape matches full-rank linear layer output shape.
    Assumptions: Called within an nn.Module; gradients flow through A and B.
    Side Effects: Registers parameters A, B, and optionally bias.
    Failure Modes: rank > min(d_in, d_out) -> no compression but still correct.
    Error Handling: Assertion enforces rank >= 1.
    Constraints: No activation inside the layer; caller composes activations.
    Verification: Forward pass shape test; FLOPs comparison vs nn.Linear.
    References: LoRA (Hu et al., 2021); Tucker decomposition literature.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        assert rank >= 1, "rank must be >= 1"
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # A: down-projection  (in_features -> rank)
        self.A = nn.Parameter(torch.empty(in_features, rank))
        # B: up-projection    (rank -> out_features)
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters(init_scale)

    def _reset_parameters(self, init_scale: float) -> None:
        """
        ID: MODEL-LRL-INIT-001
        Requirement: Initialize A with Kaiming uniform; B with near-zero values.
        Purpose: Ensure stable training start and approximate zero output at t=0.
        Inputs: init_scale (float) scaling multiplier for B.
        Outputs: None (modifies self.A and self.B in-place).
        Preconditions: self.A and self.B are allocated.
        Postconditions: self.A values in [-sqrt(6/fan_in), sqrt(6/fan_in)];
                        self.B values near 0.
        Assumptions: Called once in __init__.
        Side Effects: Modifies parameter data in-place.
        Failure Modes: None expected.
        Error Handling: None required.
        Constraints: None.
        Verification: Check weight stats after init.
        References: Kaiming initialization (He et al., 2015).
        """
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.normal_(self.B, std=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ID: MODEL-LRL-FWD-001
        Requirement: Compute x @ A @ B + bias in two sequential matmuls.
        Purpose: Exploit low-rank structure to reduce FLOP count.
        Inputs: x - tensor of shape (..., in_features).
        Outputs: tensor of shape (..., out_features).
        Preconditions: x.shape[-1] == in_features.
        Postconditions: Output shape is x.shape[:-1] + (out_features,).
        Assumptions: A and B are on the same device as x.
        Side Effects: None.
        Failure Modes: Device mismatch raises RuntimeError.
        Error Handling: PyTorch raises descriptive errors on device mismatch.
        Constraints: None.
        Verification: Compare numerical output against nn.Linear with A@B weight.
        References: None.
        """
        # Step 1: down-project  (..., rank)
        h = x @ self.A
        # Step 2: up-project    (..., out_features)
        out = h @ self.B
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}"
        )


class LowRankAttention(nn.Module):
    """
    ID: MODEL-LRA-001
    Requirement: Multi-head attention with low-rank Q/K/V projections.
    Purpose: Reduce attention projection FLOPs by factor of rank/d_model.
    Inputs:
      - d_model (int): model dimension.
      - n_heads (int): number of attention heads.
      - rank (int): low-rank bottleneck for Q/K/V projections.
      - dropout (float): attention dropout probability.
    Outputs: Tensor of shape (B, L, d_model).
    Preconditions: d_model % n_heads == 0; rank >= 1.
    Postconditions: Output shape == input shape.
    Assumptions: Causal masking applied for autoregressive use; optional here.
    Side Effects: None beyond standard attention ops.
    Failure Modes: d_model % n_heads != 0 -> assertion error.
    Error Handling: Assertion in __init__.
    Constraints: None.
    Verification: Shape test; gradient flow test.
    References: Attention Is All You Need (Vaswani et al., 2017);
                Linformer (Wang et al., 2020).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rank: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank = rank
        self.scale = self.d_head ** -0.5

        self.q_proj = LowRankLinear(d_model, d_model, rank, bias=False)
        self.k_proj = LowRankLinear(d_model, d_model, rank, bias=False)
        self.v_proj = LowRankLinear(d_model, d_model, rank, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        ID: MODEL-LRA-FWD-001
        Requirement: Scaled dot-product attention with low-rank projections.
        Purpose: Full attention semantics at reduced projection cost.
        Inputs:
          - x: (B, L, d_model) input tensor.
          - mask: optional (B, 1, L, L) boolean attention mask (True=ignore).
        Outputs: (B, L, d_model) attended tensor.
        Preconditions: x.shape[-1] == d_model.
        Postconditions: Output shape == x.shape.
        Assumptions: Softmax is numerically stable via float32 upcasting.
        Side Effects: None.
        Failure Modes: OOM on very long sequences; use chunked attention.
        Error Handling: None; caller manages sequence length.
        Constraints: None.
        Verification: Attention pattern sum to 1.0 along key dimension.
        References: None.
        """
        B, L, _ = x.shape

        # Low-rank Q, K, V projections
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        attn = self.dropout(attn)

        # Weighted sum and reshape
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)
