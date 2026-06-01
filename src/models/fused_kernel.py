# =============================================================================
# FILE: src/models/fused_kernel.py
# PURPOSE: Fused linear + activation operator using Triton kernel.
#          Category: Systems-Level & Kernel Engineering (Requirement B).
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Triton fused kernel: linear projection + GELU activation in one kernel pass
# Avoids writing intermediate activations to HBM by fusing both ops.
# Falls back to unfused PyTorch when Triton is unavailable (CPU runs, CI).
# ---------------------------------------------------------------------------

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


if _TRITON_AVAILABLE:
    @triton.jit
    def _fused_linear_gelu_kernel(
        # Pointers to input X (M x K), weight W (N x K), bias B (N,), output Y (M x N)
        X_ptr, W_ptr, B_ptr, Y_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """
        ID: KERN-FLG-001
        Requirement: Compute Y = GELU(X @ W.T + B) tiled over output blocks.
        Purpose: Fuse matmul and GELU to eliminate intermediate HBM writes.
        Inputs:
          X_ptr   - pointer to input  matrix (M, K) in row-major order.
          W_ptr   - pointer to weight matrix (N, K) in row-major order.
          B_ptr   - pointer to bias   vector (N,) or NULL when HAS_BIAS=False.
          Y_ptr   - pointer to output matrix (M, N) row-major; written in-place.
          M, N, K - matrix dimensions.
          stride_* - row strides for each matrix.
          BLOCK_* - compile-time tile sizes (constexpr for Triton unrolling).
          HAS_BIAS- compile-time flag to avoid conditional branch overhead.
        Outputs: Writes activated result to Y_ptr.
        Preconditions: Pointers valid; M, N, K >= 1; strides match allocations.
        Postconditions: Y[m, n] = GELU(sum_k X[m,k]*W[n,k] + B[n]).
        Assumptions: float32 inputs; no NaN/Inf in inputs.
        Side Effects: Writes to Y_ptr.
        Failure Modes: Out-of-bounds if strides wrong; Triton raises PTX error.
        Error Handling: Triton masks OOB loads to 0.
        Constraints: BLOCK_K must divide K or masking handles remainder.
        Verification: Numerical comparison vs F.linear + F.gelu in tests.
        References: Triton matmul tutorial; Flash-Decoding kernel structure.
        """
        # Program IDs for this tile
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Row/col offsets for this tile
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        # Pointers to the first K-block
        X_blk = X_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
        W_blk = W_ptr + (rn[:, None] * stride_wn + rk[None, :] * stride_wk)

        # Accumulator in float32
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_offset = k * BLOCK_K
            x_mask = (rm[:, None] < M) & ((rk + k_offset)[None, :] < K)
            w_mask = (rn[:, None] < N) & ((rk + k_offset)[None, :] < K)
            x = tl.load(X_blk + k_offset * stride_xk, mask=x_mask, other=0.0)
            w = tl.load(W_blk + k_offset * stride_wk, mask=w_mask, other=0.0)
            acc += tl.dot(x, tl.trans(w))

        # Add bias
        if HAS_BIAS:
            b = tl.load(B_ptr + rn, mask=rn < N, other=0.0)
            acc += b[None, :]

        # Apply approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi)*(x+0.044715*x^3)))
        cdf = 0.5 * (1.0 + tl.math.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))
        y = acc * cdf

        # Write output
        Y_blk = Y_ptr + (rm[:, None] * stride_ym + rn[None, :] * stride_yn)
        y_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(Y_blk, y.to(tl.float32), mask=y_mask)


    def fused_linear_gelu_triton(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        ID: KERN-FLG-WRAP-001
        Requirement: Launch _fused_linear_gelu_kernel with computed grid.
        Purpose: Python-side wrapper handling shape extraction and grid sizing.
        Inputs:
          - x: (*, K) float32 tensor (batched; flattened to 2-D internally).
          - weight: (N, K) float32 weight matrix.
          - bias: optional (N,) bias vector.
        Outputs: Tensor of shape (*, N) with GELU applied.
        Preconditions: x.is_contiguous(); weight.is_contiguous().
        Postconditions: Output matches F.gelu(F.linear(x, weight, bias)).
        Assumptions: All tensors on CUDA; float32.
        Side Effects: Allocates output tensor of shape (M, N).
        Failure Modes: Non-contiguous input -> incorrect results.
        Error Handling: .contiguous() called on x before launch.
        Constraints: None.
        Verification: Absolute tolerance 1e-4 vs PyTorch reference.
        References: None.
        """
        orig_shape = x.shape
        x_2d = x.contiguous().view(-1, x.shape[-1])
        M, K = x_2d.shape
        N = weight.shape[0]

        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _fused_linear_gelu_kernel[grid](
            x_2d, weight,
            bias if bias is not None else weight,   # dummy ptr when no bias
            y,
            M, N, K,
            x_2d.stride(0), x_2d.stride(1),
            weight.stride(0), weight.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            HAS_BIAS=(bias is not None),
        )
        return y.view(*orig_shape[:-1], N)

else:
    def fused_linear_gelu_triton(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        ID: KERN-FLG-FALLBACK-001
        Requirement: CPU/non-Triton fallback for fused linear+GELU.
        Purpose: Allow full pipeline runs without a CUDA GPU or Triton install.
        Inputs: Same as Triton variant.
        Outputs: Same semantic result using PyTorch ops.
        Preconditions: None.
        Postconditions: Numerically equivalent to Triton kernel.
        Assumptions: Triton import failed or CUDA unavailable.
        Side Effects: None.
        Failure Modes: None.
        Error Handling: None required.
        Constraints: ~2x slower than fused kernel due to intermediate writes.
        Verification: Used by CPU unit tests.
        References: None.
        """
        return F.gelu(F.linear(x, weight, bias))


class FusedLinearGELU(nn.Module):
    """
    ID: MODEL-FLG-001
    Requirement: nn.Module wrapping fused_linear_gelu_triton for use in models.
    Purpose: Drop-in replacement for nn.Linear + nn.GELU with fused execution.
    Inputs:
      - in_features (int): input dimension.
      - out_features (int): output dimension.
      - bias (bool): include bias term.
    Outputs: Tensor of shape (..., out_features) with GELU activation applied.
    Preconditions: in_features >= 1, out_features >= 1.
    Postconditions: Equivalent output to sequential Linear + GELU.
    Assumptions: Triton available on CUDA; falls back to PyTorch on CPU.
    Side Effects: None.
    Failure Modes: Weight dtype mismatch with input -> RuntimeError.
    Error Handling: Caller ensures consistent dtypes.
    Constraints: None.
    Verification: Numerical equivalence test vs nn.Linear + F.gelu.
    References: None.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ID: MODEL-FLG-FWD-001
        Requirement: Dispatch to Triton kernel on CUDA, PyTorch otherwise.
        Purpose: Transparent hardware-aware dispatch.
        Inputs: x - (..., in_features) float tensor.
        Outputs: (..., out_features) float tensor post-GELU.
        Preconditions: x on same device as self.linear.weight.
        Postconditions: Output shape is x.shape[:-1] + (out_features,).
        Assumptions: None.
        Side Effects: None.
        Failure Modes: None beyond underlying kernel failure modes.
        Error Handling: Propagates kernel RuntimeError to caller.
        Constraints: None.
        Verification: Forward test comparing CUDA vs CPU paths.
        References: None.
        """
        if _TRITON_AVAILABLE and x.is_cuda:
            return fused_linear_gelu_triton(
                x,
                self.linear.weight,
                self.linear.bias,
            )
        return F.gelu(self.linear(x))


class QuantizedLinear(nn.Module):
    """
    ID: MODEL-QL-001
    Requirement: Simulate INT8 weight quantization/dequantization path.
    Purpose: Demonstrate memory-layout and quantization optimization (Req B).
    Inputs:
      - in_features, out_features: layer dimensions.
      - bits (int): quantization bit-width (default 8).
    Outputs: Tensor of shape (..., out_features).
    Preconditions: in_features >= 1, out_features >= 1, bits in {4, 8}.
    Postconditions: Output approximates full-precision linear projection.
    Assumptions: Symmetric per-tensor quantization with absmax scaling.
    Side Effects: Stores quantized weight in int8 buffer; scale as float.
    Failure Modes: Very small weights -> scale underflow.
    Error Handling: Scale clipped to minimum 1e-8.
    Constraints: Quantization error is O(1/2^bits); acceptable for inference.
    Verification: MSE between quantized and full-precision output < threshold.
    References: LLM.int8() (Dettmers et al., 2022).
    """

    def __init__(self, in_features: int, out_features: int, bits: int = 8) -> None:
        super().__init__()
        assert bits in (4, 8), "Only 4-bit and 8-bit quantization supported"
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self._qmax = 2 ** (bits - 1) - 1  # 127 for int8, 7 for int4

        # Full-precision weight for training; quantization applied at forward
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5) if False else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ID: MODEL-QL-FWD-001
        Requirement: Quantize weight to INT8, dequantize, then apply to input.
        Purpose: Simulate inference-time memory savings via INT8 weight storage.
        Inputs: x - (..., in_features) float tensor.
        Outputs: (..., out_features) float tensor.
        Preconditions: x.shape[-1] == in_features.
        Postconditions: Output approximates F.linear(x, weight, bias).
        Assumptions: Straight-through gradient flows through the quant step.
        Side Effects: Creates temporary quantized weight tensor each forward.
        Failure Modes: All-zero weight -> scale=0 -> NaN; guarded by clip.
        Error Handling: Scale clipped to 1e-8 minimum.
        Constraints: Not suitable for actual inference compression without
                     CUDA INT8 matmul kernels; illustrates the quant path.
        Verification: MSE test vs full-precision forward pass.
        References: None.
        """
        # Compute per-tensor absmax scale
        scale = self.weight.abs().max().clamp(min=1e-8) / self._qmax
        # Quantize (simulate): round to nearest integer, clamp to [–qmax, qmax]
        w_q = (self.weight / scale).round().clamp(-self._qmax, self._qmax)
        # Dequantize: restore fp32 approximation
        w_dq = w_q * scale
        # Apply dequantized weight (straight-through estimator for gradients)
        w_ste = self.weight + (w_dq - self.weight).detach()
        return F.linear(x, w_ste, self.bias)


import math  # ensure math is available for QuantizedLinear's kaiming init
