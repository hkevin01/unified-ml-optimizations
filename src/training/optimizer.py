# =============================================================================
# FILE: src/training/optimizer.py
# PURPOSE: AdamW optimizer factory with parameter group separation.
#          Category: Optimization & Convergence Dynamics (Requirement C).
# =============================================================================

from typing import List, Tuple

import torch
import torch.nn as nn


def build_adamw_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    ID: OPT-ADW-001
    Requirement: Construct AdamW with separate parameter groups to skip weight
                 decay on 1-D parameters (biases, LayerNorm scales/shifts).
    Purpose: Proper AdamW setup preventing unnecessary regularization of 1-D
             parameters, which degrades convergence.
    Inputs:
      - model: nn.Module to optimize.
      - lr: base learning rate.
      - weight_decay: L2 regularization coefficient for 2-D+ parameters.
      - betas: AdamW momentum coefficients (beta1, beta2).
      - eps: AdamW numerical stability epsilon.
    Outputs: Configured torch.optim.AdamW instance.
    Preconditions: model has at least one trainable parameter; lr > 0.
    Postconditions: Returns optimizer with two parameter groups:
                    group 0 - weight decay applied (ndim >= 2),
                    group 1 - no weight decay (ndim < 2).
    Assumptions: model.parameters() yields all trainable params.
    Side Effects: None.
    Failure Modes: Empty model -> empty param list -> optimizer raises.
    Error Handling: Assertion checks param list is non-empty.
    Constraints: None.
    Verification: Check optimizer.param_groups[0]['weight_decay'] == weight_decay
                  and optimizer.param_groups[1]['weight_decay'] == 0.0.
    References: AdamW (Loshchilov & Hutter, 2019);
                Andrej Karpathy's nanoGPT optimizer config.
    """
    # Separate params: decay for matrices, no decay for vectors/scalars
    decay_params: List[torch.Tensor] = []
    no_decay_params: List[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    assert len(decay_params) + len(no_decay_params) > 0, \
        "Model has no trainable parameters"

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        fused=torch.cuda.is_available(),  # use fused CUDA kernel when available
    )
    return optimizer


class DynamicPruner:
    """
    ID: OPT-PRUNE-001
    Requirement: Apply magnitude-based unstructured sparsity to weight tensors
                 during training, increasing sparsity linearly over a warm-up.
    Purpose: Demonstrate dynamic pruning as a convergence-dynamics optimization.
    Inputs:
      - model (nn.Module): target model.
      - target_sparsity (float): final fraction of weights zeroed (0..1).
      - start_step (int): step at which pruning begins.
      - end_step (int): step at which target_sparsity is reached.
    Outputs: DynamicPruner instance; call .step(current_step) each training step.
    Preconditions: 0 <= target_sparsity < 1; start_step < end_step.
    Postconditions: After end_step, exactly target_sparsity fraction of 2-D
                    weights are masked to zero.
    Assumptions: Gradients still flow through non-pruned weights.
    Side Effects: Modifies weight.data in-place (zeroes pruned entries).
    Failure Modes: Prune before optimizer step -> gradients might restore zeroes;
                   apply mask after optimizer step.
    Error Handling: Sparsity clamped to [0, target_sparsity].
    Constraints: Mask not stored persistently; re-computed each step.
    Verification: Count zero-valued weights after end_step.
    References: Gradual Magnitude Pruning (Zhu & Gupta, 2018).
    """

    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        start_step: int = 200,
        end_step: int = 800,
    ) -> None:
        assert 0.0 <= target_sparsity < 1.0
        assert start_step < end_step
        self.model = model
        self.target_sparsity = target_sparsity
        self.start_step = start_step
        self.end_step = end_step
        # Collect all 2-D weight tensors (Linear layers)
        self._weight_tensors = [
            p for n, p in model.named_parameters()
            if p.ndim == 2 and "weight" in n
        ]

    def _current_sparsity(self, step: int) -> float:
        """
        ID: OPT-PRUNE-SPARSITY-001
        Requirement: Linearly ramp sparsity from 0 to target_sparsity.
        Purpose: Gradual pruning avoids abrupt accuracy drops.
        Inputs: step (int) - current training step.
        Outputs: float in [0, target_sparsity].
        Preconditions: step >= 0.
        Postconditions: Returns 0 before start_step; target_sparsity after end_step.
        Assumptions: None.
        Side Effects: None.
        Failure Modes: None.
        Error Handling: Clamped to [0, target_sparsity].
        Constraints: None.
        Verification: Spot-check at start, midpoint, end.
        References: Gradual pruning schedule (Zhu & Gupta, 2018).
        """
        if step < self.start_step:
            return 0.0
        if step >= self.end_step:
            return self.target_sparsity
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        return self.target_sparsity * progress

    def step(self, current_step: int) -> None:
        """
        ID: OPT-PRUNE-STEP-001
        Requirement: Zero out the lowest-magnitude weights up to current sparsity.
        Purpose: Apply in-place pruning mask after each optimizer update.
        Inputs: current_step (int) - current training step index.
        Outputs: None (modifies weight data in-place).
        Preconditions: Called after optimizer.step() to avoid mask being overridden.
        Postconditions: Bottom-k weights (by magnitude) are zeroed.
        Assumptions: Gradient graphs re-built each step (dynamic graph mode).
        Side Effects: Writes to weight.data tensors.
        Failure Modes: None; zero-magnitude writes are no-ops.
        Error Handling: None.
        Constraints: None.
        Verification: After step, weight.abs().min() == 0.
        References: None.
        """
        sparsity = self._current_sparsity(current_step)
        if sparsity <= 0.0:
            return

        with torch.no_grad():
            for w in self._weight_tensors:
                k = int(sparsity * w.numel())
                if k == 0:
                    continue
                # Find the k-th smallest magnitude threshold
                flat = w.abs().view(-1)
                threshold = flat.kthvalue(k).values
                mask = w.abs() > threshold
                w.data.mul_(mask.float())
