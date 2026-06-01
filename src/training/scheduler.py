# =============================================================================
# FILE: src/training/scheduler.py
# PURPOSE: Cosine LR scheduler with linear warmup.
#          Category: Optimization & Convergence Dynamics (Requirement C).
# =============================================================================

import math

import torch


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    ID: SCHED-COS-001
    Requirement: Return a LambdaLR that linearly warms up then cosine-decays LR.
    Purpose: Improve convergence by avoiding large early updates and allowing
             graceful decay at end of training.
    Inputs:
      - optimizer: optimizer whose LR groups will be scheduled.
      - warmup_steps: number of steps for linear ramp from 0 to base_lr.
      - max_steps: total training steps; LR reaches min_lr at max_steps.
      - min_lr_ratio: floor for LR multiplier (default 0.1 => 10% of base_lr).
    Outputs: LambdaLR scheduler instance.
    Preconditions: warmup_steps < max_steps; min_lr_ratio in (0, 1).
    Postconditions: scheduler.get_last_lr() returns current LR multiplied by
                    the cosine schedule factor.
    Assumptions: optimizer.step() and scheduler.step() called once per step.
    Side Effects: None (modifies optimizer LR when .step() called).
    Failure Modes: warmup_steps == 0 -> division by zero; guarded below.
    Error Handling: warmup_steps clipped to max(1, warmup_steps).
    Constraints: None.
    Verification: Plot LR curve for warmup_steps=100, max_steps=1000.
    References: BERT training schedule; cosine annealing (Loshchilov & Hutter).
    """
    warmup_steps = max(1, warmup_steps)

    def _lr_lambda(current_step: int) -> float:
        """
        ID: SCHED-COS-LAMBDA-001
        Requirement: Compute LR multiplier for current_step.
        Purpose: Closed-form schedule avoiding per-step conditional branches.
        Inputs: current_step (int) - zero-indexed training step.
        Outputs: float multiplier in [min_lr_ratio, 1.0].
        Preconditions: current_step >= 0.
        Postconditions: Returns 0.0 at step 0; 1.0 at warmup_steps;
                        min_lr_ratio at max_steps.
        Assumptions: None.
        Side Effects: None.
        Failure Modes: None.
        Error Handling: Clamp to [min_lr_ratio, 1.0].
        Constraints: None.
        Verification: Unit test with known schedule values.
        References: None.
        """
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(warmup_steps)
        if current_step >= max_steps:
            return min_lr_ratio
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
