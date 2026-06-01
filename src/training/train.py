# =============================================================================
# FILE: src/training/train.py
# PURPOSE: Main training loop integrating all optimization components.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.utils.config import Config
from src.utils.metrics import AverageMeter, accuracy
from src.utils.profiler import ThroughputProfiler
from src.training.optimizer import build_adamw_optimizer, DynamicPruner
from src.training.scheduler import get_cosine_schedule_with_warmup


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    pruner: DynamicPruner,
    cfg: Config,
    global_step: int,
    profiler: ThroughputProfiler,
) -> tuple:
    """
    ID: TRAIN-EPOCH-001
    Requirement: Run one full pass over the training DataLoader.
    Purpose: Integrate AMP, grad clipping, pruning, and LR scheduling.
    Inputs:
      - model: OptimizedTransformer in train mode.
      - loader: training DataLoader.
      - optimizer: AdamW instance.
      - scheduler: LambdaLR cosine schedule.
      - scaler: GradScaler for AMP.
      - pruner: DynamicPruner instance.
      - cfg: root Config.
      - global_step: step counter at epoch start.
      - profiler: ThroughputProfiler for throughput measurement.
    Outputs: (avg_loss: float, avg_acc: float, global_step: int).
    Preconditions: model is on cfg.training.device; loader yields (ids, labels).
    Postconditions: optimizer and scheduler stepped once per batch;
                    pruner applied after each optimizer step.
    Assumptions: cfg.training.use_amp implies CUDA available.
    Side Effects: Modifies model weights; writes profiler state.
    Failure Modes: NaN loss -> training diverges; grad clip partially mitigates.
    Error Handling: Skips scaler update if gradients overflow (via scaler).
    Constraints: None.
    Verification: Loss decreases monotonically on overfit batch test.
    References: PyTorch AMP tutorial; gradient clipping best practices.
    """
    model.train()
    device = cfg.training.device
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    use_amp = cfg.training.use_amp and torch.cuda.is_available()

    for batch_idx, (input_ids, labels) in enumerate(loader):
        if global_step >= cfg.training.max_steps:
            break

        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        profiler.start_step()

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', enabled=use_amp):
            logits = model(input_ids)
            loss = F.cross_entropy(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()

        # --- Category C: Dynamic pruning applied after optimizer step ---
        pruner.step(global_step)

        scheduler.step()
        profiler.end_step()

        loss_val = loss.item()
        acc_val = accuracy(logits.detach(), labels)
        loss_meter.update(loss_val)
        acc_meter.update(acc_val)
        global_step += 1

        if global_step % cfg.training.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"  step {global_step:5d} | "
                f"loss {loss_meter.value:.4f} | "
                f"acc {acc_meter.value:.3f} | "
                f"lr {lr:.2e}"
            )

    return loss_meter.global_avg, acc_meter.global_avg, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: str,
) -> tuple:
    """
    ID: TRAIN-EVAL-001
    Requirement: Run inference-only pass over validation DataLoader.
    Purpose: Measure held-out loss and accuracy without gradient computation.
    Inputs:
      - model: nn.Module in eval mode.
      - loader: validation DataLoader.
      - device: target device string.
    Outputs: (avg_loss: float, avg_acc: float).
    Preconditions: model is on device; loader yields (ids, labels).
    Postconditions: Model remains in eval mode; no weight updates.
    Assumptions: torch.no_grad() context prevents memory accumulation.
    Side Effects: None.
    Failure Modes: OOM if batch too large; reduce batch_size.
    Error Handling: None; caller handles OOM.
    Constraints: None.
    Verification: Val accuracy non-decreasing on synthetic data after training.
    References: None.
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for input_ids, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(input_ids)
        loss = F.cross_entropy(logits, labels)
        loss_meter.update(loss.item(), n=input_ids.shape[0])
        acc_meter.update(accuracy(logits, labels), n=input_ids.shape[0])

    return loss_meter.global_avg, acc_meter.global_avg


def train(model: nn.Module, train_loader, val_loader, cfg: Config) -> dict:
    """
    ID: TRAIN-MAIN-001
    Requirement: Full training run with AdamW, cosine LR, AMP, pruning, profiling.
    Purpose: Top-level orchestrator called from main.py.
    Inputs:
      - model: OptimizedTransformer.
      - train_loader / val_loader: DataLoader instances.
      - cfg: root Config.
    Outputs: dict with keys 'train_losses', 'val_losses', 'val_accs', 'profiler'.
    Preconditions: model allocated; cfg valid.
    Postconditions: Model trained for cfg.training.max_steps steps.
    Assumptions: CUDA available if cfg.training.device == 'cuda'.
    Side Effects: Prints training log; modifies model weights.
    Failure Modes: Device unavailable -> RuntimeError.
    Error Handling: Falls back to CPU if CUDA unavailable.
    Constraints: None.
    Verification: Loss curve decreases; profiler.report() shows throughput > 0.
    References: None.
    """
    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available; falling back to CPU.")
        device = "cpu"
        cfg.training.device = "cpu"
        cfg.training.use_amp = False

    model = model.to(device)

    # --- Category C: AdamW with proper weight-decay groups ---
    optimizer = build_adamw_optimizer(
        model,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # --- Category C: Cosine LR schedule with warmup ---
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=cfg.training.warmup_steps,
        max_steps=cfg.training.max_steps,
    )

    scaler = GradScaler('cuda', enabled=(cfg.training.use_amp and torch.cuda.is_available()))

    # --- Category C: Dynamic magnitude pruning ---
    pruner = DynamicPruner(
        model,
        target_sparsity=0.3,
        start_step=cfg.training.warmup_steps,
        end_step=int(cfg.training.max_steps * 0.8),
    )

    profiler = ThroughputProfiler(cfg.training.batch_size, device)

    history = {"train_losses": [], "val_losses": [], "val_accs": []}
    global_step = 0
    epoch = 0

    print(f"[Train] Starting: max_steps={cfg.training.max_steps}, device={device}")
    print(f"[Train] Model params: {sum(p.numel() for p in model.parameters()):,}")

    while global_step < cfg.training.max_steps:
        epoch += 1
        print(f"\n--- Epoch {epoch} ---")
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, pruner, cfg, global_step, profiler,
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["val_accs"].append(val_acc)
        print(
            f"  [Epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    profiler.print_report()
    history["profiler"] = profiler.report()
    return history
