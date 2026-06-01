# =============================================================================
# FILE: src/main.py
# PURPOSE: Entry point for the unified-ml-optimizations project.
# =============================================================================

import sys
import os

# Ensure the project root is on sys.path so src.* imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config, ModelConfig, TrainingConfig
from src.data.dataset import get_dataloaders
from src.models.base_model import OptimizedTransformer
from src.training.train import train
from src.utils.metrics import estimate_flops

import torch


def main() -> None:
    """
    ID: MAIN-001
    Requirement: Parse minimal CLI flags, construct config, run training.
    Purpose: Single entry point demonstrating all three optimization categories.
    Inputs: None (reads sys.argv for --fast flag to run a quick smoke test).
    Outputs: None (prints training log and final metrics to stdout).
    Preconditions: All src.* modules importable; PyTorch installed.
    Postconditions: Training completes; final metrics printed.
    Assumptions: Default config is valid for a one-GPU or CPU run.
    Side Effects: Prints to stdout; may write profiler trace to ./profile_trace/.
    Failure Modes: Import error if dependencies missing; CUDA OOM if config too large.
    Error Handling: Falls back to CPU; prints warning.
    Constraints: None.
    Verification: python src/main.py --fast should complete without errors.
    References: None.
    """
    fast_mode = "--fast" in sys.argv

    # Build config - use smaller dims for fast/CPU smoke test
    if fast_mode:
        model_cfg = ModelConfig(
            vocab_size=1024,
            d_model=128,
            n_heads=4,
            n_layers=2,
            seq_len=32,
            rank=16,
            num_classes=10,
        )
        train_cfg = TrainingConfig(
            batch_size=32,
            lr=3e-4,
            max_steps=50,
            warmup_steps=10,
            log_interval=10,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_amp=torch.cuda.is_available(),
        )
    else:
        model_cfg = ModelConfig()
        train_cfg = TrainingConfig()

    cfg = Config(model=model_cfg, training=train_cfg)

    print("=" * 60)
    print("  Unified ML Optimizations Demo")
    print("  A. Mathematical: Low-rank Q/K/V attention")
    print("  B. Systems:      Triton fused Linear+GELU + INT8 head")
    print("  C. Convergence:  AdamW + cosine warmup + dynamic pruning")
    print("=" * 60)

    # Build data
    train_loader, val_loader = get_dataloaders(
        batch_size=cfg.training.batch_size,
        seq_len=cfg.model.seq_len,
        vocab_size=cfg.model.vocab_size,
        num_classes=cfg.model.num_classes,
        num_train=8192 if not fast_mode else 512,
        num_val=1024 if not fast_mode else 128,
        num_workers=0,  # safe default for all platforms
    )

    # Build model
    model = OptimizedTransformer(cfg.model)

    # FLOPs estimate before training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = torch.randint(0, cfg.model.vocab_size, (1, cfg.model.seq_len)).to(device)
    model_tmp = model.to(device).eval()
    flops = estimate_flops(model_tmp, sample)
    print(f"[FLOPs] Estimated forward-pass FLOPs: {flops:,}")
    model_tmp = model_tmp.cpu()
    del model_tmp

    # Train
    history = train(model, train_loader, val_loader, cfg)

    # Final summary
    print("\n" + "=" * 60)
    print("  Training Complete")
    if history["val_accs"]:
        print(f"  Best val accuracy: {max(history['val_accs']):.3f}")
    prof = history.get("profiler", {})
    if prof:
        print(f"  Throughput: {prof['throughput_samples_s']:.1f} samples/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
