"""
Full-parameter SFT on OpenHermes-2.5 for Qwen3 small (0.6B) or large (1.7B).

Usage (from the LlamaFactory root directory):
    python examples/train_full/openhermes_qwen3_full_sft.py small
    python examples/train_full/openhermes_qwen3_full_sft.py large

Model-size-dependent defaults are chosen automatically.  Any setting can be
overridden by passing extra key=value pairs after the size argument, e.g.:
    python examples/train_full/openhermes_qwen3_full_sft.py small max_samples=50000 num_train_epochs=1

Time estimates on a single RTX A6000 (48 GB):
    small (0.6B)  3 epochs / 150 K samples  ≈ 18–22 h
    large (1.7B)  1 epoch  / 150 K samples  ≈ 18–24 h
Both are sized for roughly equal wall-clock time so results are directly comparable.
"""

import sys


# ── Size-dependent defaults ────────────────────────────────────────────────────

SIZE_CONFIGS = {
    "small": {
        # 0.6B — fast enough for 3 epochs within the 48 h budget
        "model_name_or_path": "Qwen/Qwen3-0.6B",
        "output_dir": "saves/qwen3-0.6b/full/sft_openhermes",
        "per_device_train_batch_size": "4",
        "gradient_accumulation_steps": "4",   # effective batch = 16
        "num_train_epochs": "3.0",
        "gradient_checkpointing": "false",
    },
    "large": {
        # 1.7B — one epoch keeps wall-clock comparable to the 0.6B 3-epoch run;
        #        gradient checkpointing guards against OOM with full activations
        "model_name_or_path": "Qwen/Qwen3-1.7B",
        "output_dir": "saves/qwen3-1.7b/full/sft_openhermes",
        "per_device_train_batch_size": "2",
        "gradient_accumulation_steps": "8",   # effective batch = 16
        "num_train_epochs": "1.0",
        "gradient_checkpointing": "true",
    },
}

# ── Shared defaults (same for both sizes) ─────────────────────────────────────

SHARED_DEFAULTS = {
    "stage": "sft",
    "do_train": "true",
    "finetuning_type": "full",
    "dataset": "open_hermes",
    "template": "qwen3_nothink",
    "cutoff_len": "1024",
    "max_samples": "150000",
    "preprocessing_num_workers": "16",
    "dataloader_num_workers": "4",
    "learning_rate": "1.0e-5",
    "lr_scheduler_type": "cosine",
    "warmup_ratio": "0.03",
    "bf16": "true",
    "logging_steps": "50",
    "save_steps": "500",
    "plot_loss": "true",
    "overwrite_output_dir": "true",
    "save_only_model": "false",
    "report_to": "none",   # choices: none, wandb, tensorboard, swanlab, mlflow
    "ddp_timeout": "180000000",
}


def parse_overrides(args: list[str]) -> dict[str, str]:
    """Parse extra key=value arguments from the command line."""
    overrides = {}
    for arg in args:
        if "=" not in arg:
            print(f"Warning: ignoring unrecognised argument '{arg}' (expected key=value format)")
            continue
        key, _, value = arg.partition("=")
        overrides[key.strip()] = value.strip()
    return overrides


def main() -> None:
    # ── Validate size argument ─────────────────────────────────────────────
    if len(sys.argv) < 2 or sys.argv[1] not in SIZE_CONFIGS:
        print("Error: first argument must be 'small' or 'large'.\n")
        print(f"Usage: python {sys.argv[0]} [small|large] [key=value ...]")
        print("  small  ->  Qwen/Qwen3-0.6B   (~18-22 h on A6000, 3 epochs)")
        print("  large  ->  Qwen/Qwen3-1.7B   (~18-24 h on A6000, 1 epoch)")
        sys.exit(1)

    size = sys.argv[1]
    overrides = parse_overrides(sys.argv[2:])

    # ── Merge configs: shared → size-specific → user overrides ────────────
    config = {**SHARED_DEFAULTS, **SIZE_CONFIGS[size], **overrides}

    # trust_remote_code is a boolean flag with no value; handle separately
    use_trust_remote = config.pop("trust_remote_code", "true").lower() != "false"

    eff_batch = int(config["per_device_train_batch_size"]) * int(config["gradient_accumulation_steps"])

    print("=" * 50)
    print(f" OpenHermes-2.5 full SFT — {size} model")
    print("=" * 50)
    print(f"  Model            : {config['model_name_or_path']}")
    print(f"  Dataset          : {config['dataset']} (max {config['max_samples']} samples)")
    print(f"  Epochs           : {config['num_train_epochs']}")
    print(f"  Batch / grad-acc : {config['per_device_train_batch_size']} / "
          f"{config['gradient_accumulation_steps']}  (effective = {eff_batch})")
    print(f"  Output dir       : {config['output_dir']}")
    print("=" * 50)

    # ── Build final args dict and call run_exp directly ───────────────────
    if use_trust_remote:
        config["trust_remote_code"] = True

    # Convert numeric strings to their proper Python types so the
    # HfArgumentParser dataclasses receive correctly typed values.
    for key in ("per_device_train_batch_size", "gradient_accumulation_steps",
                "max_samples", "cutoff_len", "preprocessing_num_workers",
                "dataloader_num_workers", "logging_steps", "save_steps", "ddp_timeout"):
        if key in config:
            config[key] = int(config[key])

    for key in ("num_train_epochs", "learning_rate", "warmup_ratio"):
        if key in config:
            config[key] = float(config[key])

    for key in ("bf16", "plot_loss", "overwrite_output_dir", "save_only_model",
                "gradient_checkpointing", "do_train"):
        if key in config:
            val = config[key]
            config[key] = val if isinstance(val, bool) else val.lower() == "true"

    from llamafactory.train.tuner import run_exp

    run_exp(args=config)


if __name__ == "__main__":
    main()
