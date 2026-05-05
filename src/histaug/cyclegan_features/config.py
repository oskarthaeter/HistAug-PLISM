from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CycleGANConfig:
    # Data
    paired_data: str = ""
    val_data: Optional[str] = None
    unpaired_data: Optional[str] = None
    val_split: float = 0.1
    feature_dim: int = 1024

    # Multi-target conditioning: set scanner_vocab to enable.
    # The same generator G(x, tgt_id) handles all scanner pairs conditioned on the
    # target scanner only — no source scanner identity needed at inference.
    scanner_vocab: Optional[List[str]] = None

    # Optionally restrict training/val to a subset of scanners present in the .pt file.
    # Rows whose src or tgt scanner is not in this list are dropped at load time.
    # Useful to run a smaller experiment without re-running prepare_data.
    filter_scanners: Optional[List[str]] = None

    # Generator architecture
    discriminator_hidden_dim: int = 1024
    discriminator_dropout: float = 0.3
    discriminator_use_spectral_norm: bool = False
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 1024])
    use_film: bool = False  # False = concat conditioning, True = FiLM modulation

    # Training
    batch_size: int = 256
    epochs: int = 100
    lr: float = 2e-4
    lr_d: Optional[float] = None  # discriminator LR; defaults to lr if None
    lr_decay_epochs: int = 0  # linearly decay LR to 0 over final N epochs (0 = off)
    beta1: float = 0.5
    beta2: float = 0.999
    data_fraction: float = 1.0  # fraction of training patches to use (0 < x <= 1.0)

    # Loss weights
    lambda_cyc: float = 10.0
    lambda_idt: float = 5.0
    lambda_paired: float = 10.0
    lambda_adv: float = 1.0
    # Feature loss blend: 1.0 = pure L1, 0.0 = pure cosine, 0.5 = equal blend
    loss_alpha: float = 0.2
    loss_delta: float = 1.0  # Huber loss delta parameter
    lambda_coral: float = 0.0  # CORAL covariance alignment (fake vs real distribution)
    lambda_rel: float = 0.0  # Relational pairwise-similarity structure preservation

    # Stabilization
    grad_clip_norm: float = (
        0.0  # max gradient norm applied after each backward; 0 = disabled
    )
    lambda_adv_ramp_epochs: int = (
        0  # linearly ramp lambda_adv from 0 to its target over N epochs; 0 = disabled
    )
    generator_init_checkpoint: str = (
        ""  # path to a CycleGAN or Lightning scanner-transfer checkpoint for warm init; "" = disabled
    )

    # Misc
    output_dir: str = "cyclegan_output"
    replay_buffer_size: int = 50
    seed: int = 42
    log_interval: int = 10
    save_interval: Optional[int] = None
    num_workers: int = 4

    # Wandb — leave wandb_project empty/None to disable logging
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: str = "CycleGAN"
    wandb_tags: List[str] = field(default_factory=list)
