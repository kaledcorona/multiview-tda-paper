from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    # ==========================================
    #  Constant parameters
    # ==========================================
    file_path : Path = Path("mnist_784.csv")
    chunk_size : int = 70000   # Number of rows processed per data chunk
    views: set[str] = field(default_factory=lambda: {
        "original", "tda", 
        "top_left", "top_right", 
        "bottom_left", "bottom_right"
    })
    num_jobs : int = 28  # Each job uses ~3 Gb
    save_sample_views: bool = False
    vary_train_split: bool = False
    subsample_dataset: bool = True


    # ==========================================
    #  Experiment varibles
    # ==========================================
    exp_name: str = "exp_icecream"
    random_state : int = 56
    train_split: float = 0.7
    test_split: float = 0.3
    noise_enabled: bool = False
    noise_type: str = ""   # Available: ("lines", "circles", "salt")
    noise_quantity: int = 0
    noise_transparency: float = 0   # percentage [0-1]
    test_noise = False

    
    # ==========================================
    #  Model controls
    # ==========================================
    # Which model training routines to run for each view-combination
    models: list[str] = field(default_factory=lambda: ["multiview_stacking_rf"])
    # Optional per-model hyperparameters (by model key)
    model_params: dict[str, dict] = field(default_factory=dict)
    
    # ==========================================
    #  Derived parameters
    # ==========================================
    results_dir : Path = Path(f"results/{exp_name}")
    log_dir : Path = Path(f"logs/{exp_name}")
    

