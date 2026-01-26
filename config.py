import os
import torch
from pathlib import Path

# Hardware Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()
MAIN_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Project Paths
PROJECT_ROOT = Path(__file__).parent.absolute()
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "meta-llama/Meta-Llama-3.1-8B-Instruct")
DEFAULT_DTYPE = torch.float32

DATASET_NAME = "truthful_qa"
DATASET_CONFIG = "generation" 
NUM_PROMPTS = 20
