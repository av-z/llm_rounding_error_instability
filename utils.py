import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import warnings
import os
from pathlib import Path
import random
import numpy as np

try:
    import config
except ImportError:
    from . import config

warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """
    Sets the seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Global seed set to {seed}")

def load_model(model_path: str = None, device_map="auto", dtype=None):
    """
    Load model and tokenizer with proper configuration.
    
    Args:
        model_path: Path or HF ID. If None, uses config.DEFAULT_MODEL_PATH.
        device_map: "auto", "cpu", or specific device.
        dtype: torch dtype. If None, uses config.DEFAULT_DTYPE.
    """
    if model_path is None:
        model_path = config.DEFAULT_MODEL_PATH
        
    if dtype is None:
        dtype = config.DEFAULT_DTYPE

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_path} with dtype {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        dtype=dtype,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    print(f"Model loaded. Primary device: {model.device}")
    return model, tokenizer

def get_dataset_prompts(dataset_name: str = None, num_samples: int = None, seed: int = 42) -> List[str]:
    """
    Loads prompts from a HuggingFace dataset.
    
    Args:
        dataset_name: Name of the dataset to load. If None, uses config.DATASET_NAME.
        num_samples: Number of prompts to load. If None, uses config.NUM_PROMPTS.
        seed: Random seed for shuffling.
    """
    # 1. Resolve Dataset Name and Count
    target_dataset = dataset_name if dataset_name is not None else config.DATASET_NAME
    
    if num_samples is None:
        num_samples = config.NUM_PROMPTS

    print(f"Loading {num_samples} prompts from dataset: {target_dataset}...")
    
    try:
        # 2. Load Dataset
        # TruthfulQA specifically requires a configuration ("generation")
        if target_dataset == "truthful_qa":
            try:
                ds = load_dataset(target_dataset, "generation", split="validation")
            except:
                ds = load_dataset(target_dataset, split="validation")
        else:
            # For generic datasets, try validation first, then train
            try:
                ds = load_dataset(target_dataset, split="validation")
            except Exception:
                print(f"Validation split not found for {target_dataset}, trying 'train'...")
                ds = load_dataset(target_dataset, split="train", trust_remote_code=True)

        # 3. Shuffle and Select
        if len(ds) > num_samples:
            ds = ds.shuffle(seed=seed).select(range(num_samples))
        else:
            print(f"Dataset only has {len(ds)} samples, using all of them.")

        # 4. Robust Column Detection
        cols = ds.column_names
        
        # Priority list of common prompt column names for various datasets
        target_cols = ['question', 'prompt', 'goal', 'instruction', 'text', 'content', 'context']
        
        found_col = None
        for col in target_cols:
            if col in cols:
                found_col = col
                break
        
        if found_col:
            print(f"Found prompts in column: '{found_col}'")
            return [str(x) for x in ds[found_col]]
        else:
            # Fallback to the very first column if no known name matches
            first_col = cols[0]
            print(f"Warning: Unknown column format. Using first column: '{first_col}'")
            return [str(x) for x in ds[first_col]]
            
    except Exception as e:
        print(f"Error loading dataset '{target_dataset}': {e}.")
        return

def get_final_representation(model, tokenizer, input_text: str) -> torch.Tensor:
    """
    Get the final hidden representation (normalized).
    """
    device = model.device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        normalized_states = model.model.norm(last_hidden_states)
        final_representation = normalized_states[0, -1, :].float()
        
    return final_representation

def compute_jacobian_svd(model, embeddings, last_token_idx):
    """
    Compute Jacobian SVD for the last token.
    Uses torch.autograd.functional.jacobian.
    """
    # Prefer parameter dtype to avoid model.dtype mismatches (e.g., mixed bf16 weights)
    emb_dtype = next(model.parameters()).dtype

    def forward_fn(flat_emb):
        emb = flat_emb.view(1, -1)
        # Clone to avoid in-place errors, but keep grad connection
        mod_emb = embeddings.clone().to(emb_dtype)
        mod_emb[0, last_token_idx, :] = emb.to(emb_dtype)
        outputs = model(inputs_embeds=mod_emb, output_hidden_states=True)
        return outputs.hidden_states[-1][0, last_token_idx, :]

    # Detach and require grad for the specific token we are analyzing
    last_emb = (
        embeddings[0, last_token_idx, :]
        .clone()
        .detach()
        .to(emb_dtype)
        .requires_grad_(True)
    )
    
    # Compute Jacobian
    use_vectorize = embeddings.device.type != "cuda"
    try:
        jacobian = torch.autograd.functional.jacobian(
            forward_fn,
            last_emb,
            vectorize=use_vectorize
        )
    except (RuntimeError, torch.AcceleratorError) as e:
        if embeddings.device.type == "cuda" and use_vectorize:
            print("Jacobian failed on CUDA; retrying in non-vectorized mode...")
            jacobian = torch.autograd.functional.jacobian(
                forward_fn,
                last_emb,
                vectorize=False
            )
        else:
            raise
        
    # Compute SVD
    jacobian_fp32 = jacobian.float()
    try:
        U, S, Vt = torch.linalg.svd(jacobian_fp32, full_matrices=False)
    except (NotImplementedError, RuntimeError) as e:
        if "svd_cuda_gesvdj" in str(e) or "BFloat16" in str(e) or "bfloat16" in str(e):
            U, S, Vt = torch.linalg.svd(jacobian_fp32.cpu(), full_matrices=False)
        else:
            raise
    return U, S, Vt

def ensure_dir(path: Union[str, Path]):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)
