"""
Experiment 14: Lipschitz Constants via Small Steps Analysis

Investigates local smoothness at machine precision scales by measuring
consecutive output differences ||y_t - y_{t-1}||. This reveals discrete
jumps and 'stalling' caused by floating-point rounding errors.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import utils

def main():
    print("Loading model in Float32...")
    model, tokenizer = utils.load_model(dtype=torch.float32)
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"])
    last_idx = inputs["input_ids"].shape[1] - 1
    
    print("Computing Jacobian SVD...")
    U, S, Vt = utils.compute_jacobian_svd(model, embeddings, last_idx)
    direction = Vt[0, :].to(device=embeddings.device, dtype=embeddings.dtype)
    
    start_eps = 1e-6 
    delta = 2e-14
    num_steps = 200
    
    steps = [start_eps + i * delta for i in range(num_steps)]
    
    consecutive_diffs = []
    output_magnitudes = []
    prev_out = None
    
    print(f"Stepping {num_steps} times with delta={delta:.2e}...")
    
    for eps in tqdm(steps):
        pert = embeddings.clone()
        pert[0, last_idx, :] += eps * direction
        
        with torch.no_grad():
            # Get the Last Pseudo Token (LPT) - normalized last hidden state
            out_full = model(inputs_embeds=pert, output_hidden_states=True)
            out = model.model.norm(out_full.hidden_states[-1])[0, last_idx, :]
            
        if prev_out is not None:
            # Measure ||y_t - y_{t-1}||
            diff = torch.norm(out - prev_out, p=2).item()
            consecutive_diffs.append(diff)
        else:
            # First step has no diff
            consecutive_diffs.append(0.0) 
            
        prev_out = out.clone()
        output_magnitudes.append(torch.norm(out).item())

    steps_arr = np.array(steps)
    diffs_arr = np.array(consecutive_diffs)
    
    # Identify Jumps vs Stalls
    zero_mask = diffs_arr < 1e-20
    num_zeros = np.sum(zero_mask)
    num_jumps = len(diffs_arr) - num_zeros
    
    print(f"\nAnalysis Results:")
    print(f"Total Steps: {num_steps}")
    print(f"Zero-Difference Steps (Stalls): {num_zeros}")
    print(f"Discrete Jumps Detected: {num_jumps}")
    print(f"Max Jump Magnitude: {np.max(diffs_arr):.2e}")
    
    save_dir = config.RESULTS_DIR / "exp14_small_steps"
    utils.ensure_dir(save_dir)
    
    plt.figure(figsize=(10, 8))
    
    # Subplot 1: Consecutive Differences (The "Discrete Jumps")
    plt.subplot(2, 1, 1)
    plt.plot(steps_arr[1:], diffs_arr[1:], '.-', color='blue', linewidth=0.5, markersize=3)
    plt.title(f"Micro-Continuity Check (Delta={delta:.2e})\nDiscrete Jumps vs Stalls")
    plt.ylabel("Consecutive Diff ||y_t - y_{t-1}||")
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Cumulative Change (The "Staircase")
    cumulative = np.cumsum(diffs_arr)
    plt.subplot(2, 1, 2)
    plt.plot(steps_arr, cumulative, color='red')
    plt.title("Cumulative Change in Output Representation")
    plt.xlabel("Epsilon")
    plt.ylabel("Cumulative L2 Distance")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "exp14_discontinuity_analysis.png", dpi=300)
    print(f"Plot saved to {save_dir}")

if __name__ == "__main__":
    main()
