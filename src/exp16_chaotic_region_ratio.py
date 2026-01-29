"""
Experiment 16: Deep Representation Instability Analysis

This script runs the deep instability analysis for a SINGLE model on a SINGLE set of prompts.

It traverses the most sensitive singular vector direction in microscopic steps, measuring:
1. Local Instability (Lipschitz Constant of LPT trajectory)
2. Global Drift (L2 distance from baseline representation)
3. Logit Margin (Top-1 minus Top-2 logit difference)

Outputs:
1. Separate, high-DPI plots for Instability, Drift, and Margin per prompt.
2. A summary CSV facilitating future aggregation across multiple models.

Sample Usage:
python exp16_chaotic_region_ratio.py --model_path meta-llama/Meta-Llama-3.1-8B-Instruct --dataset_name truthful_qa --output_dir ./results/llama3_tqa --num_prompts 5
python exp16_chaotic_region_ratio.py --model_path openai/gpt-oss-20b --dataset_name walledai/AdvBench --output_dir ./results/exp16/gpt-oss_adv --num_prompts 5
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import utils

def traverse_and_measure(model, tokenizer, input_text, steps=1000, range_mag=2e-5):
    """
    Traverses the singular vector direction and records LPT metrics.
    """
    device = model.device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"])
        last_idx = inputs["input_ids"].shape[1] - 1
        
        try:
            model_dtype = next(model.parameters()).dtype
        except:
            model_dtype = torch.float32

        base_out = model(inputs_embeds=embeddings.to(model_dtype), output_hidden_states=True)
        base_lpt = model.model.norm(base_out.hidden_states[-1])[0, last_idx, :].clone().float()

    # Get Direction (Jacobian SVD)
    try:
        # compute_jacobian_svd handles dtype casting internally in utils.py
        U, S, Vt = utils.compute_jacobian_svd(model, embeddings, last_idx)
        direction = Vt[0, :].to(device=embeddings.device, dtype=torch.float32)
    except Exception as e:
        print(f"Error computing Jacobian for prompt '{input_text}': {e}")
        return None

    # Create Trajectory
    epsilons, step_size = np.linspace(-range_mag, range_mag, steps, retstep=True)
    
    lpt_trajectory = []
    logit_margins = []
    
    for eps in epsilons:
        # Math happens in Float32 (High Precision)
        pert_vector = (eps * direction)
        curr_embeddings = embeddings.clone().float() # Base embedding in float
        curr_embeddings[0, last_idx, :] = curr_embeddings[0, last_idx, :] + pert_vector
        input_embeddings_cast = curr_embeddings.to(model_dtype)
        
        with torch.no_grad():
            outputs = model(inputs_embeds=input_embeddings_cast, output_hidden_states=True)
            
            # Capture LPT and cast back to Float32 immediately
            curr_lpt = model.model.norm(outputs.hidden_states[-1])[0, last_idx, :].float()
            lpt_trajectory.append(curr_lpt.cpu())
            
            # Capture Logits
            logits = model.lm_head(outputs.hidden_states[-1])
            final_logits = logits[0, last_idx, :].float() # Cast logits to float
            top_k = torch.topk(final_logits, k=2)
            margin = (top_k.values[0] - top_k.values[1]).item()
            logit_margins.append(margin)

    lpt_stack = torch.stack(lpt_trajectory)
    global_drift = torch.norm(lpt_stack - base_lpt.cpu(), p=2, dim=1).numpy()
    
    # Local Instability (Numerical Derivative)
    diffs = lpt_stack[1:] - lpt_stack[:-1]
    local_dist = torch.norm(diffs, p=2, dim=1).numpy()
    local_instability = np.insert(local_dist / step_size, 0, 0)
    
    return epsilons, global_drift, local_instability, np.array(logit_margins)

def save_paper_plots(save_dir, prompt_idx, prompt_text, eps, instability, drift, margins):
    """
    Saves three separate, high-DPI plots suitable for publication.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    safe_prompt_id = f"P{prompt_idx:02d}"
    
    plot_kwargs = {'linewidth': 1.5, 'alpha': 0.9}
    font_sizes = {'title': 16, 'label': 14, 'ticks': 12}

    # Local Instability Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(eps, instability, color='#d62728', **plot_kwargs)
    ax.set_yscale('log')
    ax.set_title("Microscopic Representation Instability", fontsize=font_sizes['title'], fontweight='bold')
    ax.set_xlabel("Input Perturbation (ε)", fontsize=font_sizes['label'])
    ax.set_ylabel("Local Lipschitz Constant", fontsize=font_sizes['label'])
    ax.tick_params(axis='both', which='major', labelsize=font_sizes['ticks'])
    ax.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_dir / f"{safe_prompt_id}_1_instability_log.png", dpi=300)
    plt.close()

    # Global Drift Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(eps, drift, color='#1f77b4', **plot_kwargs)
    ax.set_title("Macroscopic Representation Drift", fontsize=font_sizes['title'], fontweight='bold')
    ax.set_xlabel("Input Perturbation (ε)", fontsize=font_sizes['label'])
    ax.set_ylabel("L2 Distance from Baseline Center", fontsize=font_sizes['label'])
    ax.tick_params(axis='both', which='major', labelsize=font_sizes['ticks'])
    plt.tight_layout()
    plt.savefig(save_dir / f"{safe_prompt_id}_2_drift_l2.png", dpi=300)
    plt.close()

    # Logit Margin Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(eps, margins, color='#ff7f0e', **plot_kwargs)
    ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label="Decision Boundary")
    ax.set_title("Token Prediction Confidence Margin", fontsize=font_sizes['title'], fontweight='bold')
    ax.set_xlabel("Input Perturbation (ε)", fontsize=font_sizes['label'])
    ax.set_ylabel("Logit Margin (Top1 - Top2)", fontsize=font_sizes['label'])
    ax.tick_params(axis='both', which='major', labelsize=font_sizes['ticks'])
    ax.legend(fontsize=font_sizes['ticks'])
    plt.tight_layout()
    plt.savefig(save_dir / f"{safe_prompt_id}_3_logit_margin.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run Exp 16 for a single model/dataset configuration.")
    parser.add_argument("--model_path", type=str, required=True, help="HF model ID or local path.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name.")
    parser.add_argument("--output_dir", type=str, required=True, help="Specific directory to save results.")
    parser.add_argument("--num_prompts", type=int, default=5, help="Number of prompts to analyze.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps in trajectory.")
    parser.add_argument("--range_mag", type=float, default=2e-5, help="Magnitude of epsilon range (+/-).")
    
    args = parser.parse_args()
    run_dir = Path(args.output_dir)
    utils.ensure_dir(run_dir)
    
    with open(run_dir / "config.txt", "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    print(f"\nStarting Experiment 16 Run:")
    print(f"Model: {args.model_path}")
    print(f"Output: {run_dir}")
    
    try:
        model, tokenizer = utils.load_model(model_path=args.model_path, device_map="auto") # For gpt-oss, device_map="cpu"
    except Exception as e:
        print(f"Critical Error loading model {args.model_path}: {e}")
        sys.exit(1)

    try:
        prompts = utils.get_dataset_prompts(
            dataset_name=args.dataset_name, 
            num_samples=args.num_prompts
        )
        print(f"Loaded {len(prompts)} prompts.")
    except Exception as e:
        print(f"Error loading dataset {args.dataset_name}: {e}")
        sys.exit(1)
        
    summary_data = []

    for i, prompt in enumerate(tqdm(prompts, desc="Analyzing Prompts")):
        result = traverse_and_measure(
            model, tokenizer, prompt, steps=args.steps, range_mag=args.range_mag
        )
        
        if result is None:
            continue
            
        eps, drift, instability, margins = result
        save_paper_plots(run_dir, i+1, prompt, eps, instability, drift, margins)
        
        summary_data.append({
            "model": args.model_path.split("/")[-1],
            "dataset": args.dataset_name,
            "prompt_idx": i,
            "prompt_snippet": prompt,
            "mean_instability": np.mean(instability),
            "median_instability": np.median(instability),
            "max_drift": np.max(drift),
            "mean_margin": np.mean(margins),
            "min_margin": np.min(margins)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = run_dir / "run_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"\nRun complete. Summary saved to: {summary_csv_path}")

if __name__ == "__main__":
    main()
