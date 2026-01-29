"""
Experiment 1: Layer-wise SVD Perturbation Analysis
Comprehensive analysis of how perturbations propagate through all layers.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import utils

def get_all_layer_hidden_states(model, embeddings, last_token_idx):
    """Capture hidden states from all layers."""
    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
        all_layers = torch.stack([h[0, last_token_idx, :].float().cpu() for h in outputs.hidden_states])
    return all_layers.numpy()

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = config.RESULTS_DIR / f"exp1_{timestamp}"
    utils.ensure_dir(save_dir)
    print(f"Results will be saved to: {save_dir}")

    model, tokenizer = utils.load_model()
    prompt = "The capital of France is"
    utils.set_seed(42)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"])
    last_idx = inputs["input_ids"].shape[1] - 1
    
    print("Computing Jacobian SVD...")
    U, S, Vt = utils.compute_jacobian_svd(model, embeddings, last_idx)
    
    np.save(save_dir / "top_5_singular_vectors.npy", Vt[:5].cpu().numpy())
    np.save(save_dir / "top_5_singular_values.npy", S[:5].cpu().numpy())
    print("Saved singular vectors/values.")

    jumps = [1e-5, 2e-5, 5e-5, 1e-4] 
    direction = Vt[0, :].to(embeddings.device) # Primary sensitive direction

    print("Capturing baseline layer states...")
    baseline_layers = get_all_layer_hidden_states(model, embeddings, last_idx)

    for jump in jumps:
        print(f"\nAnalyzing perturbation magnitude: {jump}")
        
        # Apply perturbation
        pert_emb = embeddings.clone()
        pert_emb[0, last_idx, :] += jump * direction
        
        # Get Perturbed Representations
        pert_layers = get_all_layer_hidden_states(model, pert_emb, last_idx)
        
        # Calculate Differences
        diffs = np.linalg.norm(baseline_layers - pert_layers, axis=1) # L2 norm per layer
        
        np.savez(
            save_dir / f"layer_data_jump_{jump}.npz",
            baseline=baseline_layers,
            perturbed=pert_layers,
            l2_diffs=diffs,
            jump_magnitude=jump
        )
        
        # Line Plot
        plt.figure(figsize=(10, 6))
        plt.plot(diffs, marker='o')
        plt.xlabel("Layer Index")
        plt.ylabel("L2 Distance")
        plt.title(f"Layer-wise Divergence (Jump: {jump})")
        plt.grid(True)
        plt.savefig(save_dir / f"layer_divergence_jump_{jump}.pdf")
        plt.close()
        
        # Heatmap of Difference Vector (First 100 dims for readability)
        diff_vectors = np.abs(baseline_layers - pert_layers)[:, :100]
        plt.figure(figsize=(12, 8))
        sns.heatmap(diff_vectors, cmap="viridis", cbar_kws={'label': 'Absolute Difference'})
        plt.xlabel("Hidden Dimension (First 100)")
        plt.ylabel("Layer Index")
        plt.title(f"Difference Heatmap (Jump: {jump})")
        plt.savefig(save_dir / f"diff_heatmap_jump_{jump}.pdf")
        plt.close()

if __name__ == "__main__":
    main()
