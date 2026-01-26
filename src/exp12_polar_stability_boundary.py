"""
Experiment 12: Polar Stability Boundary Mapping
Maps the stability boundary by first finding the instability threshold 
via exponential search, then refining with binary search + ULP precision.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import utils

def find_precise_boundary(model, embeddings, last_idx, direction, initial_max_s=1e-4, hard_limit=10.0):
    """
    Finds the boundary using Exponential Search -> Binary Search -> ULP Refinement.
    """
    with torch.no_grad():
        base_logits = model(inputs_embeds=embeddings).logits[0, last_idx, :]
        base_token = torch.argmax(base_logits).item()

    low = 0.0
    high = initial_max_s
    is_unstable = False
    
    # Expand 'high' until we find a flip or hit hard_limit
    while high <= hard_limit:
        pert = embeddings.clone()
        pert[0, last_idx, :] += high * direction
        with torch.no_grad():
            curr_token = torch.argmax(model(inputs_embeds=pert).logits[0, last_idx, :]).item()
        
        if curr_token != base_token:
            is_unstable = True
            break
        
        low = high
        high *= 2.0
    
    if not is_unstable:
        print(f"Warning: Model stable even at perturbation {hard_limit}. Returning limit.")
        return hard_limit, hard_limit, hard_limit

    # Binary Search
    for _ in range(20): 
        mid = (low + high) / 2
        pert = embeddings.clone()
        pert[0, last_idx, :] += mid * direction
        with torch.no_grad():
            curr_token = torch.argmax(model(inputs_embeds=pert).logits[0, last_idx, :]).item()
        
        if curr_token == base_token:
            low = mid
        else:
            high = mid

    # ULP Refinement
    stable_bound = low
    curr = low
    found_exact = False
    
    for _ in range(100): 
        next_val = np.nextafter(curr, high)
        if next_val >= high:
            break
            
        pert = embeddings.clone()
        pert[0, last_idx, :] += next_val * direction
        with torch.no_grad():
            token = torch.argmax(model(inputs_embeds=pert).logits[0, last_idx, :]).item()
            
        if token == base_token:
            stable_bound = next_val
            curr = next_val
        else:
            high = next_val 
            found_exact = True
            break 
            
    return stable_bound, low, high

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = config.RESULTS_DIR / f"exp12_{timestamp}"
    utils.ensure_dir(save_dir)
    
    model, tokenizer = utils.load_model(dtype=torch.float32)
    prompt = "The capital of France is"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(inputs["input_ids"])
    last_idx = inputs["input_ids"].shape[1] - 1
    
    print("Computing Jacobian SVD...")
    U, S, Vt = utils.compute_jacobian_svd(model, embeddings, last_idx)
    v1 = Vt[0, :]
    v2 = Vt[1, :]
    
    num_angles = 1000 # High resolution
    thetas = np.linspace(0, 2*np.pi, num_angles)
    max_s_values = []
    lows = []
    highs = []
    
    print(f"Mapping stability boundary ({num_angles} angles)...")
    for theta in tqdm(thetas):
        # Create direction: d = cos(theta)*e1 + sin(theta)*e2
        direction = (np.cos(theta) * v1 + np.sin(theta) * v2).to(embeddings.device)

        max_s, low, high = find_precise_boundary(model, embeddings, last_idx, direction)
        max_s_values.append(max_s)
        lows.append(low)
        highs.append(high)
        
    np.savez(
        save_dir / "polar_boundary_data.npz",
        thetas=thetas,
        max_s_values=np.array(max_s_values),
        low_values=np.array(lows),
        high_values=np.array(highs),
        singular_values=S[:2].cpu().numpy(),
        e1=v1.cpu().numpy(),
        e2=v2.cpu().numpy(),
        input_text=prompt
    )
    
    df = pd.DataFrame({
        "theta": thetas, 
        "max_s": max_s_values, 
        "low": lows, 
        "high": highs
    })
    df.to_csv(save_dir / "polar_boundary_data.csv", index=False)
    
    # Visualization: Polar
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    ax.plot(thetas, max_s_values)
    ax.set_title("Stability Boundary (Polar)")
    plt.savefig(save_dir / "polar_stability_boundary.png")
    plt.close()
    
    # Visualization: Cartesian
    plt.figure(figsize=(10, 6))
    plt.plot(np.degrees(thetas), max_s_values)
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Max Stable Perturbation (s)")
    plt.title("Stability Boundary (Cartesian)")
    plt.grid(True)
    plt.savefig(save_dir / "stability_boundary_cartesian.png")
    plt.close()
    
    print(f"Stats: Mean={np.mean(max_s_values):.2e}, Std={np.std(max_s_values):.2e}")
    print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    main()
