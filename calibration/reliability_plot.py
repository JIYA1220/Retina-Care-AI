"""
reliability_plot.py
Calculate Expected Calibration Error (ECE) and generate reliability diagrams.
A reliability diagram plots mean confidence vs. mean accuracy in bins.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def calc_ece(probs, labels, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    Average difference between confidence and accuracy across bins.
    """
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Binary mask for samples in this bin
        in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()

def plot_reliability_diagram(probs, labels, n_bins=10, title="Reliability Diagram"):
    """
    Generates a reliability diagram plot.
    """
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        
        if in_bin.sum().item() > 0:
            bin_accs.append(accuracies[in_bin].float().mean().item())
            bin_confs.append(confidences[in_bin].mean().item())
        else:
            bin_accs.append(0)
            bin_confs.append(0)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(np.arange(0.05, 1.05, 0.1), bin_accs, width=0.1, alpha=0.3, edgecolor='black', color='blue', label='Accuracy')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Confidence')
    ax.set_title(title)
    ax.legend()
    
    ece = calc_ece(probs, labels, n_bins)
    plt.text(0.1, 0.8, f"ECE: {ece:.4f}", fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
    
    return fig

if __name__ == "__main__":
    print("This script is a utility and should be called after collecting logits/labels.")
