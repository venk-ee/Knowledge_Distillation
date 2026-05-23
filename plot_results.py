import json
import os
import matplotlib.pyplot as plt

def load_history(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def main():
    baseline = load_history("baseline_history.json")
    distill = load_history("distillation_history.json")

    if not baseline and not distill:
        print("Error: Neither 'baseline_history.json' nor 'distillation_history.json' found.")
        print("Please train your models first to generate the logs.")
        return

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Loss Plot
    if baseline:
        axes[0].plot(baseline["epoch"], baseline["train_loss"], label="Baseline Train Loss", linestyle="--", color="dodgerblue", marker='o')
        axes[0].plot(baseline["epoch"], baseline["val_loss"], label="Baseline Val Loss", color="dodgerblue", marker='o')
    if distill:
        axes[0].plot(distill["epoch"], distill["train_loss"], label="KD Train Loss", linestyle="--", color="crimson", marker='o')
        axes[0].plot(distill["epoch"], distill["val_loss"], label="KD Val Loss", color="crimson", marker='o')
    
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Accuracy Plot
    if baseline:
        axes[1].plot(baseline["epoch"], baseline["val_top1"], label="Baseline Val Top-1", color="dodgerblue", marker='o')
        axes[1].plot(baseline["epoch"], baseline["val_top5"], label="Baseline Val Top-5", linestyle=":", color="dodgerblue", marker='^')
    if distill:
        axes[1].plot(distill["epoch"], distill["val_top1"], label="KD Val Top-1", color="crimson", marker='o')
        axes[1].plot(distill["epoch"], distill["val_top5"], label="KD Val Top-5", linestyle=":", color="crimson", marker='^')

    axes[1].set_title("Validation Accuracy (Top-1 & Top-5)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_filename = "distillation_results_comparison.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Comparison plot successfully saved to '{output_filename}'!")

if __name__ == "__main__":
    main()
