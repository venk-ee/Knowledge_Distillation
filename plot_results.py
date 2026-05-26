import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    experiments = {
        "Baseline": "experiments/baseline/baseline_history.json",
        "ViTKD": "experiments/vitdk/distillation_history.json",
        "CLS Token": "experiments/cls_token/distillation_history.json",
        "Attention": "experiments/attention/distillation_history.json"
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, path in experiments.items():
        if not os.path.exists(path):
            continue
        
        with open(path, "r") as f:
            history = json.load(f)

        axes[0].plot(history["epoch"], history["val_loss"], label=name)
        axes[1].plot(history["epoch"], history["val_top1"], label=name)

    axes[0].set_title("Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("distillation_results_comparison.png", dpi=150)
    print("Plot saved successfully!")

if __name__ == "__main__":
    main()
