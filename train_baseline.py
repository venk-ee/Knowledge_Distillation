from model import create_student
from dataset import get_oxford_pet_dataloder
from torch.optim import AdamW
import torch
import torch.nn.functional as F
from metrics import calculate_topk_accuracy, AverageMeter


def train_baseline(epochs=20, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_oxford_pet_dataloder()

    # Model 
    student_model = create_student(num_classes=37)
    student_model.to(device)

    # Optimizer
    optimizer = AdamW(student_model.parameters(), lr=lr)

    best_val_top1 = 0.0
    best_val_top5 = 0.0

    for epoch in range(epochs):
        # Training
        student_model.train()
        train_loss = AverageMeter()
        train_top1 = AverageMeter()
        train_top5 = AverageMeter()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            logits = student_model(data)
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            acc1, acc5 = calculate_topk_accuracy(logits, target, topk=(1, 5))
            train_loss.update(loss.item(), data.size(0))
            train_top1.update(acc1, data.size(0))
            train_top5.update(acc5, data.size(0))

        # Val
        student_model.eval()
        val_loss = AverageMeter()
        val_top1 = AverageMeter()
        val_top5 = AverageMeter()

        with torch.inference_mode():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)

                logits = student_model(data)
                loss = F.cross_entropy(logits, target)

                # Calculate metrics
                acc1, acc5 = calculate_topk_accuracy(logits, target, topk=(1, 5))
                val_loss.update(loss.item(), data.size(0))
                val_top1.update(acc1, data.size(0))
                val_top5.update(acc5, data.size(0))

        # Logging
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss.avg:.4f} | Train Top-1: {train_top1.avg:.2f}% | Train Top-5: {train_top5.avg:.2f}% | "
            f"Val Loss: {val_loss.avg:.4f} | Val Top-1: {val_top1.avg:.2f}% | Val Top-5: {val_top5.avg:.2f}%"
        )

        # Save Best Model by Top-1 (Backward-compatible with infer_test.py using baseline_best.pth)
        if val_top1.avg > best_val_top1:
            best_val_top1 = val_top1.avg
            torch.save(student_model.state_dict(), "baseline_best.pth")
            print(f"  -> Saved best model by Top-1 (Val Top-1: {val_top1.avg:.2f}%)")

        # Save Best Model by Top-5
        if val_top5.avg > best_val_top5:
            best_val_top5 = val_top5.avg
            torch.save(student_model.state_dict(), "baseline_best_top5.pth")
            print(f"  -> Saved best model by Top-5 (Val Top-5: {val_top5.avg:.2f}%)")

    print(f"\nBaseline training complete. Best Val Top-1 Acc: {best_val_top1:.2f}% | Best Val Top-5 Acc: {best_val_top5:.2f}%")



if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 1 (Baseline): ViT-Tiny + Cross-Entropy (No Teacher)")
    print("=" * 60)
    train_baseline(epochs=20, lr=1e-4)
