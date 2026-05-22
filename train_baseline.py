from model import create_student
from dataset import get_oxford_pet_dataloder
from torch.optim import AdamW
import torch
import torch.nn.functional as F


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

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training
        student_model.train()
        train_loss = []
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            logits = student_model(data)
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            preds = logits.argmax(dim=1)
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)

        train_acc = 100.0 * train_correct / train_total

        # Val
        student_model.eval()
        val_loss = []
        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)

                logits = student_model(data)
                loss = F.cross_entropy(logits, target)

                val_loss.append(loss.item())
                preds = logits.argmax(dim=1)
                val_correct += (preds == target).sum().item()
                val_total += target.size(0)

        val_acc = 100.0 * val_correct / val_total

        # Logging
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_val_loss = sum(val_loss) / len(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student_model.state_dict(), "baseline_best.pth")
            print(f"  -> Saved best model (Val Acc: {val_acc:.2f}%)")

    print(f"\nBaseline training complete. Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 1 (Baseline): ViT-Tiny + Cross-Entropy (No Teacher)")
    print("=" * 60)
    train_baseline(epochs=20, lr=1e-4)
