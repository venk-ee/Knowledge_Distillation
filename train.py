from model import create_student_teacher
from dataset import get_oxford_pet_dataloder
from loss import ViTKDLoss
from torch.optim import AdamW
import torch
import torch.nn.functional as F
from metrics import calculate_topk_accuracy, AverageMeter
import json


def train_knowledge_distillation(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_oxford_pet_dataloder()

    student_model, teacher_model = create_student_teacher(num_classes=37)

    student_model.to(device)
    teacher_model.to(device)

    criterion = ViTKDLoss().to(device)
    optimizer = AdamW(list(student_model.parameters()) + list(criterion.parameters()), lr=1e-4)

    best_val_top1 = 0.0
    best_val_top5 = 0.0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_top1": [],
        "train_top5": [],
        "val_loss": [],
        "val_top1": [],
        "val_top5": []
    }

    for epoch in range(epochs):
        student_model.train()
        teacher_model.eval()

        train_loss = AverageMeter()
        train_top1 = AverageMeter()
        train_top5 = AverageMeter()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                teacher_features = teacher_model.forward_features(data)
            
            student_features = student_model.forward_features(data)
            student_outputs_logits = student_model.forward_head(student_features)
            
            loss = criterion(student_outputs_logits, student_features, teacher_features, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            acc1, acc5 = calculate_topk_accuracy(student_outputs_logits, target, topk=(1, 5))
            train_loss.update(loss.item(), data.size(0))
            train_top1.update(acc1, data.size(0))
            train_top5.update(acc5, data.size(0))

        student_model.eval()
        val_loss = AverageMeter()
        val_top1 = AverageMeter()
        val_top5 = AverageMeter()

        with torch.inference_mode():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)

                student_outputs_logits = student_model(data)
                loss = F.cross_entropy(student_outputs_logits, target)

                acc1, acc5 = calculate_topk_accuracy(student_outputs_logits, target, topk=(1, 5))
                val_loss.update(loss.item(), data.size(0))
                val_top1.update(acc1, data.size(0))
                val_top5.update(acc5, data.size(0))

        # Record history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss.avg)
        history["train_top1"].append(train_top1.avg)
        history["train_top5"].append(train_top5.avg)
        history["val_loss"].append(val_loss.avg)
        history["val_top1"].append(val_top1.avg)
        history["val_top5"].append(val_top5.avg)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss.avg:.4f} | Train Top-1: {train_top1.avg:.2f}% | Train Top-5: {train_top5.avg:.2f}% | "
            f"Val Loss: {val_loss.avg:.4f} | Val Top-1: {val_top1.avg:.2f}% | Val Top-5: {val_top5.avg:.2f}%"
        )

        # Save Best Student Model by Top-1
        if val_top1.avg > best_val_top1:
            best_val_top1 = val_top1.avg
            torch.save(student_model.state_dict(), "distillation_best.pth")
            print(f"  -> Saved best student model by Top-1 (Val Top-1: {val_top1.avg:.2f}%)")

        # Save Best Student Model by Top-5
        if val_top5.avg > best_val_top5:
            best_val_top5 = val_top5.avg
            torch.save(student_model.state_dict(), "distillation_best_top5.pth")
            print(f"  -> Saved best student model by Top-5 (Val Top-5: {val_top5.avg:.2f}%)")

        # Save training history to JSON incrementally
        with open("distillation_history.json", "w") as f:
            json.dump(history, f, indent=4)

    print("Saved training history to distillation_history.json")


from loss import CLSDistillationLoss
import torch.optim as optim


def train_cls_token_feature_distillation(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_oxford_pet_dataloder()

    student_model, teacher_model = create_student_teacher(num_classes=37)

    student_model.to(device)
    teacher_model.to(device)

    criterion = CLSDistillationLoss(student_dim=192, teacher_dim=1024, alpha=0.5).to(device)
    optimizer = optim.Adam([
        {'params': student_model.parameters()},
        {'params': criterion.parameters()} 
        ], lr=1e-4)

    best_val_top1 = 0.0
    best_val_top5 = 0.0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_top1": [],
        "train_top5": [],
        "val_loss": [],
        "val_top1": [],
        "val_top5": []
    }

    for epoch in range(epochs):
        student_model.train()
        teacher_model.eval()

        train_loss = AverageMeter()
        train_top1 = AverageMeter()
        train_top5 = AverageMeter()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                # Just get the features, no need for logits!
                teacher_features = teacher_model.forward_features(data)
            
            # 1. Get Student internal features
            student_features = student_model.forward_features(data)
            
            # 2. Safely pass the CLS token to the head via forward_head
            student_outputs_logits = student_model.forward_head(student_features)
            
            # 3. Calculate Loss
            loss = criterion(student_outputs_logits, student_features, teacher_features, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            acc1, acc5 = calculate_topk_accuracy(student_outputs_logits, target, topk=(1, 5))
            train_loss.update(loss.item(), data.size(0))
            train_top1.update(acc1, data.size(0))
            train_top5.update(acc5, data.size(0))

        student_model.eval()
        val_loss = AverageMeter()
        val_top1 = AverageMeter()
        val_top5 = AverageMeter()

        with torch.inference_mode():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)

                student_outputs_logits = student_model(data)
                loss = F.cross_entropy(student_outputs_logits, target)

                acc1, acc5 = calculate_topk_accuracy(student_outputs_logits, target, topk=(1, 5))
                val_loss.update(loss.item(), data.size(0))
                val_top1.update(acc1, data.size(0))
                val_top5.update(acc5, data.size(0))

        # Record history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss.avg)
        history["train_top1"].append(train_top1.avg)
        history["train_top5"].append(train_top5.avg)
        history["val_loss"].append(val_loss.avg)
        history["val_top1"].append(val_top1.avg)
        history["val_top5"].append(val_top5.avg)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss.avg:.4f} | Train Top-1: {train_top1.avg:.2f}% | Train Top-5: {train_top5.avg:.2f}% | "
            f"Val Loss: {val_loss.avg:.4f} | Val Top-1: {val_top1.avg:.2f}% | Val Top-5: {val_top5.avg:.2f}%"
        )

        # Save Best Student Model by Top-1
        if val_top1.avg > best_val_top1:
            best_val_top1 = val_top1.avg
            torch.save(student_model.state_dict(), "distillation_best.pth")
            print(f"  -> Saved best student model by Top-1 (Val Top-1: {val_top1.avg:.2f}%)")

        # Save Best Student Model by Top-5
        if val_top5.avg > best_val_top5:
            best_val_top5 = val_top5.avg
            torch.save(student_model.state_dict(), "distillation_best_top5.pth")
            print(f"  -> Saved best student model by Top-5 (Val Top-5: {val_top5.avg:.2f}%)")

        # Save training history to JSON incrementally
        with open("distillation_history.json", "w") as f:
            json.dump(history, f, indent=4)

    print("Saved training history to distillation_history.json")



from loss import AttentionTransferLoss

def train_attention_distillation(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_oxford_pet_dataloder()

    student_model, teacher_model = create_student_teacher(num_classes=37)

    student_model.to(device)
    teacher_model.to(device)

    criterion = AttentionTransferLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

    best_val_top1 = 0.0
    best_val_top5 = 0.0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_top1": [],
        "train_top5": [],
        "val_loss": [],
        "val_top1": [],
        "val_top5": []
    }

    for epoch in range(epochs):
        student_model.train()
        teacher_model.eval()

        train_loss = AverageMeter()
        train_top1 = AverageMeter()
        train_top5 = AverageMeter()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                teacher_features = teacher_model.forward_features(data)
            
            student_features = student_model.forward_features(data)
            
            student_outputs_logits = student_model.forward_head(student_features)
            
            loss = criterion(student_outputs_logits, student_features, teacher_features, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            acc1, acc5 = calculate_topk_accuracy(student_outputs_logits, target, topk=(1, 5))
            train_loss.update(loss.item(), data.size(0))
            train_top1.update(acc1, data.size(0))
            train_top5.update(acc5, data.size(0))

        student_model.eval()
        val_loss = AverageMeter()
        val_top1 = AverageMeter()
        val_top5 = AverageMeter()

        with torch.inference_mode():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)

                student_outputs_logits = student_model(data)
                loss = F.cross_entropy(student_outputs_logits, target)

                acc1, acc5 = calculate_topk_accuracy(student_outputs_logits, target, topk=(1, 5))
                val_loss.update(loss.item(), data.size(0))
                val_top1.update(acc1, data.size(0))
                val_top5.update(acc5, data.size(0))

        # Record history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss.avg)
        history["train_top1"].append(train_top1.avg)
        history["train_top5"].append(train_top5.avg)
        history["val_loss"].append(val_loss.avg)
        history["val_top1"].append(val_top1.avg)
        history["val_top5"].append(val_top5.avg)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss.avg:.4f} | Train Top-1: {train_top1.avg:.2f}% | Train Top-5: {train_top5.avg:.2f}% | "
            f"Val Loss: {val_loss.avg:.4f} | Val Top-1: {val_top1.avg:.2f}% | Val Top-5: {val_top5.avg:.2f}%"
        )

        # Save Best Student Model by Top-1
        if val_top1.avg > best_val_top1:
            best_val_top1 = val_top1.avg
            torch.save(student_model.state_dict(), "distillation_best.pth")
            print(f"  -> Saved best student model by Top-1 (Val Top-1: {val_top1.avg:.2f}%)")

        # Save Best Student Model by Top-5
        if val_top5.avg > best_val_top5:
            best_val_top5 = val_top5.avg
            torch.save(student_model.state_dict(), "distillation_best_top5.pth")
            print(f"  -> Saved best student model by Top-5 (Val Top-5: {val_top5.avg:.2f}%)")

        # Save training history to JSON incrementally
        with open("distillation_history.json", "w") as f:
            json.dump(history, f, indent=4)

    print("Saved training history to distillation_history.json")


if __name__=="__main__":
    train_attention_distillation(20)