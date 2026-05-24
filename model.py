import timm
import torch
import torch.nn as nn


def create_student(num_classes:int):
    student_model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=num_classes)
    return student_model


def create_student_teacher(num_classes:int):
    student_model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=num_classes)

    teacher_model = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=num_classes)

    student_model = load_baseline_student(student_model)

    
    for param in teacher_model.parameters():
        param.requires_grad = False

    return student_model, teacher_model


def load_baseline_student(student_model, baseline_path="experiments/baseline/baseline_best.pth"):
    try:
        student_model.load_state_dict(torch.load(baseline_path, map_location="cpu", weights_only=True))
        print(f"Successfully loaded baseline student weights from {baseline_path}")
    except Exception as e:
        print(f"Warning: Could not load baseline student weights: {e}")
    return student_model