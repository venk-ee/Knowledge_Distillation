import timm
import torch
import torch.nn as nn

def create_student_teacher(num_classes:int):

    student_model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=num_classes)

    teacher_model = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=num_classes)

    
    for param in teacher_model.parameters():
        param.requires_grad = False

    return student_model, teacher_model