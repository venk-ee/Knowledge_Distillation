import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitDistillationLoss(nn.Module):
    def __init__(self, alpha:float=0.5, temperature:float=2.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha


    def forward(self, student_logits, teacher_logits, labels):
        
        standard_loss=F.cross_entropy(student_logits,labels)

        log_softmax_teacher_probs=F.log_softmax(teacher_logits/self.temperature,dim=1)

        log_softmax_student_logits = F.log_softmax(student_logits/self.temperature,dim=1)

        distillation_loss=F.kl_div(log_softmax_student_logits, log_softmax_teacher_probs, reduction='batchmean',log_target=True)

        scaled_distillation_loss = distillation_loss * (self.temperature ** 2)


        total_loss=(1-self.alpha)*standard_loss + self.alpha*scaled_distillation_loss

        return total_loss


class ViTKDLoss(nn.Module):
    def __init__(self, student_dim: int = 192, teacher_dim: int = 1024, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        
        self.projector = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_logits, student_features, teacher_features, labels):
        
        task_loss = F.cross_entropy(student_logits, labels)

        student_patches = student_features[:, 1:, :]
        teacher_patches = teacher_features[:, 1:, :]


        projected_student_patches = self.projector(student_patches)


        distillation_loss = F.mse_loss(projected_student_patches, teacher_patches)

        total_loss = (1 - self.alpha) * task_loss + self.alpha * distillation_loss
        
        return total_loss


class CLSDistillationLoss(nn.Module):
    def __init__(self, student_dim: int = 192, teacher_dim: int = 1024, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        
        self.projector = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_logits, student_features, teacher_features, labels):
        
        task_loss = F.cross_entropy(student_logits, labels)

        student_cls = student_features[:, 0]
        teacher_cls = teacher_features[:, 0]

        projected_student_cls = self.projector(student_cls)

        distillation_loss = F.mse_loss(projected_student_cls, teacher_cls)

        total_loss = (1 - self.alpha) * task_loss + self.alpha * distillation_loss
        
        return total_loss



class AttentionTransferLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        # Look ma, no projector! 

    def forward(self, student_logits, student_features, teacher_features, labels):
        
        task_loss = F.cross_entropy(student_logits, labels)

        student_patches = student_features[:, 1:, :] # Shape: [Batch, Num_Patches, 192]
        teacher_patches = teacher_features[:, 1:, :] # Shape: [Batch, Num_Patches, 1024]
        student_attn = student_patches.norm(p=2, dim=-1) # Shape becomes: [Batch, Num_Patches]
        teacher_attn = teacher_patches.norm(p=2, dim=-1) # Shape becomes: [Batch, Num_Patches]

        student_attn = F.normalize(student_attn, p=2, dim=-1)
        teacher_attn = F.normalize(teacher_attn, p=2, dim=-1)

        distillation_loss = F.mse_loss(student_attn, teacher_attn)

        total_loss = (1 - self.alpha) * task_loss + self.alpha * distillation_loss
        
        return total_loss