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

