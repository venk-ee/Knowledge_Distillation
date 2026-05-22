from model import create_student_teacher
from dataset import get_oxford_pet_dataloder
from loss import LogitDistillationLoss
from torch.optim import AdamW
import torch
import torch.nn.functional as F

def train_knowledge_distillation(epochs):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader,val_loader=get_oxford_pet_dataloder()

    student_model,teacher_model=create_student_teacher(num_classes=37)


    student_model.to(device)

    teacher_model.to(device)

    criterion = LogitDistillationLoss(temperature=2)

    optimizer=AdamW(student_model.parameters(),lr=1e-4)

    for epoch in range(epochs):
        student_model.train()
        teacher_model.eval()

        train_loss=[]
        for batch_idx,(data,target) in enumerate(train_loader):
            data,target=data.to(device),target.to(device)

            with torch.inference_mode():
                teacher_outputs_logits=teacher_model(data)
            
            student_outputs_logits=student_model(data)
            
            loss = criterion(student_outputs_logits, teacher_outputs_logits, target)
            
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        student_model.eval()
        val_loss=[]
        for batch_idx,(data,target) in enumerate(val_loader):
            data,target=data.to(device),target.to(device)

            with torch.inference_mode():
                student_outputs_logits=student_model(data)
            loss = F.cross_entropy(student_outputs_logits, target)
            val_loss.append(loss.item())

        print(f'Epoch: {epoch} | Train Loss: {sum(train_loss)/len(train_loss):.4f} | Val Loss: {sum(val_loss)/len(val_loss):.4f}')


if __name__ == "__main__":
    print("Starting Knowledge Distillation Pipeline...")
    train_knowledge_distillation(epochs=3)
