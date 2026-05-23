import random
import torch
import torchvision.transforms as T
import torchvision.datasets as d
from model import create_student

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Dataset and Transforms
transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = d.OxfordIIITPet(root="./data/oxford-pets", split="test", transform=transform)
class_names = dataset.classes

# 2. Load the trained Student Model
model = create_student(num_classes=37).to(device)
model.load_state_dict(torch.load("baseline_best.pth", map_location=device))
model.eval()

# 3. Predict on 5 Random Validation Samples
print("--- Validation Samples Inference ---")
for idx in random.sample(range(len(dataset)), 50):
    img, label = dataset[idx]
    img = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0]
        
    pred = logits.argmax(dim=1).item()
    confidence = probs[pred].item() * 100
    
    true_name = class_names[label].replace("_", " ").title()
    pred_name = class_names[pred].replace("_", " ").title()
    status = "✅ CORRECT" if pred == label else "❌ INCORRECT"
    
    print(f"True: {true_name:<25} | Pred: {pred_name:<25} ({confidence:.1f}%) | {status}")
