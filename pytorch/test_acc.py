import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import *
# === 用户配置 ===
MODEL_PATH = "./models/fedfm/globalmodelexperiment_log-2025-10-23-1135-14.pth"
BATCH_SIZE = 128
NUM_CLASSES = 10  # CIFAR10

# === 1. 定义或导入模型结构 ===
import torchvision.models as models
model = ModelFedCon_noheader("resnet18_7", 256, 10, None)

# === 2. 加载模型参数 ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# === 3. 准备 CIFAR-10 测试集 ===
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, 
                                            download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=0)

# === 4. 测试准确率 ===
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        print(outputs)
        if isinstance(outputs, tuple):
            outputs = outputs[2]
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ CIFAR-10 测试集准确率: {accuracy:.2f}%")
