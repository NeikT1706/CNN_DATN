import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.quantization import prepare_qat, convert
from ptflops import get_model_complexity_info

class CNNModel_Cifar(nn.Module):
    def __init__(self):
        super(CNNModel_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, bias=False)
        self.fc1 = nn.Linear(32 * 4 * 4, 128, bias=False)
        self.fc2 = nn.Linear(128, 32, bias=False)
        self.fc3 = nn.Linear(32, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def calculate_accuracy(model, dataloader, device):
    model.eval()  # Đặt mô hình ở chế độ đánh giá
    correct = 0
    total = 0

    with torch.no_grad():  # Tắt gradient để tăng tốc độ và tiết kiệm bộ nhớ
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Lấy nhãn dự đoán (lớp có xác suất cao nhất)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total  # Trả về tỷ lệ chính xác (accuracy)

# Mô hình ban đầu
# Chuẩn bị dữ liệu

# CIFAR-10
transform_cifar = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
cifar_train = datasets.CIFAR10(root='./data', train=True, transform=transform_cifar, download=True)
cifar_test = datasets.CIFAR10(root='./data', train=False, transform=transform_cifar, download=True)

# DataLoader

cifar_loader = DataLoader(cifar_train, batch_size=64, shuffle=True)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,  # Dữ liệu kiểm tra
    download=True,
    transform=transform_cifar
)

# Tạo DataLoader cho dữ liệu kiểm tra
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình
model_cifar = CNNModel_Cifar().to(device)

# Tối ưu và loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cifar.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
for epoch in range(10):
    model_cifar.train()
    running_loss = 0.0
    for inputs, labels in cifar_loader:  # Thay cifar_loader nếu dùng CIFAR-10
        inputs, labels = inputs.to(device), labels.to(device)

        # Reset gradient
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = model_cifar(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    accuracy = calculate_accuracy(model_cifar, cifar_loader,
                                  device)  # Thay bằng test_loader nếu muốn kiểm tra trên tập kiểm tra
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(cifar_loader)}, Accuracy: {accuracy:.4f}")

# Dự đoán với dữ liệu kiểm tra


correct = 0
total = 0

# Không cần tính gradient trong khi dự đoán
with (torch.no_grad()):
    for images, labels in test_loader:
        # Đầu vào cho mô hình
        outputs = model_cifar(images)

        # Lấy lớp có xác suất cao nhất (dự đoán)
        _, predicted = torch.max(outputs.data, 1)

        # Tính số dự đoán đúng
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # for i in range(len(images)):
        #     print(f"Image {total - len(images) + i + 1}: True Label: {labels[i].item()}, Predicted: {predicted[i].item()}")

# Tính độ chính xác
accuracy = 100 * correct / total
print(f'Accuracy of the model on CIFAR10 test images: {accuracy:.2f}%')

torch.save(model_cifar, "model_cifar.pth")