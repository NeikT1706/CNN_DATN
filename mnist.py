import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.quantization import prepare_qat, convert
from ptflops import get_model_complexity_info


class CNNModel_mnist(nn.Module):
    def __init__(self):
        super(CNNModel_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, bias=False)
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
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST
mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
mnist_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

transform = transforms.Compose([
    transforms.ToTensor(),  # Chuyển đổi ảnh PIL thành Tensor
    transforms.Normalize((0.5,), (0.5,))  # Chuẩn hóa ảnh (trung bình 0.5, độ lệch chuẩn 0.5)
])

# Tải dữ liệu MNIST
test_dataset = datasets.MNIST(
    root='./data',
    train=False,  # Dữ liệu kiểm tra
    download=True,
    transform=transform
)

# Tạo DataLoader cho dữ liệu kiểm tra
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình
model_mnist = CNNModel_mnist().to(device)

# Tối ưu và loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_mnist.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    model_mnist.train()
    running_loss = 0.0
    for inputs, labels in mnist_loader:  # Thay cifar_loader nếu dùng CIFAR-10
        inputs, labels = inputs.to(device), labels.to(device)

        # Reset gradient
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = model_mnist(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    accuracy = calculate_accuracy(model_mnist, mnist_loader,
                                  device)  # Thay bằng test_loader nếu muốn kiểm tra trên tập kiểm tra
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(mnist_loader)}, Accuracy: {accuracy:.4f}")

model_mnist.eval()

# Dự đoán với dữ liệu kiểm tra
correct = 0
total = 0

# Không cần tính gradient trong khi dự đoán
with (torch.no_grad()):
    for images, labels in test_loader:
        # Đầu vào cho mô hình
        outputs = model_mnist(images)

        # Lấy lớp có xác suất cao nhất (dự đoán)
        _, predicted = torch.max(outputs.data, 1)

        # Tính số dự đoán đúng
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # for i in range(len(images)):
        #     print(f"Image {total - len(images) + i + 1}: True Label: {labels[i].item()}, Predicted: {predicted[i].item()}")

# Tính độ chính xác
accuracy = 100 * correct / total
print(f'Accuracy of the model on MNIST test images: {accuracy:.2f}%')

torch.save(model_mnist, "model_mnist.pth")