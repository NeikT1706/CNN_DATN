import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.quantization import prepare_qat, convert
from ptflops import get_model_complexity_info

def apply_svd_fc(fc_layer, rank):
    # print("Applying SVD...")

    # Trích xuất trọng số và bias
    weight = fc_layer.weight.data
    # bias = fc_layer.bias.data

    # Áp dụng SVD
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)  # full_matrices=False để tối ưu hiệu suất
    V = Vh.T  # Chuyển vị để có dạng đúng

    # Giữ lại hạng đã chọn
    U_r = U[:, :rank]
    S_r = torch.diag(S[:rank])
    V_r = V[:, :rank]

    # Tạo hai lớp FC mới thay thế cho một lớp duy nhất
    fc1 = nn.Linear(V_r.size(0), rank, bias=False)  # Không cần bias ở đây
    fc2 = nn.Linear(rank, U_r.size(0), bias=False)

    # Gán trọng số đã nén
    fc1.weight.data = V_r.T  # (rank, input_dim)
    fc2.weight.data = U_r @ S_r  # (output_dim, rank)
    # fc2.bias.data = bias  # Giữ nguyên bias

    return nn.Sequential(fc1, fc2)  # Trả về hai lớp liên tiếp

def apply_svd_conv(conv_layer, rank):
    # Lấy trọng số ban đầu
    C_out, C_in, K, _ = conv_layer.weight.shape  # (C_out, C_in, K, K)

    weight_matrix = conv_layer.weight.data.view(C_out, C_in * K * K)

    U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False)

    U_reduced = U[:, :rank]
    S_reduced = torch.diag(S[:rank])
    Vt_reduced = Vt[:rank, :]

    # print(f"U_reduced shape: {U_reduced.shape}")
    # print(f"S_reduced shape: {S_reduced.shape}")
    # print(f"Vt_reduced shape: {Vt_reduced.shape}")

    # Sửa kernel_size từ (K, 1) thành (K, K)
    conv1 = nn.Conv2d(C_in, rank, kernel_size=(K, K), stride=1, padding=(0, 0), bias=False)
    conv2 = nn.Conv2d(rank, C_out, kernel_size=(1, 1), stride=1, bias=False)

    conv1_weight = Vt_reduced.view(rank, C_in, K, K)
    conv2_weight = (U_reduced @ S_reduced).reshape(C_out, rank, 1, 1)

    # print(f"conv1_weight shape: {conv1_weight.shape}")
    # print(f"conv2_weight shape: {conv2_weight.shape}")

    conv1.weight.data = conv1_weight
    conv2.weight.data = conv2_weight

    return nn.Sequential(conv1, conv2)


# Mô hình CNN với SVD cho cả Conv và FC
class CNNModel_SVD_Cifar(nn.Module):
    def __init__(self):
        super(CNNModel_SVD_Cifar, self).__init__()
        #self.quant = torch.ao.quantization.QuantStub()
        #
        # # Áp dụng SVD cho Conv Layers
        self.conv1 = apply_svd_conv(nn.Conv2d(3, 32, kernel_size=5), rank=8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = apply_svd_conv(nn.Conv2d(32, 32, kernel_size=5), rank=8)
        # self.quant = torch.ao.quantization.QuantStub()
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        # Áp dụng SVD cho Fully Connected Layers
        self.fc1 = apply_svd_fc(nn.Linear(32 * 4 * 4, 128), rank=24)
        self.fc2 = apply_svd_fc(nn.Linear(128, 32), rank=16)
        self.fc3 = nn.Linear(32, 10, bias=False)

        #self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        #x = self.quant(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.dequant(x)
        return x
# class CNNModel(nn.Module):
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
#         self.fc1 = nn.Linear(32 * 4 * 4, 128)
#         self.fc2 = nn.Linear(128, 32)
#         self.fc3 = nn.Linear(32, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = torch.flatten(x, 1)  # Flatten
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

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
model_cifar = CNNModel_SVD_Cifar().to(device)

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

torch.save(model_cifar, "model_cifar_svd.pth")