import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.quantization import prepare_qat, convert
import time
import matplotlib.pyplot as plt
from torchinfo import summary


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


model = torch.load("model_cifar.pth")
model2 = torch.load("model_cifar_2_svd.pth")
model.eval()
model2.eval()
summary(model, input_size=(1, 3, 28, 28))
summary(model2, input_size=(1, 3, 28, 28))

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),  # Chuyển đổi ảnh PIL thành Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Chuẩn hóa ảnh (trung bình 0.5, độ lệch chuẩn 0.5)
])

# Tải dữ liệu MNIST
test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,  # Dữ liệu kiểm tra
    download=True,
    transform=transform
)

# Tạo DataLoader cho dữ liệu kiểm tra
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



# Dự đoán với dữ liệu kiểm tra
correct = 0
total = 0
start_time1 = time.time()
# Không cần tính gradient trong khi dự đoán
with torch.no_grad():
    for images, labels in test_loader:
        # Đầu vào cho mô hình
        outputs = model(images)
        # Lấy lớp có xác suất cao nhất (dự đoán)
        _, predicted = torch.max(outputs.data, 1)
        #print(f"Image {i}: True Label: {labels[i].item()}, Predicted: {predicted[i].item()}")
        # Tính số dự đoán đúng
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # for i in range(len(images)):
        #     print(f"Image {total - len(images) + i + 1}: True Label: {labels[i].item()}, Predicted: {predicted[i].item()}")

cifar_time = time.time() - start_time1
# Tính độ chính xác
accuracy = 100 * correct / total
print(f'Accuracy of the model on CIFAR10 test images: {accuracy:.2f}%')
print(f'Time: {cifar_time} second')

# Dự đoán với dữ liệu kiểm tra
correct = 0
total = 0
start_time2 = time.time()
# Không cần tính gradient trong khi dự đoán
with torch.no_grad():
    for images, labels in test_loader:
        # Đầu vào cho mô hình
        outputs = model2(images)
        # Lấy lớp có xác suất cao nhất (dự đoán)
        _, predicted = torch.max(outputs.data, 1)
        #print(f"Image {i}: True Label: {labels[i].item()}, Predicted: {predicted[i].item()}")
        # Tính số dự đoán đúng
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # for i in range(len(images)):
        #     print(f"Image {total - len(images) + i + 1}: True Label: {labels[i].item()}, Predicted: {predicted[i].item()}")

cifar_svd_time = time.time() - start_time2

# Tính độ chính xác
accuracy = 100 * correct / total

print(f'Accuracy of the model on CIFAR10 with SVD test images: {accuracy:.2f}%')
print(f'Time: {cifar_svd_time} second')

# Tên các biến và giá trị tương ứng
categories = ['CNN', 'CNN with SVD']
values = [cifar_time, cifar_svd_time]

# Thiết lập kích thước biểu đồ
plt.figure(figsize=(8, 6))

# Vẽ biểu đồ cột
plt.bar(categories, values, color=['blue', 'orange'], width=0.5)

# Nhãn và tiêu đề
plt.xlabel('Dataset')
plt.ylabel('Time (s)')
plt.title('Time Comparison for CNN and CNN with SVD on CIFAR10 Dataset')
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Lưới chỉ trên trục y

# Hiển thị và lưu biểu đồ
plt.savefig('TimeComparison.svg', format='svg')
plt.show()
