import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.quantization import prepare_qat, convert
from ptflops import get_model_complexity_info
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print()

class CNNModel_Cifar(nn.Module):
    def __init__(self):
        super(CNNModel_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)

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

class CNNModel_mnist(nn.Module):
    def __init__(self):
        super(CNNModel_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)

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

def apply_svd_fc(fc_layer, rank):

    # Trích xuất trọng số và bias
    weight = fc_layer.weight.data
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
    return nn.Sequential(fc1, fc2)  # Trả về hai lớp liên tiếp

def apply_svd_conv(conv_layer, rank):
    # Lấy trọng số ban đầu
    C_out, C_in, K, _ = conv_layer.weight.shape  # (C_out, C_in, K, K)

    weight_matrix = conv_layer.weight.data.view(C_out, C_in * K * K)

    U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False)

    U_reduced = U[:, :rank]
    S_reduced = torch.diag(S[:rank])
    Vt_reduced = Vt[:rank, :]

    # Sửa kernel_size từ (K, 1) thành (K, K)
    conv1 = nn.Conv2d(C_in, rank, kernel_size=(K, K), stride=1, padding=(0, 0), bias=False)
    conv2 = nn.Conv2d(rank, C_out, kernel_size=(1, 1), stride=1, bias=False)

    conv1_weight = Vt_reduced.view(rank, C_in, K, K)
    conv2_weight = (U_reduced @ S_reduced).reshape(C_out, rank, 1, 1)

    conv1.weight.data = conv1_weight
    conv2.weight.data = conv2_weight

    return nn.Sequential(conv1, conv2)


# Mô hình CNN với SVD cho cả Conv và FC
class CNNModel_SVD_mnist(nn.Module):
    def __init__(self):
        super(CNNModel_SVD_mnist, self).__init__()
        self.conv1 = apply_svd_conv(nn.Conv2d(1, 32, kernel_size=5), rank=8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = apply_svd_conv(nn.Conv2d(32, 32, kernel_size=5), rank=8)
        self.fc1 = apply_svd_fc(nn.Linear(32 * 4 * 4, 128), rank=24)
        self.fc2 = apply_svd_fc(nn.Linear(128, 32), rank=16)
        self.fc3 = nn.Linear(32, 10)

        #self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class CNNModel_SVD_Cifar(nn.Module):
    def __init__(self):
        super(CNNModel_SVD_Cifar, self).__init__()
        # # Áp dụng SVD cho Conv Layers
        self.conv1 = apply_svd_conv(nn.Conv2d(3, 32, kernel_size=5), rank=8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = apply_svd_conv(nn.Conv2d(32, 32, kernel_size=5), rank=8)
        self.fc1 = apply_svd_fc(nn.Linear(32 * 4 * 4, 128), rank=24)
        self.fc2 = apply_svd_fc(nn.Linear(128, 32), rank=16)
        self.fc3 = nn.Linear(32, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def count_flops(model, input_shape=(1, 28, 28)):  # CIFAR-10 có input (3, 32, 32)
    macs, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True)
    print(f"MACs: {macs}")
    print(f"Parameters: {params}")

# model_mnist = CNNModel_mnist()
# model_mnist_svd = CNNModel_SVD_mnist()
# model_cifar = CNNModel_Cifar()
# model_cifar_svd = CNNModel_SVD_Cifar()

model_mnist = torch.load('model_mnist.pth')
model_mnist_svd = torch.load('model_mnist_svd.pth')
model_cifar = torch.load('model_cifar.pth')
model_cifar_svd = torch.load('model_cifar_svd.pth')
# model_mnist_qat = torch.load('model_mnist_qat.pth')
#
# print(model_mnist_qat)

# for key, value in model_mnist_qat.items():
#     if "weight" in key:
#         print(f"🔹 Tên tham số: {key}")
#         print(f"  - Kích thước: {value.size()}")
#         print(f"  - Trọng số:\n{value}\n")

# print(model_mnist_qat.keys())

# print(model_mnist)
# print(model_mnist_svd)
# print(model_cifar)
# print(model_cifar_svd)

count_flops(model_mnist)
count_flops(model_mnist_svd)
count_flops(model_cifar, (3,28,28))
count_flops(model_cifar_svd, (3,28,28))

models = {
    "CNN_MNIST": CNNModel_mnist(),
    "CNN_MNIST_SVD": CNNModel_SVD_mnist(),
    "CNN_CIFAR": CNNModel_Cifar(),
    "CNN_CIFAR_SVD": CNNModel_SVD_Cifar(),
}

# Input shape của từng mô hình
input_shapes = {
    "CNN_MNIST": (1, 28, 28),
    "CNN_MNIST_SVD": (1, 28, 28),
    "CNN_CIFAR": (3, 28, 28),
    "CNN_CIFAR_SVD": (3, 28, 28),
}

# Lưu trữ số lượng phép tính và số tham số
macs_values = []
param_values = []
labels = []

for name, model in models.items():
    macs, params = get_model_complexity_info(model, input_shapes[name], as_strings=False, print_per_layer_stat=False)

    macs_values.append(macs / 1e6)  # Chuyển MACs sang triệu (M)
    param_values.append(params / 1e3)  # Chuyển Params sang triệu (M)
    labels.append(name)

# Vẽ biểu đồ
fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # 2 biểu đồ trên 1 hàng

# Biểu đồ 1: MACs
axes[0].bar(labels, macs_values, color='blue')
axes[0].set_title("MACs Comparison")
axes[0].set_ylabel("MACs (Millions)")
axes[0].set_xticklabels(labels, rotation=20)

# Biểu đồ 2: Params
axes[1].bar(labels, param_values, color='orange')
axes[1].set_title("Parameters Comparison")
axes[1].set_ylabel("Parameters (Thousands)")
axes[1].set_xticklabels(labels, rotation=20)


# Hiển thị giá trị trên cột
def add_labels(ax, values):
    max_val = max(values)
    offset = max_val * 0.02

    for i, v in enumerate(values):
        ax.text(i, v + offset, f"{v:.2f}", ha='center', fontsize=10)


add_labels(axes[0], macs_values)
add_labels(axes[1], param_values)

plt.tight_layout()
plt.show()