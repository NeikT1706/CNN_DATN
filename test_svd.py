import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Áp dụng SVD cho Fully Connected Layers
def apply_svd_fc(layer, rank):
    U, S, Vt = torch.svd(layer.weight.data)
    U_reduced = U[:, :rank]
    S_reduced = torch.diag(S[:rank])
    Vt_reduced = Vt[:rank, :]

    fc1 = nn.Linear(U_reduced.shape[0], U_reduced.shape[1], bias=False)
    fc2 = nn.Linear(Vt_reduced.shape[0], Vt_reduced.shape[1], bias=True)

    fc1.weight.data = U_reduced
    fc2.weight.data = torch.matmul(S_reduced, Vt_reduced)

    if layer.bias is not None:
        fc2.bias.data = layer.bias.data  # Giữ nguyên bias từ lớp gốc

    return nn.Sequential(fc1, fc2)

# Áp dụng SVD cho Convolutional Layers
def apply_svd_conv(conv_layer, rank):
    """
    Chuyển bộ lọc Conv2D thành ma trận, áp dụng SVD để nén và tạo lại hai Conv2D liên tiếp
    """
    weight = conv_layer.weight.data
    C_out, C_in, k, k = weight.shape  # Lấy kích thước kernel

    # Chuyển kernel về ma trận (C_out, C_in * k * k)
    weight_matrix = weight.view(C_out, -1)

    # Áp dụng SVD
    U, S, Vt = torch.svd(weight_matrix)
    U_reduced = U[:, :rank]
    S_reduced = torch.diag(S[:rank])
    Vt_reduced = Vt[:rank, :]

    # Chuyển Vt_reduced thành Conv1x1
    conv1 = nn.Conv2d(C_in, rank, kernel_size=1, bias=False)
    conv1.weight.data = Vt_reduced.view(rank, C_in, 1, 1)

    # Chuyển U_reduced thành Conv3x3 hoặc 5x5 (giữ kích thước gốc)
    conv2 = nn.Conv2d(rank, C_out, kernel_size=k, padding=k // 2, bias=True)
    conv2.weight.data = (U_reduced @ S_reduced).view(C_out, rank, k, k)
    conv2.bias.data = conv_layer.bias.data if conv_layer.bias is not None else torch.zeros(C_out)

    return nn.Sequential(conv1, conv2)

# Mô hình CNN với SVD cho cả Conv và FC
class QAT_CNNModel_SVD(nn.Module):
    def __init__(self):
        super(QAT_CNNModel_SVD, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()

        # Áp dụng SVD cho Conv Layers
        self.conv1 = apply_svd_conv(nn.Conv2d(1, 32, kernel_size=5), rank=16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = apply_svd_conv(nn.Conv2d(32, 32, kernel_size=5), rank=16)

        # Áp dụng SVD cho Fully Connected Layers
        self.fc1 = apply_svd_fc(nn.Linear(32 * 4 * 4, 128), rank=32)
        self.fc2 = apply_svd_fc(nn.Linear(128, 32), rank=16)
        self.fc3 = nn.Linear(32, 10)

        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dequant(x)
        return x

def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def measure_inference_time(model, dataloader, device):
    model.eval()
    total_time = 0
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            start_time = time.time()
            _ = model(inputs)
            total_time += time.time() - start_time
    return total_time / len(dataloader)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Load dữ liệu
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình gốc
model_original = QAT_CNNModel().to(device)
model_original.eval()

# Khởi tạo mô hình SVD
model_svd = QAT_CNNModel_SVD().to(device)
model_svd.eval()

# Đo độ chính xác
acc_original = calculate_accuracy(model_original, test_loader, device)
acc_svd = calculate_accuracy(model_svd, test_loader, device)

# Đo tốc độ inferencing
time_original = measure_inference_time(model_original, test_loader, device)
time_svd = measure_inference_time(model_svd, test_loader, device)

# Đếm số lượng tham số
params_original = count_parameters(model_original)
params_svd = count_parameters(model_svd)

# Hiển thị kết quả
print("📊 Kết quả so sánh:")
print(f"🔹 Mô hình gốc - Accuracy: {acc_original:.4f}, Inference time: {time_original:.6f} s, Parameters: {params_original}")
print(f"🔹 Mô hình SVD - Accuracy: {acc_svd:.4f}, Inference time: {time_svd:.6f} s, Parameters: {params_svd}")

# Đánh giá tổng thể
print("\n📢 Nhận xét:")
if acc_svd >= acc_original * 0.98:  # Giảm dưới 2% là chấp nhận được
    print("✅ Mô hình SVD duy trì độ chính xác trong phạm vi chấp nhận được!")
else:
    print("⚠️ Mô hình SVD bị mất quá nhiều độ chính xác!")

if time_svd < time_original:
    print("✅ Mô hình SVD nhanh hơn!")
else:
    print("⚠️ Mô hình SVD không cải thiện tốc độ inferencing!")

if params_svd < params_original:
    print("✅ Mô hình SVD giảm số lượng tham số đáng kể!")
else:
    print("⚠️ Mô hình SVD chưa giảm nhiều tham số!")


# Kiểm tra mô hình sau khi áp dụng SVD
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_svd = QAT_CNNModel_SVD().to(device)
print(model_svd)
