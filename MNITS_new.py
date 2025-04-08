import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class QAT_CNNModel(nn.Module):
    def __init__(self):
        super(QAT_CNNModel, self).__init__()
        # Lớp lượng tử hóa đầu vào và đầu ra
        self.quant = torch.ao.quantization.QuantStub()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)  # Lượng tử hóa đầu vào
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)  # Giải lượng tử hóa đầu ra
        return x

# def apply_svd_fc(layer, rank):
#     U, S, Vt = torch.svd(layer.weight.data)
#     U_reduced = U[:, :rank]
#     S_reduced = torch.diag(S[:rank])
#     Vt_reduced = Vt[:rank, :]
#
#     fc1 = nn.Linear(U_reduced.shape[0], U_reduced.shape[1], bias=False)
#     fc2 = nn.Linear(Vt_reduced.shape[0], Vt_reduced.shape[1], bias=True)
#
#     fc1.weight.data = U_reduced
#     fc2.weight.data = torch.matmul(S_reduced, Vt_reduced)
#
#     if layer.bias is not None:
#         fc2.bias.data = layer.bias.data  # Giữ nguyên bias từ lớp gốc
#
#     return nn.Sequential(fc1, fc2)

def apply_svd_fc(fc_layer, rank):
    print("Applying SVD...")

    # Trích xuất trọng số và bias
    weight = fc_layer.weight.data
    #bias = fc_layer.bias.data

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
    #fc2.bias.data = bias  # Giữ nguyên bias

    return nn.Sequential(fc1, fc2)  # Trả về hai lớp liên tiếp


# Áp dụng SVD cho Convolutional Layers
# def apply_svd_conv(conv_layer, rank):
#     # Lấy trọng số ban đầu
#     C_out, C_in, K, _ = conv_layer.weight.shape  # (C_out, C_in, K, K)
#
#     # Chuyển về ma trận 2D (C_out, C_in*K*K)
#     weight_matrix = conv_layer.weight.data.view(C_out, C_in * K * K)
#
#     # Áp dụng SVD
#     U, S, Vt = torch.svd(weight_matrix)
#
#     # Giảm rank bằng cách lấy `rank` cột đầu tiên
#     U_reduced = U[:, :rank]  # (C_out, rank)
#     S_reduced = torch.diag(S[:rank])  # (rank, rank)
#     Vt_reduced = Vt[:, :rank]  # (C_in*K*K, rank)
#
#     # Tạo hai Conv2D layers thay thế
#     conv1 = nn.Conv2d(C_in, rank, kernel_size=(K, 1), stride=1, padding=(K // 2, 0), bias=False)
#     conv2 = nn.Conv2d(rank, C_out, kernel_size=(1, K), stride=1, padding=(0, K // 2), bias=True)
#
#     # Ánh xạ trọng số
#     conv1.weight.data = Vt_reduced.T.view(rank, C_in, K, 1)
#     conv2.weight.data = (U_reduced @ S_reduced).view(C_out, rank, 1, K)
#
#     return conv1, conv2

def apply_svd_conv(conv_layer, rank):
    # Lấy trọng số ban đầu
    C_out, C_in, K, _ = conv_layer.weight.shape  # (C_out, C_in, K, K)

    weight_matrix = conv_layer.weight.data.view(C_out, C_in * K * K)

    U, S, Vt = torch.linalg.svd(weight_matrix, full_matrices=False)

    U_reduced = U[:, :rank]
    S_reduced = torch.diag(S[:rank])
    Vt_reduced = Vt[:rank, :]

    print(f"U_reduced shape: {U_reduced.shape}")
    print(f"S_reduced shape: {S_reduced.shape}")
    print(f"Vt_reduced shape: {Vt_reduced.shape}")

    # Sửa kernel_size từ (K, 1) thành (K, K)
    conv1 = nn.Conv2d(C_in, rank, kernel_size=(K, K), stride=1, padding=(0, 0), bias=False)
    conv2 = nn.Conv2d(rank, C_out, kernel_size=(1, 1), stride=1, bias=False)

    conv1_weight = Vt_reduced.view(rank, C_in, K, K)
    conv2_weight = (U_reduced @ S_reduced).reshape(C_out, rank, 1, 1)

    print(f"conv1_weight shape: {conv1_weight.shape}")
    print(f"conv2_weight shape: {conv2_weight.shape}")

    conv1.weight.data = conv1_weight
    conv2.weight.data = conv2_weight

    return nn.Sequential(conv1, conv2)


# Mô hình CNN với SVD cho cả Conv và FC
class QAT_CNNModel_SVD(nn.Module):
    def __init__(self):
        super(QAT_CNNModel_SVD, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        #
        # # Áp dụng SVD cho Conv Layers
        self.conv1 = apply_svd_conv(nn.Conv2d(1, 32, kernel_size=5), rank=8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = apply_svd_conv(nn.Conv2d(32, 32, kernel_size=5), rank=8)
        self.fc1 = apply_svd_fc(nn.Linear(32 * 4 * 4, 128), rank=24)
        self.fc2 = apply_svd_fc(nn.Linear(128, 32), rank=16)
        self.fc3 = nn.Linear(32, 10, bias=False)

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình QAT
model_qat = QAT_CNNModel_SVD().to(device)

# Cấu hình lượng tử hóa
model_qat.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')  # Dành cho CPU

# Chuyển đổi mô hình sang chế độ QAT
torch.ao.quantization.prepare_qat(model_qat, inplace=True)

# Kiểm tra mô hình đã được chuẩn bị lượng tử hóa
print(model_qat)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_qat.parameters(), lr=0.001)

# Training loop (Huấn luyện lại mô hình QAT)
for epoch in range(5):  # Ít epoch hơn do QAT mất nhiều thời gian
    model_qat.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_qat(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Kiểm tra độ chính xác trên tập test trước khi convert
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

accuracy_before = calculate_accuracy(model_qat, test_loader, device)
print(f"Final accuracy before quantization: {accuracy_before:.2f}")

# Chuyển đổi mô hình sang INT8
model_qat.cpu()  # Chuyển sang CPU trước khi convert
torch.ao.quantization.convert(model_qat, inplace=True)

# Lưu mô hình lượng tử hóa INT8
torch.save(model_qat.state_dict(), "model_mnist_qat.pth")

print("✅ Mô hình đã được lượng tử hóa và lưu!")
