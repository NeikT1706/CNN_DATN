import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
from torchinfo import summary

from ptflops import get_model_complexity_info
# 🔹 Định nghĩa lại mô hình lượng tử hóa (INT8)
# class QAT_CNNModel(nn.Module):
#     def __init__(self):
#         super(QAT_CNNModel, self).__init__()
#         self.quant = torch.ao.quantization.QuantStub()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
#         self.fc1 = nn.Linear(32 * 4 * 4, 128)
#         self.fc2 = nn.Linear(128, 32)
#         self.fc3 = nn.Linear(32, 10)
#         self.dequant = torch.ao.quantization.DeQuantStub()
#
#     def forward(self, x):
#         x = self.quant(x)
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = self.dequant(x)
#         return x

# 🔹 Khởi tạo mô hình và kích hoạt lượng tử hóa

def apply_svd_fc(fc_layer, rank):

    # Trích xuất trọng số và bias
    weight = fc_layer.weight.data
    bias = fc_layer.bias.data

    # Áp dụng SVD
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)  # full_matrices=False để tối ưu hiệu suất
    V = Vh.T  # Chuyển vị để có dạng đúng

    # Giữ lại hạng đã chọn
    U_r = U[:, :rank]
    S_r = torch.diag(S[:rank])
    V_r = V[:, :rank]

    # Tạo hai lớp FC mới thay thế cho một lớp duy nhất
    fc1 = nn.Linear(V_r.size(0), rank, bias=False)  # Không cần bias ở đây
    fc2 = nn.Linear(rank, U_r.size(0), bias=True)

    # Gán trọng số đã nén
    fc1.weight.data = V_r.T  # (rank, input_dim)
    fc2.weight.data = U_r @ S_r  # (output_dim, rank)
    fc2.bias.data = bias  # Giữ nguyên bias

    return nn.Sequential(fc1, fc2)  # Trả về hai lớp liên tiếp


# Áp dụng SVD cho Convolutional Layers
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
    conv2 = nn.Conv2d(rank, C_out, kernel_size=(1, 1), stride=1, bias=True)

    conv1_weight = Vt_reduced.view(rank, C_in, K, K)
    conv2_weight = (U_reduced @ S_reduced).reshape(C_out, rank, 1, 1)

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
        # self.quant = torch.ao.quantization.QuantStub()
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        # Áp dụng SVD cho Fully Connected Layers
        self.fc1 = apply_svd_fc(nn.Linear(32 * 4 * 4, 128), rank=24)
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
model = QAT_CNNModel_SVD()
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
torch.ao.quantization.prepare_qat(model, inplace=True)
torch.ao.quantization.convert(model, inplace=True)  # Convert sang INT8

# 🔹 Load trọng số đã lưu (trọng số này đã ở dạng lượng tử hóa)
model.load_state_dict(torch.load("model_mnist_qat.pth"))
print(model)
# Đặt mô hình về chế độ đánh giá (inference)
model.eval()
print("✅ Mô hình đã được load thành công!")

# Load dataset test
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Chạy dự đoán và hiển thị ảnh
def predict_and_show(model, dataloader):
    fig, axes = plt.subplots(1, 10, figsize=(10, 3))
    model.cpu()  # Chạy trên CPU

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= 10:
                break
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Hiển thị ảnh và kết quả dự đoán
            img = inputs.squeeze().numpy()
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(f"Pred: {predicted.item()}")
            axes[i].axis("off")

    plt.show()
def evaluate_model(model, dataloader):
    correct = 0
    total = 0
    total_flops = 0
    with torch.no_grad():  # Không cần tính gradient khi đánh giá
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Độ chính xác của mô hình trên tập test: {accuracy:.2f}%")

# Load tập test với batch_size=1 để đảm bảo chạy trên toàn bộ 10k ảnh
# def count_flops(model, input_shape=(1, 32, 32)):  # CIFAR-10 có input (3, 32, 32)
#     macs, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True)
#     print(f"MACs: {macs}")
#     print(f"Parameters: {params}")
#
# count_flops(model)

# Gọi hàm đánh giá
evaluate_model(model, test_loader)
# Gọi hàm để chạy dự đoán
predict_and_show(model, test_loader)
# In tất cả các trọng số trong state_dict
# Lấy trọng số ở dạng INT8
# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.quantized.Conv2d) or isinstance(module, torch.nn.quantized.Linear):
#         print(f"Layer: {name}")
#         print("Size: ", module.weight().size())
#         print("Weight (int8):", module.weight().int_repr())  # Lấy trọng số dưới dạng INT8
#         print("Scale:", module.scale)
#         print("Zero point:", module.zero_point)
#         print("-" * 50)


