import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.quantization
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

# def print_quantized_weights(model):
#     for name, module in model.items():
#         if isinstance(module, nn.quantized.Linear):
#             # Unpack the weight and bias
#             weight, bias = module._packed_params.unpack()
#
#             # Dequantize the weights to get the float values
#             weight = weight.dequantize()
#
#             print(f"{name}.weight:", weight)
#             print(f"{name}.bias:", bias)
#             print(f"{name}.scale:", module.scale)
#             print(f"{name}.zero_point:", module.zero_point)
#             print()

def apply_svd_fc(fc_layer, rank):
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
class QAT_CNNModel_SVD(nn.Module):
    def __init__(self):
        super(QAT_CNNModel_SVD, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        # # Áp dụng SVD cho Conv Layers
        self.conv1 = apply_svd_conv(nn.Conv2d(1, 32, kernel_size=5), rank=8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = apply_svd_conv(nn.Conv2d(32, 32, kernel_size=5), rank=8)
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

# Nạp mô hình lượng tử hóa
# model_qat = QAT_CNNModel_SVD()
# model_qat.load_state_dict(torch.load("model_mnist_qat.pth"))

    model_mnist_qat = torch.load("model_mnist_qat.pth")
    model_cifar_qat = torch.load("model_cifar_qat.pth")
    weights_list_mnist = []
    weights_list_cifar = []
    # print(model_qat)
    for name, module in model_mnist_qat.items():
        print(name)
        print(module)
        if "weight" in name:
            print(f"🔹 {name}")
            weights_list_mnist.append(module.int_repr().numpy())
        elif "_packed_params._packed_params" in name:
            print(f"🔹 {name}")
            weights_list_mnist.append(module[0].int_repr().numpy())

    for name, module in model_cifar_qat.items():
        if "weight" in name:
            print(f"🔹 {name}")
            weights_list_cifar.append(module.int_repr().numpy())
        elif "_packed_params._packed_params" in name:
            print(f"🔹 {name}")
            weights_list_cifar.append(module[0].int_repr().numpy())
    # for name, module in model_qat.items():
    #         if isinstance(module, torch.nn.Linear):
    #             weight = module._packed_params[0]  # Lấy trọng số từ tuple
    #             print(f"Layer: {name}, Weight shape: {weight.shape}")
    # print(weights_list_mnist)
    # print_quantized_weights(model_qat)
    for item in weights_list_mnist:
        print(item.shape)

    for item in weights_list_cifar:
        print(item.shape)
    # print(weights_list_mnist[2])
    # weights_list_mnist[2].reshape(8, 800)
    # print(weights_list_mnist[2])
    # weights_list_mnist[3]
    # print(weights_list_mnist[3].shape)
    # print(type(weights_list_mnist[0][0]))
    # weights_list_mnist[0] = weights_list_mnist[0].reshape(8, 25)
    # weights_list_mnist[1] = weights_list_mnist[1].reshape(32, 8)
    # weights_list_mnist[2] = weights_list_mnist[2].reshape(8, 800)
    # weights_list_mnist[3] = weights_list_mnist[3].reshape(32, 8)

    np.savez("weights_mnist.npz", conv1_0=weights_list_mnist[0], conv1_1=weights_list_mnist[1], conv2_0=weights_list_mnist[2], conv2_1=weights_list_mnist[3], fc1_0=weights_list_mnist[4],
             fc1_1=weights_list_mnist[5], fc2_0=weights_list_mnist[6], fc2_1=weights_list_mnist[7], fc3=weights_list_mnist[8])

    np.savez("weights_cifar.npz", conv1_0=weights_list_cifar[0], conv1_1=weights_list_cifar[1], conv2_0=weights_list_cifar[2], conv2_1=weights_list_cifar[3], fc1_0=weights_list_cifar[4],
             fc1_1=weights_list_cifar[5], fc1_2=weights_list_cifar[6], fc2_0=weights_list_cifar[7], fc2_1=weights_list_cifar[8])



