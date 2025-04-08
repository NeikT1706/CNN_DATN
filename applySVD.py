import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
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
#
# def apply_svd_to_fc_layer(fc_layer, rank):
#     # Trích xuất trọng số và bias
#     weight = fc_layer.weight.data
#     bias = fc_layer.bias.data
#
#     # Thực hiện SVD
#     U, S, Vh = torch.linalg.svd(weight)
#     V = Vh.T
#     # Giảm hạng (low-rank approximation)
#     U_r = U[:, :rank]  # Lấy các cột đầu tiên của U
#     S_r = torch.diag(S[:rank])  # Lấy các giá trị kỳ dị lớn nhất
#     V_r = V[:, :rank]  # Lấy các hàng đầu tiên của V
#
#     # Tạo các lớp thay thế
#     fc_v = nn.Linear(V_r.size(0), U_r.size(1), bias=False)
#     fc_v.weight.data = V_r.T
#
#     fc_s = nn.Linear(S_r.size(0), S_r.size(1), bias=False)
#     fc_s.weight.data = S_r
#
#     fc_u = nn.Linear(V_r.size(1), U_r.size(0))
#     fc_u.weight.data = U_r
#     fc_u.bias.data = bias  # Giữ lại bias gốc
#     return nn.Sequential(fc_v, fc_s, fc_u)
#
#     # Tầng áp dụng \( S_r \) (nhân trực tiếp)
#
#
#
# def print_svd_weights(model):
#     for name, layer in model.named_modules():
#         if isinstance(layer, nn.Sequential):
#             print(f"Layer: {name}")
#             fc_u, fc_s, fc_v = layer[0], layer[1], layer[2]
#
#             print("Matrix U (fc_u weights):")
#             print(fc_u.weight.data.numpy())
#             print(fc_u.weight.shape)
#             print("Matrix S (fc_s weights):")
#             print(fc_s.weight.data.numpy())
#             print(fc_s.weight.shape)
#             print("Matrix V^T (fc_v weights):")
#             print(fc_v.weight.data.numpy())
#             print(fc_v.weight.shape)
#             print("-" * 50)
#
# model = torch.load("model_cifar.pth")
#
# print(model)
#
# model.fc1 = apply_svd_to_fc_layer(model.fc1, 25)
# model.fc2 = apply_svd_to_fc_layer(model.fc2, 15)
# #model.fc3 = apply_svd_to_fc_layer(model.fc3, 10)
#
# #print_svd_weights(model)
# #print(model)
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values :\n {param.data}")
# torch.save(model, "model_cifar_svd.pth")

def apply_svd_to_fc_layer(fc_layer, rank):
    weight = fc_layer.weight.data
    bias = fc_layer.bias.data

    U, S, Vh = torch.linalg.svd(weight)
    V = Vh.T

    U_r = U[:, :rank]
    S_r = torch.diag(S[:rank])
    V_r = V[:, :rank]

    fc_approx = nn.Linear(V_r.size(0), U_r.size(0), bias=True)
    fc_approx.weight.data = (U_r @ S_r) @ V_r.T
    fc_approx.bias.data = bias

    return fc_approx


def apply_svd_to_conv_layer(conv_layer, rank):
    weight = conv_layer.weight.data  # [out_channels, in_channels, kH, kW]
    bias = conv_layer.bias.data if conv_layer.bias is not None else None

    out_channels, in_channels, kH, kW = weight.shape
    reshaped_weight = weight.view(out_channels, -1)  # [out_channels, in_channels * kH * kW]

    U, S, Vh = torch.linalg.svd(reshaped_weight)
    V = Vh.T

    U_r = U[:, :rank]
    S_r = torch.diag(S[:rank])
    V_r = V[:, :rank]

    reduced_conv1 = nn.Conv2d(in_channels, rank, kernel_size=(kH, kW), padding=conv_layer.padding, bias=False)
    reduced_conv1.weight.data = V_r.T.view(rank, in_channels, kH, kW)

    reduced_conv2 = nn.Conv2d(rank, out_channels, kernel_size=1, bias=True if bias is not None else False)
    reduced_conv2.weight.data = (U_r @ S_r).view(out_channels, rank, 1, 1)

    if bias is not None:
        reduced_conv2.bias.data = bias

    return nn.Sequential(reduced_conv1, reduced_conv2)


# Load model
model = torch.load("model_mnist.pth")

# Apply SVD to convolutional layers
model.conv1 = apply_svd_to_conv_layer(model.conv1, rank=6)
model.conv2 = apply_svd_to_conv_layer(model.conv2, rank=12)

# Apply SVD to fully connected layers
model.fc1 = apply_svd_to_fc_layer(model.fc1, 25)
model.fc2 = apply_svd_to_fc_layer(model.fc2, 15)

# Print weights after SVD
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values :\n {param.data}")

# Save modified model
torch.save(model, "model_mnist_2_svd.pth")
