import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

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
def calculate_cumulative_energy(weight_matrix):
    """
    Tính cumulative energy từ ma trận trọng số sử dụng SVD.

    Parameters:
    - weight_matrix: numpy.ndarray, ma trận trọng số của lớp.

    Returns:
    - cumulative_energy: numpy.ndarray, phần trăm năng lượng tích lũy tại từng rank.
    - singular_values: numpy.ndarray, các giá trị kỳ dị (singular values).
    """
    # Áp dụng SVD
    U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)

    # Tính cumulative energy
    singular_values = S
    cumulative_energy = np.cumsum(S ** 2) / np.sum(S ** 2)

    return cumulative_energy, singular_values


# Ví dụ:
model = torch.load("model_mnist.pth")
# Giả sử trọng số của lớp fully connected (fc1)
weight_fc1 = model.fc1.weight.data.numpy()  # Ví dụ một ma trận trọng số ngẫu nhiên
weight_fc2 = model.fc2.weight.data.numpy()
weight_fc3 = model.fc3.weight.data.numpy()
# Tính cumulative energy
cumulative_energy1, singular_values1 = calculate_cumulative_energy(weight_fc1)
cumulative_energy2, singular_values2 = calculate_cumulative_energy(weight_fc2)
cumulative_energy3, singular_values3 = calculate_cumulative_energy(weight_fc3)

# In cumulative energy
for rank, energy in enumerate(cumulative_energy1, start=1):
    if(rank == 25):
        print(f"Rank {rank}: Cumulative Energy = {energy:.4f}")

for rank, energy in enumerate(cumulative_energy2, start=1):
    if(rank == 15):
        print(f"Rank {rank}: Cumulative Energy = {energy:.4f}")

for rank, energy in enumerate(cumulative_energy3, start=1):
    if(rank == 10):
        print(f"Rank {rank}: Cumulative Energy = {energy:.4f}")

# Biểu đồ năng lượng tích lũy (tùy chọn)
#
# print(f"Rank {rank}: Cumulative Energy = {energy:.4f}")
# print(f"Rank {rank}: Cumulative Energy = {energy:.4f}")
# print(f"Rank {rank}: Cumulative Energy = {energy:.4f}")


plt.figure(figsize=(10, 6))
plt.plot(cumulative_energy1, label="fc1 (512x128)", marker='o')
plt.plot(cumulative_energy2, label="fc2 (128x32)", marker='s')
plt.plot(cumulative_energy3, label="fc3 (32x10)", marker='^')
plt.xlabel('Rank')
plt.ylabel('Cumulative Energy')
plt.title('Cumulative Energy vs. Rank')
plt.grid()
plt.legend()
plt.savefig('cumulativeEnergy_cifar.svg', format='svg')
plt.show()
