import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def im2col(image, kernel_size, stride=1, padding=0):
    """
    Chuyển đổi ảnh MNIST (grayscale) thành ma trận im2col.

    Parameters:
    - image: (H, W) - Ảnh grayscale đầu vào.
    - kernel_size: (kh, kw) - Kích thước kernel (kh, kw).
    - stride: Bước trượt (mặc định 1).
    - padding: Padding ảnh (mặc định 0).

    Returns:
    - col_matrix: Ma trận (kh * kw, số cửa sổ tích chập).
    """
    H, W = image.shape
    kh, kw = kernel_size

    # Thêm padding
    image_padded = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    # Kích thước đầu ra sau tích chập
    out_h = (H + 2 * padding - kh) // stride + 1
    out_w = (W + 2 * padding - kw) // stride + 1

    # Tạo ma trận đầu ra
    col_matrix = np.zeros((kh * kw, out_h * out_w))

    col_index = 0
    for y in range(out_h):
        for x in range(out_w):
            patch = image_padded[y * stride:y * stride + kh, x * stride:x * stride + kw]
            col_matrix[:, col_index] = patch.flatten()
            col_index += 1

    return col_matrix

def flatten_kernel(kernel):
    """
    Chuyển kernel từ dạng (kh, kw) sang dạng vector hàng (1, kh * kw)

    Parameters:
    - kernel: NumPy array có kích thước (kh, kw)

    Returns:
    - kernel_flat: NumPy array có kích thước (1, kh * kw)
    """
    return kernel.reshape(1, -1)

# 🔹 Ví dụ sử dụng



# # 🔹 Load ảnh MNIST
# (x_train, y_train), _ = mnist.load_data()
# image = x_train[0] / 255.0  # Lấy ảnh đầu tiên và chuẩn hóa

image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
# 🔹 Chạy im2col
kernel_size = (5, 5)
stride = 1
padding = 0
image_im2col = im2col(image, kernel_size, stride, padding)
col_matrix = image_im2col[:, 0]
# 🔹 In kết quả
print(image)
print(col_matrix)
print("Ảnh đầu vào MNIST:", image.shape)
print("Ma trận im2col:", image_im2col.shape)

loaded = np.load("quantized_weights.npz")

for key in loaded.files:
    print(key)
    print(loaded[key].shape)



kernel = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])

kernel_flat = flatten_kernel(kernel)
print("Kernel gốc:\n", kernel)
print("\nKernel sau khi trải phẳng:\n", kernel_flat)