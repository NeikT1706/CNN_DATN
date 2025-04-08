import numpy as np

weights_8bit = np.load("flattened_weights.npy")  # int8 hoặc uint8

# Đảm bảo là uint8 trước khi gộp (dù dữ liệu gốc là int8)
weights_8bit = weights_8bit.astype(np.uint8)

# Đảm bảo chia hết cho 4
assert weights_8bit.size % 4 == 0

# Reshape
weights_reshaped = weights_8bit.reshape(-1, 4)

# Gộp theo little-endian: byte[0] là LSB
weights_32bit = (
    (weights_reshaped[:, 0].astype(np.uint32)) |
    (weights_reshaped[:, 1].astype(np.uint32) << 8) |
    (weights_reshaped[:, 2].astype(np.uint32) << 16) |
    (weights_reshaped[:, 3].astype(np.uint32) << 24)
)

# Test in kết quả
print("First packed word (hex):", hex(weights_32bit[0]))  # ← nên là 0x1F127FB5
print(weights_32bit.shape)

for i in range(weights_32bit.shape[0]):
    print(hex(weights_32bit[i]))

np.save("weights_32bit.npy", weights_32bit)