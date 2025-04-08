import numpy as np

def combine_to_32bit(a, b, c, d):
    # Convert to unsigned 8-bit by ensuring values are between 0 and 255
    a = a % 256
    b = b % 256
    c = c % 256
    d = d % 256
    result = (a << 24) | (b << 16) | (c << 8) | d
    return result

def to_signed_bin(n, bits):
    if n < 0:
        n = (1 << bits) + n  # Chuyển số âm sang two's complement
    return format(n, f'0{bits}b')  # Trả về chuỗi nhị phân với số bit cố định

# Load tệp .npz
loaded = np.load("quantized_weights.npz")

# Tạo danh sách chứa tất cả trọng số dạng một chiều
result = [loaded[key].flatten() for key in loaded.files]

# Gộp tất cả thành một mảng NumPy (nếu muốn)
concatenated_result = np.concatenate(result)

# In thông tin
print(f"Số lượng tensor: {len(result)}")
for i, arr in enumerate(result):
    print(f"Tensor {i}: Shape gốc {loaded[loaded.files[i]].shape} -> Sau khi flatten: {arr.shape}")

# In tổng số phần tử
print(f"Tổng số phần tử sau khi nối: {concatenated_result.shape[0]}")
print(concatenated_result[0:4])
concatenated_result = concatenated_result.astype(np.int32)
combined_32bit = []
print(hex(concatenated_result[0]))
print(to_signed_bin(concatenated_result[0], 8))
print(hex(combine_to_32bit(concatenated_result[0], concatenated_result[1], concatenated_result[2], concatenated_result[3])))
    # print(f"{num:08X}")  # Format the number as a 32-bit hex string, padded with zeros


print(concatenated_result[0] << 24)
print(type(concatenated_result[0]))
# Chuyển danh sách thành mảng NumPy với kiểu dữ liệu uint32
combined_32bit_array = np.array(combined_32bit, dtype=np.int32)
for i in range(0, len(concatenated_result), 4):
    combine_to_32bit(concatenated_result[i], concatenated_result[i+1], concatenated_result[i+2], concatenated_result[i+3])
# In kết quả
print("Mảng kết quả (32-bit):", combined_32bit_array)
# In kết quả
print(type(combined_32bit))  # Kiểm tra kiểu dữ liệu
# print(combined_32bit[0])  # In mảng kết quả

np.save("flattened_weights.npy", concatenated_result)
np.save("combine_32bit.npy", combined_32bit_array)
# print(hex(combined_32bit[0]))
for i in range(0, 20):
    print(concatenated_result[i])  # In từng phần tử dưới dạng hex

weights = np.load("combine_32bit.npy")  # allow object types

print("Type:", type(weights))
print("Content:", weights)