import numpy as np

def twos_complement(value, bit_width):
    """ Chuyển đổi số bù hai nếu giá trị âm """
    if value >= (1 << (bit_width - 1)):  # Nếu bit MSB = 1 (số âm)
        value -= (1 << bit_width)  # Chuyển về giá trị âm đúng
    return value

def read_coe_to_24x24_array(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Loại bỏ ký tự xuống dòng để ghép thành một dòng duy nhất
    content = content.replace("\n", " ").replace("\r", " ")

    # Lấy phần sau "memory_initialization_vector="
    if "memory_initialization_vector=" in content:
        content = content.split("memory_initialization_vector=")[1]

    # Xóa dấu ',' và ';' rồi tách thành danh sách các số nhị phân 32-bit
    data_list = content.replace(",", " ").replace(";", "").split()

    # Chỉ giữ lại các chuỗi có đúng 32-bit nhị phân
    clean_data = [x for x in data_list if len(x) == 32 and all(c in '01' for c in x)]

    print(f"Đọc được {len(clean_data)} giá trị nhị phân 32-bit từ file COE.")

    # Kiểm tra số lượng phần tử hợp lệ
    if len(clean_data) < 144:
        raise ValueError(f"File COE không đủ dữ liệu! Cần ít nhất 144 giá trị 32-bit, nhưng chỉ có {len(clean_data)}.")

    # Chỉ lấy 144 giá trị đầu tiên để tạo ra 576 byte 8-bit
    clean_data = clean_data[:144]

    # Chuyển từ chuỗi nhị phân (32-bit có dấu) sang số nguyên
    int32_list = [twos_complement(int(x, 2), 32) for x in clean_data]

    # Chuyển mỗi 32-bit thành 4 số 8-bit (`int8`)
    int8_list = []
    for value in int32_list:
        for i in range(4):  # Cắt thành 4 byte
            byte = (value >> (8 * (3 - i))) & 0xFF  # Lấy từng byte
            int8_list.append(np.int8(twos_complement(byte, 8)))  # Chuyển thành số `int8`

    # Chỉ lấy 576 giá trị đầu tiên (24x24)
    int8_list = int8_list[:576]

    # Chuyển danh sách thành mảng NumPy 2D (24x24)
    matrix_24x24 = np.array(int8_list, dtype=np.int8).reshape(24, 24)

    return matrix_24x24

# Ví dụ sử dụng
filename = "memory.coe"
array_2d = read_coe_to_24x24_array(filename)

print("Mảng 24x24 từ file COE:")
for row in array_2d:
    print(' '.join(f'{num:3d}' for num in row))  # `{num:3d}` giúp căn chỉnh đẹp