import numpy as np
import torch
# Ma trận đầu vào dạng int8
# Tạo tensor với giá trị int8
def int8_to_bin(value):
    value = np.int8(value)  # Đảm bảo nằm trong kiểu int8
    if value < 0:
        value = (256 + value)  # Chuyển sang bù 2
    return format(value, '08b')  # Chuyển thành chuỗi nhị phân 8-bit
tensor = np.array([[[56, 37, -54, -78, 37],
                    [107, -44, -38, -75, 27],
                    [127, -32, -61, -48, 62],
                    [86, -11, -64, 26, 26],
                    [54, 18, -6, 10, -42]]], dtype=np.int8)

# In tensor
print("Tensor:")
print(tensor)
print(type(tensor))
# Duyệt qua từng phần tử trong tensor và in nhị phân
print("\nBinary representation of each element:")
for row in tensor[0]:  # Duyệt qua hàng (mảng 2D trong tensor)
    for value in row:  # Duyệt qua từng phần tử trong hàng
        print('{0:08b}'.format(value))
        print(int8_to_bin(value))
# Hàm chuyển đổi một giá trị int8 sang dạng nhị phân 8-bit có dấu
# def int8_to_bin(value):
#     # Kiểm tra xem giá trị có trong phạm vi của int8 hay không
#     if value < -128:
#         value = -128  # Giới hạn nhỏ nhất của int8
#     elif value > 127:
#         value = 127  # Giới hạn lớn nhất của int8
#
#     # Nếu giá trị âm, chuyển thành dạng 2's complement
#     if value < 0:
#         value = (value + 256) % 256  # Chuyển giá trị âm thành 2's complement trong phạm vi 8 bit
#
#     return format(value, '08b')  # Lấy 8 bit thấp nhất

# Hàm tạo Verilog code cho ma trận 5x5
# def generate_verilog_code(tensor):
#     rows, cols = tensor.shape[1], tensor.shape[2]  # Lấy kích thước của tensor (5x5)
#     verilog_code = "module matrix_5x5_to_8bit;\n"
#     verilog_code += f"    // Khai báo ma trận {rows}x{cols}, mỗi phần tử là giá trị 8-bit signed\n"
#     verilog_code += f"    reg signed [7:0] matrix [0:{rows - 1}][0:{cols - 1}];\n\n"
#     verilog_code += "    initial begin\n"
#
#     # Duyệt qua từng phần tử trong tensor và thêm vào Verilog code
#     for i in range(rows):
#         for j in range(cols):
#             bin_value = int8_to_bin(tensor[0, i, j])  # Chuyển đổi giá trị sang nhị phân
#             verilog_code += f"        matrix[{i}][{j}] = 8'b{bin_value};  // {tensor[0, i, j]}\n"
#
#     verilog_code += "    end\n"
#     verilog_code += "endmodule\n"
#     return verilog_code
#
#
# # Sinh Verilog code
# verilog_code = generate_verilog_code(tensor)
#
# # Lưu vào file .v
# with open('verilog_matrix_5x5.v', 'w') as f:
#     f.write(verilog_code)
#
# # In kết quả
# print(verilog_code)
