import numpy as np

def matrix_chain_order(p):
    n = len(p) - 1
    m = np.zeros((n, n))  # Ma trận lưu số phép nhân tối thiểu
    s = np.zeros((n, n), dtype=int)  # Ma trận lưu vị trí cắt tốt nhất

    for l in range(2, n+1):  # l là độ dài chuỗi ma trận
        for i in range(n-l+1):
            j = i + l - 1
            m[i, j] = float('inf')
            for k in range(i, j):
                q = m[i, k] + m[k+1, j] + p[i] * p[k+1] * p[j+1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k

    return m, s

def print_optimal_parens(s, i, j):
    if i == j:
        print(f"A{i+1}", end="")
    else:
        print("(", end="")
        print_optimal_parens(s, i, s[i, j])
        print(" × ", end="")
        print_optimal_parens(s, s[i, j] + 1, j)
        print(")", end="")

# Ví dụ
#p = [10, 20, 30, 40, 30]
p = [1, 512, 128, 32, 10]
m, s = matrix_chain_order(p)

print("Số phép nhân tối thiểu:", m[0, len(p)-2])
print("Cách đặt dấu ngoặc tối ưu:")
print_optimal_parens(s, 0, len(p)-2)
