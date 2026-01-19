import numpy as np
import math
import sys
try:
    from sympy import symbols, expand
    sympy_available = True
except ImportError:
    sympy_available = False

# Đọc dữ liệu từ file input.txt
def read_data(filename="input.txt"):
    x = []  # Danh sách lưu các giá trị x
    y = []  # Danh sách lưu các giá trị y tương ứng
    
    # Mở file và đọc dữ liệu
    with open(filename,'r+') as f: # đọc file input
        for line in f.readlines(): # duyệt từng hàng trong file
            parts = line.split()
            if not parts:
                continue
            if float(parts[0]) not in x:
                x.append(float(parts[0]))
                y.append(float(parts[1]))
            else:
                print(f"x[{float(parts[0])}] da ton tai")

    x = np.array(x)
    y = np.array(y)
    
    for i in range(1, len(x)): # kiếm tra cách đều
        if abs(x[i] - x[i-1] - (x[1] - x[0])) > 1e-6:
            print("Các điểm không cách đều")
            sys.exit()
    if len(x) % 2 == 1:
        print("Số điểm dữ liệu lẻ, không thể áp dụng công thức Bessel.")
        sys.exit()
    x_interpolate = float(input("Nhập giá trị x cần nội suy: "))
    return len(x), np.array(x), np.array(y), x_interpolate

# Tính bảng sai phân
def forward_difference_table(y):
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i + 1, j - 1] - table[i, j - 1]
    return table

def print_difference_table(x, table):
    n = len(x)
    print("\nBảng sai phân:")
    
    # Header
    header = f"{'x_i':^10} | {'y_i':^12}"
    for i in range(1, n):
        header += f" | {'Δ^'+str(i)+'y':^12}"
    print(header)
    print("-" * len(header))

    # Rows
    for i in range(n):
        row_str = f"{x[i]:^10.4f} | {table[i, 0]:^12.6f}"
        for j in range(1, i + 1):
             row_str += f" | {table[i - j, j]:^12.6f}"
        print(row_str)
    print("-" * len(header))


# Áp dụng công thức Bessel
def bessel_interpolation(n, x, y, x_interpolate):
    h = x[1] - x[0]
    x0_index = (n // 2) - 1
    x0 = x[x0_index]
    t_val = (x_interpolate - x0) / h

    # Bảng sai phân
    diff_table = forward_difference_table(y)
    print_difference_table(x, diff_table)
    print(f"\nQuy đổi: t = (x - x0) / h = ({x_interpolate} - {x0}) / {h} = {t_val:.6f}")
    print(f"Đặt u = t - 0.5 = {t_val:.6f} - 0.5 = {t_val - 0.5:.6f}")

    # ==============================================================================
    # REFACTOR THEO YÊU CẦU MỚI
    # ==============================================================================

    # 1. Thu thập đầy đủ các vector hệ số sai phân chẵn và lẻ
    print("\n" + "="*80)
    print("LỰA CHỌN CÁC HỆ SỐ SAI PHÂN".center(80))
    print("="*80)
    
    y0_avg = (y[x0_index] + y[x0_index + 1]) / 2
    delta_y0 = diff_table[x0_index, 1] if x0_index < n - 1 else 0.0
    print(f"Hệ số chẵn ban đầu: (y₀+y₁)/2 = {y0_avg:.6f}")
    print(f"Hệ số lẻ ban đầu:   Δ¹y₀ = {delta_y0:.6f}")
    
    c_even_terms = []
    c_odd_terms = []
    for k in range(2, n):
        m = k // 2
        if k % 2 == 0: # chẵn
            idx1 = x0_index - m
            idx2 = x0_index - m + 1
            if idx1 >= 0 and idx2 < n - k: 
                val = (diff_table[idx1, k] + diff_table[idx2, k]) / 2
                c_even_terms.append(val)
                print(f"Lấy hệ số chẵn bậc {k}: (Δ^{k}y({-m}) + Δ^{k}y({-m+1}))/2 = {val:.6f}")
        else: # lẻ
            idx = x0_index - m
            if idx >= 0 and idx < n - k: 
                val = diff_table[idx, k]
                c_odd_terms.append(val)
                print(f"Lấy hệ số lẻ bậc {k}:   Δ^{k}y({-m}) = {val:.6f}")

    c_even_full = [y0_avg] + c_even_terms
    c_odd_full = [delta_y0] + c_odd_terms
    
    # Đảm bảo 2 vector dài bằng nhau để tạo ma trận vuông
    N = max(len(c_even_full), len(c_odd_full))
    c_even_full.extend([0.0] * (N - len(c_even_full)))
    c_odd_full.extend([0.0] * (N - len(c_odd_full)))

    print(f"Vector hệ số chẵn (W_even): {np.round(c_even_full, 6).tolist()}")
    print(f"Vector hệ số lẻ (W_odd):   {np.round(c_odd_full, 6).tolist()}")
    
    # 2. Xây dựng ma trận cơ sở Q
    print("\n" + "="*120)
    print("MA TRẬN CƠ SỞ Q_k(u)".center(120))
    print("="*120)
    
    def poly_mul_by_monomial(p, c):
        """Nhân đa thức p (dạng mảng) với (u - c) - 'Nhân Horner'."""
        new_poly = np.zeros(len(p) + 1)
        new_poly[:-1] = p      # Tương ứng với u * p
        new_poly[1:] -= c * p  # Tương ứng với -c * p
        return new_poly

    q_polys = [np.array([1.0])] # Q_0(u) = 1
    roots = []
    for k in range(N - 1):
        root = ((2 * k + 1)**2) / 4.0
        roots.append(root)
        next_q = poly_mul_by_monomial(q_polys[-1], root)
        q_polys.append(next_q)

    # In bảng Q
    max_coeffs = len(q_polys[-1]) if q_polys else 1
    CELL_WIDTH = 15
    header = f"{'k':<4} | {'Hệ số nhân Horner':<{CELL_WIDTH+5}} |"
    for j in reversed(range(max_coeffs)):
        header += f" {'u^'+str(j):^{CELL_WIDTH}} |"
    print(header)
    print("-" * len(header))

    # Hàng k=0
    row_str_0 = f"{0:<4} | {'-':<{CELL_WIDTH+5}} |"
    padding_0 = max_coeffs - len(q_polys[0])
    row_str_0 += " " * (padding_0 * (CELL_WIDTH + 2))
    row_str_0 += f" {q_polys[0][0]:^{CELL_WIDTH}.6g} |"
    print(row_str_0)

    # Các hàng k > 0
    for k in range(1, len(q_polys)):
        poly = q_polys[k]
        root_val = roots[k-1]
        row_str = f"{k:<4} | {root_val:<{CELL_WIDTH+5}.6g} |"
        padding = max_coeffs - len(poly)
        row_str += " " * (padding * (CELL_WIDTH + 2))
        for coeff in poly:
            row_str += f" {coeff:^{CELL_WIDTH}.6g} |"
        print(row_str)
    print("-" * len(header))
    
    # 3. Xây dựng các ma trận cơ sở vuông M_even và M_odd
    print("\n" + "="*80)
    print("XÂY DỰNG ĐA THỨC QUA PHÉP NHÂN MA TRẬN".center(80))
    print("="*80)

    M_even = np.zeros((N, N))
    M_odd = np.zeros((N, N))
    
    M_even[0, -1] = 1.0 # Cơ sở cho y0_avg là P(u)=1
    M_odd[0, -1] = 1.0  # Cơ sở cho delta_y0 là P(u)=1

    for i in range(1, N):
        # Cơ sở cho c_even_full[i] (sai phân bậc 2i) là Q_i(u) / (2i)!
        poly_even_basis = q_polys[i] / math.factorial(2 * i)
        M_even[i, -len(poly_even_basis):] = poly_even_basis

        # Cơ sở cho c_odd_full[i] (sai phân bậc 2i+1) là Q_i(u) / (2i+1)!
        poly_odd_basis = q_polys[i] / math.factorial(2 * i + 1)
        M_odd[i, -len(poly_odd_basis):] = poly_odd_basis
        
    # 4. Tính đa thức chẵn và lẻ
    W_even = np.array(c_even_full)
    W_odd = np.array(c_odd_full)

    P_even_u = W_even @ M_even
    P_odd_u = W_odd @ M_odd
    
    print(f"Đa thức P_even(u) = {np.round(P_even_u, 9).tolist()}")
    print(f"Đa thức P_odd(u)  = {np.round(P_odd_u, 9).tolist()}")

    # 5. Ghép lại thành đa thức P(t') cuối cùng
    # P(t') = P_even(t'^2) + t' * P_odd(t'^2)
    
    P_even_t_prime = np.zeros(2 * len(P_even_u) - 1)
    if len(P_even_u) > 0: P_even_t_prime[::2] = P_even_u
    
    P_odd_t_prime = np.zeros(2 * len(P_odd_u))
    if len(P_odd_u) > 0: P_odd_t_prime[:-1:2] = P_odd_u

    max_len = max(len(P_even_t_prime), len(P_odd_t_prime))
    P_t_prime = np.pad(P_even_t_prime, (max_len - len(P_even_t_prime), 0)) + \
                np.pad(P_odd_t_prime, (max_len - len(P_odd_t_prime), 0))
    
    # 6. Khai triển đa thức P(t') thành P(t) bằng cách thay t' = t - 0.5
    print("\n" + "="*80)
    print("Đa thức P(t)".center(80))
    for i, c in enumerate(P_t_prime):
        deg = len(P_t_prime) - 1 - i
        if abs(c) < 1e-9: continue
        print(f"({c: 15.6g}) * t^{deg}")

    def expand_poly_from_t_prime(p_t_prime, c=0.5):
        """Khai triển đa thức P(t)"""
        final_poly = np.zeros(len(p_t_prime))
        n = len(p_t_prime)
        for i in range(n): # i là bậc của t'
            coeff = p_t_prime[n - 1 - i]
            if abs(coeff) < 1e-9: continue
            
            # Khai triển coeff * (t - c)^i
            for j in range(i + 1): # j là bậc của t
                # Hệ số của t^j trong (t-c)^i là C(i,j) * (-c)^(i-j)
                binomial_coeff = math.comb(i, j) * ((-c)**(i - j))
                # Cộng vào hệ số của t^j trong đa thức cuối cùng
                final_poly[n - 1 - j] += coeff * binomial_coeff
        return final_poly
        
    P_t = expand_poly_from_t_prime(P_t_prime)
    
    print(f"Thay t' = t - 0.5, ta có đa thức cuối cùng P(t'):")
    for i, c in enumerate(P_t):
        deg = len(P_t) - 1 - i
        if abs(c) < 1e-9: continue
        print(f"({c: 15.6g}) * t^{deg}")

    # 7. Tính giá trị nội suy
    y_interp = np.polyval(P_t, t_val)
    
    print(f"\nVới t = {t_val:.6f}, thì t' = {t_val - 0.5:.6f}, giá trị nội suy là:")
    print(f"P(t={t_val:.6f}) = {y_interp:.6f}")

    return y_interp

def main():
    try:
        n, x, y, x_interpolate = read_data("Noi_suy_trungtam/input.txt")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file input.txt.")
        return
    except ValueError:
        print("Lỗi: Dữ liệu không hợp lệ trong file input.txt.")
        return

    if n != len(x) or n != len(y):
        print("Lỗi: Dữ liệu n, x, y không khớp.")
        return

    print(f"Số điểm dữ liệu n = {n}\nCác điểm x: {x}\nCác điểm y: {y}\n" + "-"*30)
    result = bessel_interpolation(n, x, y, x_interpolate)
    print(f"\nGiá trị nội suy tại x = {x_interpolate} là y = {result:.6f}")

if __name__ == "__main__":
    main()
