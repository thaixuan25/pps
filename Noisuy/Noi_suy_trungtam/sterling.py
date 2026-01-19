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
    if len(x) % 2 == 0:
        print("Số điểm dữ liệu chẵn, không thể áp dụng công thức Sterling.")
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
        row = f"{x[i]:^10.4f} | {table[i, 0]:^12.6f}"
        for j in range(1, i + 1):
            if i - j < n and j < n:
                row += f" | {table[i - j, j]:^12.6f}"
        print(row)
    print("-" * len(header))

# Áp dụng công thức Stirling
def stirling_interpolation(n, x, y, x_interpolate):
    """
    Thực hiện nội suy Stirling (hay công thức trung tâm Gauss).
    Công thức này là trung bình cộng của Gauss tiến và Gauss lùi, hiệu quả khi
    giá trị cần nội suy ở gần trung tâm bảng dữ liệu (-0.25 < t < 0.25).
    Yêu cầu số điểm dữ liệu phải là số lẻ.

    Công thức tổng quát:
    P(x) = y₀ + t⋅(Δy₀ + Δy₋₁)/2 + t²/2! ⋅ Δ²y₋₁ + t(t²-1²)/3! ⋅ (Δ³y₋₁ + Δ³y₋₂)/2 + t²(t²-1²)/4! ⋅ Δ⁴y₋₂ + ...

    Trong đó:
    - t = (x - x₀) / h
    - x₀ là điểm mốc được chọn ở chính giữa bảng dữ liệu.
    - h là bước nhảy.
    - Δ²y₋₁, Δ⁴y₋₂, ... là các sai phân trung tâm cấp chẵn.
    - (Δy₀ + Δy₋₁)/2, (Δ³y₋₁ + Δ³y₋₂)/2, ... là trung bình của các sai phân cấp lẻ.

    Các bước thực hiện thuật toán:
    1. Chuẩn bị dữ liệu: Đọc các điểm (x_i, y_i). Yêu cầu số điểm n phải là số lẻ.
       Kiểm tra các điểm x_i phải cách đều nhau.
    2. Chọn điểm gốc x₀: Do n lẻ, điểm gốc x₀ là điểm nằm chính giữa của dãy x.
    3. Tính tham số t: Tính t = (x - x₀) / h.
    4. Lập bảng sai phân: Xây dựng bảng sai phân từ các giá trị y.
    5. Áp dụng công thức:
       - Lấy các sai phân cần thiết từ bảng: các sai phân chẵn nằm trên dòng trung tâm x₀
         và trung bình cộng của các cặp sai phân lẻ đối xứng qua dòng trung tâm.
       - Thay các giá trị này và t vào công thức Stirling để tính kết quả nội suy.
    """
    h = x[1] - x[0]
    
    # Công thức Sterling yêu cầu số điểm lẻ, x0 là điểm chính giữa
    x0_index = n // 2
    x0 = x[x0_index]
    u_val = (x_interpolate - x0) / h

    # Bảng sai phân
    diff_table = forward_difference_table(y)
    print_difference_table(x, diff_table)
    print(f"\nQuy đổi: t = (x - x0) / h = ({x_interpolate} - {x0}) / {h} = {u_val:.6f}")
    print(f"Đặt u = t^2 = {u_val**2:.6f}")

    # ==============================================================================
    # REFACTOR THEO YÊU CẦU MỚI
    # ==============================================================================
    
    # 1. Tách y0 và thu thập các hệ số sai phân chẵn, lẻ
    print("\n" + "="*80)
    print("LỰA CHỌN CÁC HỆ SỐ SAI PHÂN".center(80))
    print("="*80)
    
    y0 = y[x0_index]
    c_even_terms = []
    c_odd_terms = []
    
    print(f"Hằng số ban đầu: y₀ = {y0:.6f}")

    for k in range(1, n):
        m = (k - 1) // 2
        if k % 2 != 0: # Lẻ, k = 2m + 1
            idx1 = x0_index - m - 1
            idx2 = x0_index - m
            if idx1 >= 0 and idx2 < n - k:
                val = (diff_table[idx1, k] + diff_table[idx2, k]) / 2
                c_odd_terms.append(val)
                print(f"Lấy hệ số lẻ bậc {k}: (Δ^{k}y({-m-1}) + Δ^{k}y({-m}))/2 = {val:.6f}")
        else: # Chẵn, k = 2m
            m = k // 2
            idx = x0_index - m
            if idx >= 0 and idx < n - k:
                val = diff_table[idx, k]
                c_even_terms.append(val)
                print(f"Lấy hệ số chẵn bậc {k}: Δ^{k}y({-m}) = {val:.6f}")
    
    # Kích thước ma trận Q sẽ dựa trên vector dài hơn
    N = max(len(c_even_terms), len(c_odd_terms))
    c_even_terms.extend([0.0] * (N - len(c_even_terms)))
    c_odd_terms.extend([0.0] * (N - len(c_odd_terms)))
    
    print(f"\nVector hệ số chẵn (W_even): {np.round(c_even_terms, 6).tolist()}")
    print(f"Vector hệ số lẻ (W_odd):   {np.round(c_odd_terms, 6).tolist()}")

    # 2. Xây dựng ma trận cơ sở Q duy nhất
    def poly_mul_by_monomial(p, c):
        new_poly = np.zeros(len(p) + 1)
        new_poly[:-1] = p
        new_poly[1:] -= c * p
        return new_poly

    q_polys = [np.array([1.0])] # Q_0(v) = 1
    roots = []
    for m in range(1, N):
        root = float(m**2) # Nghiệm là 1^2, 2^2, 3^2...
        roots.append(root)
        next_q = poly_mul_by_monomial(q_polys[-1], root)
        q_polys.append(next_q)
    
    # Tạo ma trận Q từ các đa thức cơ sở
    Q_matrix = np.zeros((N, N))
    for m in range(N):
        poly = q_polys[m]
        Q_matrix[m, -len(poly):] = poly
    
    print_q_matrix(Q_matrix, roots)

    # 3. Chia hệ số cho giai thừa tương ứng
    W_even_scaled = np.zeros(N)
    W_odd_scaled = np.zeros(N)
    for m in range(N):
        W_even_scaled[m] = c_even_terms[m] / math.factorial(2 * (m + 1))
        W_odd_scaled[m] = c_odd_terms[m] / math.factorial(2 * m + 1)

    # 4. Nhân vector hệ số với ma trận Q để ra đa thức P(v)
    print("\n" + "="*80)
    print("XÂY DỰNG ĐA THỨC QUA PHÉP NHÂN MA TRẬN".center(80))
    print("="*80)
    
    P_even_v = W_even_scaled @ Q_matrix
    P_odd_v = W_odd_scaled @ Q_matrix
    
    print(f"Đa thức P_even(v) = {np.round(P_even_v, 9).tolist()}")
    print(f"Đa thức P_odd(v)  = {np.round(P_odd_v, 9).tolist()}")

    # 5. Ghép lại thành đa thức P(u) và cộng hằng số y0
    # P(u) = y₀ + u² * P_even(v) + u * P_odd(v)
    
    # Bậc cao nhất có thể là 2N (từ u²*v^{N-1}), nên độ dài đa thức là 2N+1
    final_poly_len = 2 * N + 1
    P_u = np.zeros(final_poly_len)

    # Đặt hằng số y₀ vào vị trí bậc 0
    P_u[-1] = y0
    
    # Ghép các hệ số của P_even_v vào các vị trí bậc chẵn, bắt đầu từ u²
    # P_even_v: [c_{N-1}, ..., c_0] tương ứng với v^{N-1}, ..., v^0
    # Hệ số c_m của v^m sẽ là hệ số của u^{2m+2}
    if N > 0:
        for m in range(N):
            c_m = P_even_v[(N - 1) - m]  # Lấy c_m, từ c_0 đến c_{N-1}
            idx_even = final_poly_len - 1 - (2 * m + 2)
            if idx_even >= 0:
                P_u[idx_even] += c_m

    # Ghép các hệ số của P_odd_v vào các vị trí bậc lẻ, bắt đầu từ u¹
    # P_odd_v: [d_{N-1}, ..., d_0] tương ứng với v^{N-1}, ..., v^0
    # Hệ số d_m của v^m sẽ là hệ số của u^{2m+1}
    if N > 0:
        for m in range(N):
            d_m = P_odd_v[(N - 1) - m] # Lấy d_m, từ d_0 đến d_{N-1}
            idx_odd = final_poly_len - 1 - (2 * m + 1)
            if idx_odd >= 0:
                P_u[idx_odd] += d_m
    
    # Bỏ các hệ số 0 ở đầu đa thức
    first_nonzero = np.where(np.abs(P_u) > 1e-12)[0]
    if len(first_nonzero) > 0:
        P_u = P_u[first_nonzero[0]:]
    
    print("\n" + "="*80)
    print("ĐA THỨC NỘI SUY HOÀN CHỈNH P(u)".center(80))
    print("="*80)
    for i, c in enumerate(P_u):
        deg = len(P_u) - 1 - i
        if abs(c) < 1e-12: continue
        print(f"Hệ số của u^{deg}: {c: 15.6g}")

    # 6. Tính giá trị nội suy bằng Horner
    y_interp = 0.0
    for coeff in P_u:
        y_interp = y_interp * u_val + coeff
        
    print(f"\nVới t = {u_val:.6f}, thì u = t^2 = {u_val**2:.6f}, giá trị nội suy là:")
    print(f"P({u_val:.6f}) = {y_interp:.6f}")

    return y_interp

def print_q_matrix(q_matrix, roots):
    """In ma trận cơ sở Q."""
    print("\n" + "="*120)
    print("MA TRẬN CƠ SỞ Q(v)".center(120))
    print("="*120)

    N = len(q_matrix)
    CELL_WIDTH = 15
    header = f"{'m':<4} | {'Nghiệm nhân (k^2)':<{CELL_WIDTH+5}} |"
    for j in reversed(range(N)):
        header += f" {'v^'+str(j):^{CELL_WIDTH}} |"
    print(header)
    print("-" * len(header))

    # Hàng m=0
    row_str_0 = f"{0:<4} | {'-':<{CELL_WIDTH+5}} |"
    row_str_0 += " " * ((N - 1) * (CELL_WIDTH + 2))
    row_str_0 += f" {q_matrix[0, -1]:^{CELL_WIDTH}.6g} |"
    print(row_str_0)

    # Các hàng m > 0
    for m in range(1, N):
        poly_coeffs = q_matrix[m]
        root_val = roots[m-1]
        row_str = f"{m:<4} | {root_val:<{CELL_WIDTH+5}.6g} |"
        
        padding = N - len(np.trim_zeros(poly_coeffs, 'f'))
        row_str += " " * (padding * (CELL_WIDTH + 2))
        
        first_coeff_idx = -1
        for i, c in enumerate(poly_coeffs):
            if abs(c) > 1e-12:
                first_coeff_idx = i
                break
        
        if first_coeff_idx != -1:
            for coeff in poly_coeffs[first_coeff_idx:]:
                row_str += f" {coeff:^{CELL_WIDTH}.6g} |"

        print(row_str)
    print("-" * len(header))


def main():
    try:
        n, x, y, x_interpolate = read_data("Noi_suy_trungtam/input.txt")
        
        if n != len(x) or n != len(y):
            print("Lỗi: Số lượng điểm n không khớp với số lượng giá trị x hoặc y.")
            return

        print(f"Số điểm dữ liệu n = {n}")
        print(f"Các điểm x: {x}")
        print(f"Các điểm y: {y}")
        print("-" * 30)

        result = stirling_interpolation(n, x, y, x_interpolate)
        
        print(f"\nGiá trị nội suy tại x = {x_interpolate} là y = {result:.6f}")

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'Noi_suy_trungtam/input.txt'. Vui lòng tạo file và nhập dữ liệu.")
    except Exception as e:
        print(f"Đã có lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()
