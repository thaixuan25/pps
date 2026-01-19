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
            row += f" | {table[i - j, j]:^12.6f}"
        print(row)
    print("-" * len(header))


# Hàm in bảng tính tích Gauss G_k(t)
def in_bang_tich_gauss_lui(n):
    print("\n" + "="*120)
    print("BẢNG TÍNH TÍCH GAUSS LÙI G_k(t)".center(120))
    print("="*120)
    
    # product_poly_coeffs[k][j] sẽ là hệ số của t^j trong G_k(t)
    product_poly_coeffs = np.zeros((n + 1, n + 1))
    product_poly_coeffs[0, 0] = 1  # G_0(t) = 1

    p_values = ['-'] # p for k=0 is undefined

    for k in range(1, n + 1):
        # G_k(t) = G_{k-1}(t) * (t - p_k)
        
        # Tính p_k tương ứng với công thức Gauss lùi
        if k % 2 == 0: # k chẵn
            p = -(k // 2)
        else: # k lẻ
            p = (k - 1) // 2
        p_values.append(p)
            
        # t*G_{k-1}(t) -> dịch hệ số sang phải 1 bậc
        for j in range(1, k + 1):
            product_poly_coeffs[k, j] = product_poly_coeffs[k-1, j-1]
        # -p*G_{k-1}(t) -> nhân hệ số cũ với -p
        for j in range(k):
            product_poly_coeffs[k, j] -= p * product_poly_coeffs[k-1, j]
            
    # In header của bảng, theo thứ tự bậc giảm dần
    CELL_WIDTH = 12
    header = f"{'p':<6} |"
    for j in reversed(range(n + 1)):
        header_cell = f"t^{j}"
        header += f" {header_cell:>{CELL_WIDTH - 3}} |"
    print(header)
    print("-" * len(header))
    
    # In các hàng của bảng
    for k in range(n + 1):
        p_str = str(p_values[k])
        row_str = f" {p_str:<5} |"
        
        # Thêm khoảng trống ở đầu để tạo thành tam giác vuông bên phải
        padding_cells = n - k
        row_str += " " * (padding_cells * CELL_WIDTH)
        
        # In các hệ số theo thứ tự bậc giảm dần
        for j in reversed(range(k + 1)):
            row_str += f" {product_poly_coeffs[k, j]:>{CELL_WIDTH - 3}.0f} |"
        print(row_str)
    print("=" * len(header))


# Áp dụng công thức Gauss lùi
def gauss_backward_interpolation(n, x, y, x_interpolate):
    h = x[1] - x[0]
    
    # Tìm chỉ số trung tâm
    mid = n // 2
    if n % 2 == 0:
        # Nếu n chẵn, chọn giá trị gần x_interpolate hơn
        if abs(x_interpolate - x[mid]) < abs(x_interpolate - x[mid - 1]):
            x0_index = mid
        else:
            x0_index = mid -1
    else:
        x0_index = mid

    x0 = x[x0_index]
    
    u = (x_interpolate - x0) / h

    diff_table = forward_difference_table(y)
    
    print_difference_table(x, diff_table)

    # In bảng nhân tích Gauss
    in_bang_tich_gauss_lui(n - 1)

    # Lấy ra các hệ số sai phân được chọn
    selected_diffs = []
    for k in range(n):
        diff_index = x0_index - ((k + 1) // 2)
        
        if diff_index < 0 or diff_index >= n - k:
            break # Dừng nếu chỉ số ra ngoài bảng
            
        selected_diffs.append(diff_table[diff_index, k])
    
    print("\nCác hệ số sai phân được chọn (y₀, Δy₋₁, Δ²y₋₁, ...):")
    seq = [-( (i+1) // 2) for i in range(1, len(selected_diffs)+1)]
    for i in range(len(selected_diffs)):
        print(f"Δ^{i}y{seq[i]} = {selected_diffs[i]:.6f}")

    # In các tính toán chi tiết
    y_interp = y[x0_index]
    u_product = 1.0
    polynomial_str = f"{y[x0_index]:.6f}"
    u_product_str_so_far = "1"

    # Khởi tạo đa thức symbolic nếu có sympy
    if sympy_available:
        t = symbols('t')
        P_t_symbolic = y[x0_index]
        t_product_symbolic = 1

    print(f"\nĐặt t = (x - x0) / h = ({x_interpolate} - {x0}) / {h} = {u:.6f}\n")
    print("Bảng tính các hệ số và giá trị:")
    header = f"{'k':^3} | {'p':^5} | {'Tích t':<30} | {'Sai phân (Δ^k y)':^20} | {'Hệ số P(t)':^15} | {'Giá trị số hạng':^20}"
    print(header)
    print("-" * len(header))
    print(f"{'0':^3} | {'-':^5} | {u_product_str_so_far:<30} | {y[x0_index]:^20.6f} | {y[x0_index]:^15.6f} | {y[x0_index]:^20.6f}")
    
    fact = 1.0
    
    for k in range(1, n):
        fact *= k
        
        # Xây dựng tích các term của t theo công thức Gauss lùi
        # p_k nhận các giá trị 0, -1, 1, -2, 2, ...
        if k % 2 == 0: # k chẵn
            p = -(k // 2)
        else: # k lẻ
            p = (k - 1) // 2

        # Tính giá trị số của tích
        multiplier = u - p
        u_product *= multiplier

        # Xây dựng chuỗi hiển thị cho tích
        if p == 0:
            term_str = "t"
        elif p > 0:
            term_str = f"(t-{p})"
        else: # p < 0
            term_str = f"(t+{-p})"

        if k == 1:
            u_product_str_so_far = term_str
        elif k % 2 == 0: # k chẵn, nối vào sau
            u_product_str_so_far += term_str
        else: # k lẻ, nối vào trước
            u_product_str_so_far = term_str + u_product_str_so_far
        
        # Lấy giá trị sai phân tương ứng
        diff_index = x0_index - ((k + 1) // 2)
        
        if diff_index < 0 or diff_index >= n - k:
            break
            
        diff = diff_table[diff_index, k]
        
        # Tính toán
        coeff = diff / fact
        term_value = coeff * u_product
        y_interp += term_value
        
        # Xây dựng chuỗi đa thức P(t)
        if coeff >= 0:
            polynomial_str += f" + {coeff:.6f}*{u_product_str_so_far}"
        else:
            polynomial_str += f" - {-coeff:.6f}*{u_product_str_so_far}"
        
        # Cập nhật đa thức symbolic
        if sympy_available:
            t_product_symbolic *= (t - p)
            P_t_symbolic += coeff * t_product_symbolic
            
        # In ra bảng
        print(f"{k:^3} | {p:^5} | {u_product_str_so_far:<30} | {diff:^20.6f} | {coeff:15.6g} | {term_value:^20.6f}")
        
    print(f"\nĐa thức nội suy P(t) dạng nhân tử là:")
    print(f"P(t) = {polynomial_str}")
    
    # In đa thức đã khai triển và các hệ số
    if sympy_available:
        try:
            P_t_expanded = expand(P_t_symbolic)
            print(f"\nĐa thức P(t) sau khi khai triển:")
            print(f"P(t) = {P_t_expanded}")

            # In các hệ số của đa thức khai triển
            print("\nCác hệ số của đa thức P(t):")
            poly = P_t_expanded.as_poly(t)
            coeffs = poly.all_coeffs()
            degree = poly.degree()
            
            # In từ bậc cao đến thấp
            for i, coeff_val in enumerate(coeffs):
                power = degree - i
                print(f"Hệ số của t^{power}: {float(coeff_val):15.6g}")
        except Exception as e:
            print(f"\nLỗi khi xử lý đa thức symbolic: {e}")
    else:
        print("\nĐể xem đa thức khai triển, vui lòng cài đặt thư viện 'sympy' (chạy: pip install sympy)")
        
    return y_interp

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

        result = gauss_backward_interpolation(n, x, y, x_interpolate)
        
        print(f"\nGiá trị nội suy tại x = {x_interpolate} là y = {result:.6f}")

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'Noi_suy_trungtam/input.txt'. Vui lòng tạo file và nhập dữ liệu.")
    except Exception as e:
        print(f"Đã có lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()
