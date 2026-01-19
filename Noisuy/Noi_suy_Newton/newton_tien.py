from math import *
import numpy as np
import matplotlib.pyplot as plt
import sys

np.set_printoptions(suppress=True, linewidth=np.inf, precision=9) #chỉnh số chữ số sau dấu phẩy

def Input():  #giả thiết input được sắp xếp cách đều tăng dần
    x, y = [], []
    with open('Noi_suy_Newton/input.txt','r+') as f: # đọc file input
        for line in f.readlines(): # duyệt từng hàng trong file
            if float(line.split()[0]) not in x:
                x.append(float(line.split()[0]))
                y.append(float(line.split()[1]))
    for i in range(1, len(x)): # kiếm tra cách đều
        if abs(x[i] - x[i-1] - (x[1] - x[0])) > 1e-6:
            print("Các điểm không cách đều")
            sys.exit()
    return x, y
    
#chọn ra num điểm gần x0 nhất
def pickPoint(x0, num, x, y):
    if num > len(x):
        print("Lỗi: Số lượng mốc nội suy yêu cầu lớn hơn số lượng mốc trong dữ liệu.")
        sys.exit()

    # Tìm chỉ số `i` của mốc lớn nhất mà nhỏ hơn hoặc bằng x0.
    # Mốc x[i] này là lựa chọn tốt cho điểm bắt đầu (x_0) trong công thức Newton tiến,
    # giúp cho giá trị t = (x0 - x_start)/h nhỏ và dương, tăng độ chính xác.
    # np.searchsorted tìm vị trí chèn, ta lấy vị trí bên phải rồi trừ 1.
    i = np.searchsorted(x, x0, side='right') - 1
    
    # Đảm bảo chỉ số không bị âm nếu x0 nhỏ hơn tất cả các mốc x.
    i = max(0, i)

    # Xác định chỉ số bắt đầu (`start_index`) của dãy mốc sẽ được chọn.
    # Ta muốn cửa sổ `num` điểm bắt đầu từ `i`, nhưng phải đảm bảo không vượt
    # ra ngoài biên phải của mảng. Nếu có, ta dịch cửa sổ sang trái.
    start_index = max(0, min(i, len(x) - num))
    
    end_index = start_index + num
    
    x1 = x[start_index:end_index]
    y1 = y[start_index:end_index]

    return x1, y1

def newton_forward_interpolation(x1, y1):
    n = len(x1) - 1
    h = x1[1] - x1[0]
    
    # 1. Tính và lưu trữ bảng sai phân tiến
    diff_table = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        diff_table[i][0] = y1[i]
    
    for j in range(1, n + 1):
        for i in range(n + 1 - j):
            diff_table[i][j] = diff_table[i+1][j-1] - diff_table[i][j-1]
            
    # Lấy các sai phân tiến cần thiết từ dòng đầu của bảng (đường chéo)
    forward_diffs = diff_table[0]

    # 2. Tính các hệ số b_k = Δ^k(y_0) / k!
    newton_coeffs = np.zeros(n + 1)
    for k in range(n + 1):
        newton_coeffs[k] = forward_diffs[k] / factorial(k)
        
    return newton_coeffs, diff_table

#vẽ hình
def plot(x, y, x_all, y_all, x0, y0, P_t, h, x_start):
    plt.clf()
    plt.title("Newton Forward Interpolation")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(x_all, y_all, color='gray', label='All data points')
    plt.scatter(x, y, color='red', label='Interpolation nodes')

    # Vẽ đường cong nội suy
    x_curve = np.linspace(min(x), max(x), 400)
    t_curve = (x_curve - x_start) / h
    y_curve = P_t(t_curve)
    plt.plot(x_curve, y_curve, color='blue', label="Newton Polynomial")
    
    # Đánh dấu điểm nội suy
    plt.scatter(x0, y0, color='green', zorder=5, s=100, marker='*', label=f'Interpolated Point ({x0:.2f}, {y0:.4f})')

    plt.legend()
    plt.grid(True)
    plt.savefig("mygraph.png")
    # plt.show()

# Hàm in bảng sai phân đều
def in_bang_sai_phan(x1, diff_table):
    n = len(x1)
    print("\n" + "="*90)
    print("BẢNG SAI PHÂN TIẾN (FORWARD DIFFERENCE TABLE)".center(90))
    print("="*90)
    
    # In header
    header = f"{'x[i]':>10} | {'y[i]':>12}"
    for j in range(1, n):
        col_header = f"Δ^{j}y"
        header += f" | {col_header:<12}"
    print(header)
    print("-" * len(header))
    
    # In các hàng của bảng
    for i in range(n):
        row_str = f"{x1[i]:>10.4f} | {diff_table[i][0]:>12.6f}"
        for j in range(1, i + 1):
            row_str += f" | {diff_table[i-j][j]:>12.6f}"
        print(row_str)
    print("=" * len(header))


# Hàm in bảng tính tích và trả về các hệ số
def in_bang_tinh_tich(n):
    print("\n" + "="*90)
    print("BẢNG TÍNH TÍCH P_k(t) = t(t-1)...(t-k+1)".center(90))
    print("="*90)
    
    # product_poly_coeffs[k][j] sẽ là hệ số của t^j trong P_k(t)
    product_poly_coeffs = np.zeros((n + 1, n + 1))
    product_poly_coeffs[0, 0] = 1  # P_0(t) = 1

    for k in range(1, n + 1):
        # P_k(t) = P_{k-1}(t) * (t - (k - 1))
        # t*P_{k-1}(t) -> dịch hệ số sang phải 1 bậc
        for j in range(1, k + 1):
            product_poly_coeffs[k, j] = product_poly_coeffs[k-1, j-1]
        # -(k-1)*P_{k-1}(t) -> nhân hệ số cũ với -(k-1)
        for j in range(k):
            product_poly_coeffs[k, j] -= (k - 1) * product_poly_coeffs[k-1, j]
            
    # In header của bảng, theo thứ tự bậc giảm dần
    CELL_WIDTH = 12
    header = f"{'k':<6} |"
    for j in reversed(range(n + 1)):
        header_cell = f"t^{j}"
        header += f" {header_cell:>{CELL_WIDTH - 3}} |"
    print(header)
    print("-" * len(header))
    
    # In các hàng của bảng
    for k in range(n + 1):
        row_str = f" {k:<5} |"
        
        # Thêm khoảng trống ở đầu để tạo thành tam giác vuông bên phải
        padding_cells = n - k
        row_str += " " * (padding_cells * CELL_WIDTH)
        
        # In các hệ số theo thứ tự bậc giảm dần
        for j in reversed(range(k + 1)):
            row_str += f" {product_poly_coeffs[k, j]:>{CELL_WIDTH - 3}.0f} |"
        print(row_str)
    print("=" * len(header))
    return product_poly_coeffs

# Hàm tính đa thức P(t) từ các hệ số Newton
def calculate_poly_t(newton_coeffs):
    from numpy.polynomial import Polynomial
    n = len(newton_coeffs) - 1
    
    # Bắt đầu với đa thức trong cùng: P_n(t) = b_n
    P = Polynomial([newton_coeffs[n]])
    
    # Lặp ngược để xây dựng đa thức từ trong ra ngoài
    for k in range(n - 1, -1, -1):
        # P_k(t) = b_k + (t-k) * P_{k+1}(t)
        # Tạo đa thức (t-k), tương ứng với hệ số [-k, 1]
        term_poly = Polynomial([-k, 1])
        # Tạo đa thức cho hằng số b_k
        b_k_poly = Polynomial([newton_coeffs[k]])
        
        P = b_k_poly + term_poly * P
        
    return P

# Hàm định dạng đa thức theo biến t
def get_polynomial_in_t_str(P_t):
    poly_t_coeffs = P_t.coef

    # Định dạng chuỗi đa thức P(t) để hiển thị
    poly_str = ""
    for i in range(len(poly_t_coeffs) - 1, -1, -1):
        coef = poly_t_coeffs[i]
        if abs(coef) < 1e-9: continue

        sign = " - " if coef < 0 else " + "
        coef = abs(coef)
        
        # Xử lý phần biến của số hạng
        if i == 0:
            term_var = ""
        elif i == 1:
            term_var = "t"
        else:
            term_var = f"t^{i}"
        
        # Xử lý hệ số
        if abs(coef - 1) < 1e-9 and i > 0:
             term_coef = ""
        else:
             term_coef = f"{coef:.4f}"
        
        # Kết hợp thành số hạng
        term_str = term_coef + term_var

        if not poly_str: # Số hạng đầu tiên
            poly_str = f"{"-" if sign.strip() == "-" else ""}{term_str}"
        else:
            poly_str += f"{sign}{term_str}"
            
    return poly_str if poly_str else "0.0"

# Hàm chuyển đổi và định dạng đa thức theo biến x
def get_polynomial_in_x_str(P_t, h, x_start):
    from numpy.polynomial import Polynomial
    
    # Bước 1: Đã có P(t), không cần tính lại
    
    # Bước 2: Chuyển đổi từ P(t) sang P(x)
    # t = (x - x_start) / h = (1/h)x - (x_start/h)
    t_from_x = Polynomial([-x_start / h, 1 / h])
    P_x = P_t(t_from_x)
    poly_x_coeffs = P_x.coef

    # Bước 3: Định dạng chuỗi đa thức P(x) để hiển thị
    poly_str = ""
    for i in range(len(poly_x_coeffs) - 1, -1, -1):
        coef = poly_x_coeffs[i]
        if abs(coef) < 1e-9: continue

        sign = "-" if coef < 0 else ""
        coef = abs(coef)
        
        # Xử lý phần biến của số hạng
        term_var = f"x^{i}: "
        
        # Xử lý hệ số
        if abs(coef - 1) < 1e-9 and i > 0:
             term_coef = "\n"
        else:
             term_coef = f"{coef:.8f}\n"
        
        # Kết hợp thành số hạng
        term_str = term_var + sign + term_coef

        poly_str += f"{term_str}"
            
    return poly_str if poly_str else "0.0"

def main():
    x, y = Input()
    x0 = float(input("Mời nhập giá trị cần tính: "))
    try:
        num = int(input(f"Mời nhập số lượng mốc nội suy tính (<= {len(x)}): "))
        if (num <= 0 or num > len(x)):
            print("Số lượng mốc nội suy không hợp lệ, tự động chọn số lượng mốc lớn nhất.")
            num = len(x)
    except:
        print("Số lượng mốc nội suy không hợp lệ, tự động chọn số lượng mốc lớn nhất.")
        num = len(x)

    h = x[1] - x[0]
    
    x1, y1 = pickPoint(x0, num, x, y) # chọn num điểm phù hợp cho Newton tiến
    print("\nCác mốc nội suy được chọn: ")
    print("x =", np.array(x1))
    print("y =", np.array(y1))
    
    newton_coeffs, diff_table = newton_forward_interpolation(x1, y1)
    
    # In bảng sai phân tiến
    in_bang_sai_phan(x1, diff_table)
    
    # In bảng tính tích
    n = len(x1) - 1
    product_poly_coeffs = in_bang_tinh_tich(n)
        
    print("\n" + "="*80)
    print("KẾT QUẢ NỘI SUY NEWTON TIẾN".center(80))
    print("="*80)
    
    print("Các hệ số Newton b_k = Δ^k(y_0) / k!:")
    print(newton_coeffs)
    print(f"\nCông thức dạng Newton: P(t) = b_0 + b_1*t + ... với t = (x - {x1[0]})/{round(h, 6)}")

    # Tính đa thức P(t) từ hệ số Newton
    P_t = calculate_poly_t(newton_coeffs)

    # Tính và in đa thức theo t
    print(f"\nĐa thức nội suy dạng P(t):")    
    for i, coef in enumerate(P_t.coef):
        print(f"t^{i}: {coef:.8g}")

    # Tính giá trị nội suy bằng cách thay t vào P(t)
    t_val = (x0 - x1[0]) / h
    value = P_t(t_val)
    print(f"\nGiá trị nội suy tại x = {x0} (t = {t_val:.4f}) là: P({x0}) = {value}")
    print("="*80)
    
    # Tính và in đa thức theo x
    poly_x_str = get_polynomial_in_x_str(P_t, h, x1[0])
    print("\n" + "="*90)
    print("ĐA THỨC NỘI SUY DẠNG P(x)".center(90))
    print("="*90)
    print(f" P(x):\n{poly_x_str}")
    print("="*90)

    plot(x1, y1, x, y, x0, value, P_t, h, x1[0])

    # Chức năng thêm mốc bị vô hiệu hóa vì logic đã thay đổi hoàn toàn
    # ...
if __name__=='__main__':
    main()
