from math import *
import numpy as np
import matplotlib.pyplot as plt
import sys

np.set_printoptions(suppress=True, linewidth=np.inf, precision=9) #chỉnh số chữ số sau dấu phẩy

def Input():  #giả thiết input được sắp xếp cách đều tăng dần
    x, y = [], []
    try:
        with open('Noi_suy_Nguoc/input.txt','r+') as f: # đọc file input
            for line in f.readlines(): # duyệt từng hàng trong file
                if float(line.split()[0]) not in x:
                    x.append(float(line.split()[0]))
                    y.append(float(line.split()[1]))
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'Noi_suy_Nguoc/input.txt'. Vui lòng kiểm tra lại đường dẫn.")
        sys.exit()

    # Sắp xếp các điểm theo giá trị x để đảm bảo trật tự
    sorted_pairs = sorted(zip(x, y))
    x = [p[0] for p in sorted_pairs]
    y = [p[1] for p in sorted_pairs]

    for i in range(1, len(x)): # kiếm tra cách đều
        if abs(x[i] - x[i-1] - (x[1] - x[0])) > 1e-6:
            print("Các điểm x trong file input không cách đều")
            sys.exit()
    return x, y
    
#chọn ra num điểm gần y0 nhất tạo thành một khối liên tục
def pickPointInverse(y0, num, x, y):
    if num > len(x):
        print("Lỗi: Số lượng mốc nội suy yêu cầu lớn hơn số lượng mốc trong dữ liệu.")
        sys.exit()

    # Tìm chỉ số 'i' của mốc y gần y0 nhất
    i = np.abs(np.array(y) - y0).argmin()

    # Xác định chỉ số bắt đầu (start_index) của dãy mốc
    # Cửa sổ 'num' điểm sẽ bao quanh 'i', và dịch chuyển nếu cần để không ra ngoài biên
    start_index = max(0, min(i - num // 2, len(x) - num))
    end_index = start_index + num
    
    x1 = x[start_index:end_index]
    y1 = y[start_index:end_index]

    # Kiểm tra tính đơn điệu của y1 - quan trọng cho nội suy ngược
    is_monotonic = all(y1[j] <= y1[j+1] for j in range(len(y1)-1)) or \
                   all(y1[j] >= y1[j+1] for j in range(len(y1)-1))
    if not is_monotonic:
        print("Cảnh báo: Các giá trị y của các mốc được chọn không đơn điệu.")
        print("Kết quả nội suy ngược có thể không chính xác hoặc không hội tụ.")

    return x1, y1

def newton_forward_interpolation(x1, y1):
    n = len(x1) - 1
    
    # 1. Tính và lưu trữ bảng sai phân tiến
    diff_table = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        diff_table[i][0] = y1[i]
    
    for j in range(1, n + 1):
        for i in range(n + 1 - j):
            diff_table[i][j] = diff_table[i+1][j-1] - diff_table[i][j-1]
            
    # Lấy các sai phân tiến cần thiết từ dòng đầu của bảng
    forward_diffs = diff_table[0]

    # 2. Tính các hệ số c_k = Δ^k(y_0) / k!
    newton_coeffs = np.zeros(n + 1)
    for k in range(n + 1):
        newton_coeffs[k] = forward_diffs[k] / factorial(k)
        
    return newton_coeffs, diff_table

#vẽ hình
def plot(x, y, x_all, y_all, x0, y0, P_t_func, h, x_start):
    plt.clf()
    plt.title("Inverse Newton Forward Interpolation (Iterative)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(x_all, y_all, color='gray', label='All data points')
    plt.scatter(x, y, color='red', s=50, zorder=3, label='Interpolation nodes')

    # Vẽ đường cong nội suy
    x_curve = np.linspace(min(x), max(x), 400)
    t_curve = (x_curve - x_start) / h
    y_curve = [P_t_func(t) for t in t_curve]
    plt.plot(x_curve, y_curve, color='blue', label="Newton Polynomial")
    
    # Đánh dấu điểm nội suy ngược
    plt.scatter(x0, y0, color='green', zorder=5, s=150, marker='*', label=f'Inverse-Interpolated Point ({x0:.4f}, {y0:.2f})')
    # Vẽ đường gióng
    plt.plot([x0, x0], [min(y_all), y0], 'g--')
    plt.plot([min(x_all), x0], [y0, y0], 'g--')

    plt.legend()
    plt.grid(True)
    plt.savefig("newton_nguoc_graph.png")
    # plt.show()

# Hàm in bảng sai phân đều
def in_bang_sai_phan(x1, diff_table):
    n = len(x1)
    print("\n" + "="*95)
    print("BẢNG SAI PHÂN TIẾN (FORWARD DIFFERENCE TABLE)".center(95))
    print("="*95)
    
    header = f"{'x[i]':>10} | {'y[i]':>12} |"
    for j in range(1, n):
        col_header = f"Δ^{j}y"
        header += f" {col_header:<12} |"
    print(header)
    print("-" * len(header))
    
    # In các hàng của bảng theo định dạng tam giác vuông dưới
    for i in range(n):
        row_str = f"{x1[i]:>10.4f} | {diff_table[i][0]:>12.6f} |"
        # In các giá trị sai phân tương ứng
        for j in range(1, i + 1):
            # Giá trị cần in là Δ^j y_{i-j} tương ứng với diff_table[i-j][j]
            if (i-j) >= 0:
                row_str += f" {diff_table[i-j][j]:>12.6f} |"
        print(row_str)
    print("=" * len(header))

def do_iteration(y0_target, newton_coeffs, x1):
    n = len(x1) - 1
    c = newton_coeffs # c_k = Δ^k(y_0) / k!
    
    if abs(c[1] * factorial(1)) < 1e-9: # Δy_0 = c_1 * 1!
        print("\nLỗi: Sai phân bậc nhất (delta y) quá nhỏ, phương pháp lặp không ổn định.")
        return None

    # Ước tính ban đầu cho t từ phép xấp xỉ tuyến tính: y ≈ y_0 + t*Δy_0
    t = (y0_target - c[0]) / (c[1] * factorial(1))

    max_iter = 100
    tolerance = 1e-10
    
    print("\n" + "="*80)
    print("BẮT ĐẦU QUÁ TRÌNH LẶP TÌM t".center(80))
    print("="*80)
    print(f"Công thức lặp: t_new = (y_target - c0 - P_hoa(t_old)) / c1")
    print(f"Ước tính ban đầu: t_0 = ({y0_target:.4f} - {c[0]:.4f}) / {c[1]:.4f} = {t:.6f}")
    
    for k_iter in range(max_iter):
        # Tính tổng các số hạng bậc cao P_hoa(t) = c2*t(t-1) + c3*t(t-1)(t-2) + ...
        sum_higher_order = 0
        for j in range(2, n + 1):
            prod_term = 1
            for i in range(j):
                prod_term *= (t - i)
            sum_higher_order += c[j] * prod_term
        
        # t_{k+1} = (y_target - c_0 - P_hoa(t_k)) / c_1
        # Chú ý: c_1 trong công thức là Δy_0. newton_coeffs[1] = Δy_0/1!
        delta_y0 = c[1] * factorial(1)
        t_new = (y0_target - c[0] - sum_higher_order) / delta_y0
        
        print(f"Bước {k_iter+1:2d}: t = {t_new:.8f}, sai số = {abs(t_new - t):.2e}")

        if abs(t_new - t) < tolerance:
            print(f"\nLặp hội tụ sau {k_iter + 1} bước.")
            print("="*80)
            return t_new
            
        t = t_new

    print("\nCảnh báo: Vòng lặp không hội tụ sau số lần lặp tối đa.")
    print("Kết quả có thể không chính xác.")
    print("="*80)
    return t

# Hàm tính giá trị đa thức P(t) tại một điểm t cụ thể
def evaluate_Pt(t, newton_coeffs):
    n = len(newton_coeffs) - 1
    # Tính theo kiểu Horner: P(t) = c_0 + t(c_1 + (t-1)(c_2 + ...))
    # Bắt đầu từ trong: val = c_n
    val = newton_coeffs[n]
    # Lặp ngược
    for k in range(n - 1, -1, -1):
        val = newton_coeffs[k] + (t - k) * val
    return val

def main():
    x, y = Input()
    y0 = float(input("Mời nhập giá trị y cần tìm x: "))
    try:
        num = int(input(f"Mời nhập số lượng mốc nội suy (<= {len(x)}): "))
        if not (1 < num <= len(x)):
            print(f"Số lượng mốc không hợp lệ. Tự động chọn tất cả {len(x)} mốc.")
            num = len(x)
    except ValueError:
        print(f"Đầu vào không hợp lệ. Tự động chọn tất cả {len(x)} mốc.")
        num = len(x)

    h = x[1] - x[0]
    
    # Chọn num điểm phù hợp cho nội suy ngược
    x1, y1 = pickPointInverse(y0, num, x, y) 
    print("\nCác mốc nội suy được chọn: ")
    print("x =", np.array(x1))
    print("y =", np.array(y1))
    
    # Tính bảng sai phân và hệ số Newton
    newton_coeffs, diff_table = newton_forward_interpolation(x1, y1)
    
    in_bang_sai_phan(x1, diff_table)
    
    print("\nCác hệ số Newton c_k = Δ^k(y_0) / k!:")
    print(newton_coeffs)
    
    # Thực hiện lặp để tìm t
    t_val = do_iteration(y0, newton_coeffs, x1)

    if t_val is not None:
        # Từ t tính ra x
        x0_found = x1[0] + t_val * h
        print(f"\n" + "="*80)
        print("KẾT QUẢ NỘI SUY NGƯỢC".center(80))
        print("="*80)
        print(f"Giá trị t tìm được là: t = {t_val:.8f}")
        print(f"Giá trị x tương ứng với y = {y0} là: x = {x0_found:.8f}")
        print("="*80)

        # Tạo hàm P(t) để vẽ đồ thị
        P_t_func = lambda t: evaluate_Pt(t, newton_coeffs)
        
        # Vẽ đồ thị
        plot(x1, y1, x, y, x0_found, y0, P_t_func, h, x1[0])
        print("\nĐã lưu đồ thị vào file 'graph.png'")

if __name__=='__main__':
    main()
