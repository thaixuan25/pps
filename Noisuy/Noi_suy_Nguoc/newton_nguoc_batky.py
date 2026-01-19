#!/usr/bin/env python
# coding: utf-8

# ========================================
# PHƯƠNG PHÁP NỘI SUY NGƯỢC SỬ DỤNG NEWTON
# ========================================
# Chương trình thực hiện nội suy ngược Newton để tìm giá trị x
# tương ứng với một giá trị y cho trước.
# 
# Đặc điểm:
# - Coi y là biến độc lập và x là biến phụ thuộc: x = g(y)
# - Không yêu cầu các mốc nội suy (y) phải cách đều
# - Sử dụng bảng tỷ hiệu chia (divided difference table)
# - Có thể chọn các điểm gần y0 nhất để cải thiện độ chính xác
# ========================================

import numpy as np
import matplotlib.pyplot as plt
import sys
from sympy import *
from sympy import init_printing
init_printing()

# ========================================
# HÀM NHẬP DỮ LIỆU TỪ FILE
# ========================================
def inputData():
    """
    🎯 MỤC ĐÍCH: Đọc và chuẩn bị dữ liệu cho quá trình nội suy ngược Newton.
    
    📁 INPUT: File 'input.txt' chứa các cặp giá trị (x, y) cách nhau bởi dấu cách.
    
    📊 OUTPUT:
        y: danh sách các giá trị y (coi là biến độc lập) - không trùng lặp
        x: danh sách các giá trị x (coi là biến phụ thuộc) tương ứng
        n: bậc tối đa của đa thức nội suy (= số điểm - 1)
    
    🔍 CHỨC NĂNG:
        - Đọc từng dòng trong file input.
        - Tách và chuyển đổi giá trị x, y thành số thực.
        - Tự động loại bỏ các điểm có y trùng lặp (vì y là biến độc lập).
        - Trả về dữ liệu sạch sẵn sàng cho nội suy.
    
    ⚠️ LƯU Ý: 
        - File input phải có format: "x_value y_value" trên mỗi dòng.
        - Hàm x(y) phải là hàm đơn ánh trong khoảng nội suy để kết quả có ý nghĩa.
    """
    x = []  # Danh sách lưu các giá trị x
    y = []  # Danh sách lưu các giá trị y
    
    # Mở file và đọc dữ liệu
    with open('Noi_suy_Nguoc/input.txt','r+') as f:
        for line in f.readlines():
            # Tách giá trị x và y từ mỗi dòng (cách nhau bởi dấu cách)
            xt = float(line.strip().split()[0])  # Giá trị x
            yt = float(line.strip().split()[1])  # Giá trị y
            
            # Kiểm tra xem giá trị y đã tồn tại chưa (tránh trùng lặp)
            check = True
            for y_check in y:
                if y_check == yt:
                    check = False
                    print(f"y[{yt}] da ton tai")
                    break
            
            # Nếu y chưa tồn tại thì thêm cặp (x, y) vào danh sách
            if check:
                x.append(xt)
                y.append(yt)
                
    return y, x, len(y)-1  # Trả về y, x và bậc của đa thức (coi y là biến chính)

def find_monotonic_intervals(x_points, y_points):
    """
    Phân tích dữ liệu và chia thành các khoảng đơn điệu (đồng biến hoặc nghịch biến).
    """
    # Sắp xếp các điểm theo biến độc lập (x_points) để xử lý cho đúng
    sorted_points = sorted(zip(x_points, y_points))
    x_sorted, y_sorted = zip(*sorted_points)
    x_sorted, y_sorted = list(x_sorted), list(y_sorted)

    if len(y_sorted) < 2:
        return []

    intervals = []
    # Tìm các điểm uốn (turning points) dựa trên y
    turning_point_indices = [0]
    for i in range(1, len(y_sorted) - 1):
        prev_diff = y_sorted[i] - y_sorted[i-1]
        next_diff = y_sorted[i+1] - y_sorted[i]
        # Bỏ qua các điểm nằm ngang, chỉ xét điểm đổi chiều thực sự
        if prev_diff * next_diff < 0:
            turning_point_indices.append(i)
    turning_point_indices.append(len(y_sorted) - 1)

    # Tạo các khoảng dựa trên điểm uốn
    for i in range(len(turning_point_indices) - 1):
        start_idx = turning_point_indices[i]
        end_idx = turning_point_indices[i+1]
        
        interval_x = x_sorted[start_idx : end_idx + 1]
        interval_y = y_sorted[start_idx : end_idx + 1]

        if len(interval_y) < 2:
            continue
        
        is_increasing = all(interval_y[j+1] >= interval_y[j] for j in range(len(interval_y)-1))
        is_decreasing = all(interval_y[j+1] <= interval_y[j] for j in range(len(interval_y)-1))

        interval_type = 'không đổi'
        if is_increasing and not is_decreasing:
             interval_type = 'đồng biến'
        elif is_decreasing and not is_increasing:
            interval_type = 'nghịch biến'
        
        intervals.append({
            'x': interval_x,
            'y': interval_y,
            'type': interval_type
        })
        
    return intervals

def hoocneNhan(A, xk):
    # A is coeffs [a_0, a_1, ..., a_n] (low to high)
    # Returns coeffs of P(y) * (y - xk) in the same order
    # (a_0 + a_1*y + ... + a_n*y^n) * (y - xk)
    # = -a_0*xk + (a_0 - a_1*xk)y + ... + a_n*y^{n+1}
    n = len(A) - 1
    B = [0] * (n + 2) # Bậc mới là n+1, nên có n+2 hệ số
    B[0] = -A[0] * xk
    for i in range(1, n + 1): # i đi từ 1 đến n
        B[i] = A[i-1] - A[i] * xk
    B[n+1] = A[n]
    return B

# ========================================
# HÀM XÂY DỰNG BẢNG TỶ HIỆU CHIA
# ========================================
def buildBTH(x, y, n):
    """
    🎯 MỤC ĐÍCH: Xây dựng bảng tỷ hiệu chia - nền tảng của phương pháp Newton.
    Trong nội suy ngược, 'x' là các giá trị y, 'y' là các giá trị x.
    
    📊 INPUT:
        x: danh sách các giá trị biến độc lập (y của hàm gốc)
        y: danh sách các giá trị biến phụ thuộc (x của hàm gốc)
        n: bậc của đa thức nội suy
    
    📋 OUTPUT:
        BTH: ma trận (n+1)×(n+1) chứa tất cả tỷ hiệu chia.
    
    🧮 CÔNG THỨC TỶ HIỆU:
        f[x_i, ..., x_{i+k}] = (f[x_{i+1}, ..., x_{i+k}] - f[x_i, ..., x_{i+k-1}]) / (x_{i+k} - x_i)
    """
    # Khởi tạo ma trận bảng tỷ hiệu với kích thước (n+1) x (n+1)
    BTH = np.zeros([n+1, n+1])
    
    # Gán cột đầu tiên của bảng = các giá trị y (tức là x của hàm ngược)
    for i in range(n+1):
        BTH[i, 0] = y[i]
    
    # Xây dựng các cột tỷ hiệu bậc cao hơn
    for j in range(1, n+1):         # j: bậc của tỷ hiệu (1, 2, 3, ...)
        for i in range(n+1-j):      # i: chỉ số hàng (giảm dần theo bậc)
            # Áp dụng công thức tỷ hiệu chia
            BTH[i, j] = (BTH[i+1, j-1] - BTH[i, j-1]) / (x[i+j] - x[i])
            
    return BTH

# ========================================
# HÀM NỘI SUY NEWTON TIẾN (CHO HÀM NGƯỢC)
# ========================================
def nsNewtonTien(y_points, n, BTH):
    """
    Xây dựng đa thức nội suy Newton tiến x(y) từ Bảng Tỷ Hiệu.
    
    Args:
        y_points: danh sách các giá trị y (biến độc lập)
        n: bậc của đa thức nội suy
        BTH: Bảng tỷ hiệu đã được tính toán
    
    Returns:
        f: đa thức nội suy ngược x(y) dưới dạng biểu thức symbolic
        
    Công thức Newton tiến cho hàm ngược:
        x(y) = a₀ + a₁*(y-y₀) + a₂*(y-y₀)(y-y₁) + ...
    """
    # Khởi tạo biến symbolic và đa thức ban đầu
    t = Symbol('t') # Sử dụng 't' để đại diện cho 'y'
    f = BTH[0, 0]  # Hệ số tự do = x(y₀)
    
    # Khởi tạo biến tích (y-y₀)
    var = (t - y_points[0])
    
    # Xây dựng từng số hạng của đa thức Newton
    for i in range(1, n+1):
        # Thêm số hạng: var * a_i
        f += var * BTH[0, i]
        # Cập nhật biến tích: var = (y-y₀)(y-y₁)...(y-y_i)
        var = var * (t - y_points[i])
    
    return f

# ========================================
# HÀM CHỌN ĐIỂM GẦN NHẤT
# ========================================
def pickPoints(x, x0, num):
    """
    Chọn ra num điểm gần x0 nhất từ danh sách các điểm cho trước.
    
    Args:
        x: danh sách các giá trị (trong TH này là các giá trị y)
        x0: điểm cần tính giá trị nội suy (trong TH này là y0)
        num: số lượng điểm muốn chọn
        
    Returns:
        index: danh sách chỉ số của các điểm được chọn.
    """
    if num > len(x):
        raise Exception('Số điểm yêu cầu vượt quá số điểm có sẵn! Mời nhập lại')
    else:
        # Tính khoảng cách từ x0 đến tất cả các điểm
        hieu = [abs(x[i] - x0) for i in range(len(x))]
        
        # Sắp xếp các chỉ số theo thứ tự khoảng cách tăng dần
        index = [i[0] for i in sorted(enumerate(hieu), key=lambda t:t[1])]
        
        # Trả về num điểm gần nhất
        return index[:num]

# ========================================
# HÀM ƯỚC TÍNH GIÁ TRỊ NỘI SUY NGƯỢC
# ========================================
def estimate(y_all, x_all, y0, deg):
    """
    🎯 MỤC ĐÍCH: HÀM TRUNG TÂM - Thực hiện toàn bộ quá trình nội suy ngược Newton.
    
    📊 INPUT:
        y_all: danh sách tất cả các giá trị y có sẵn
        x_all: danh sách tất cả các giá trị x tương ứng
        y0: giá trị y cần tìm x
        deg: bậc của đa thức nội suy mong muốn
        
    🎯 OUTPUT:
        f: đa thức nội suy ngược x(y) dưới dạng biểu thức symbolic
        value: giá trị x ước tính tại điểm y0
        BTH: Bảng tỷ hiệu chia được sử dụng
        y_chosen: Danh sách các điểm y được chọn để nội suy
        x_chosen: Danh sách các điểm x tương ứng
    
    🚀 QUY TRÌNH 4 BƯỚC:
        1. CHỌN ĐIỂM THÔNG MINH: Lấy deg+1 điểm y gần y0 nhất.
        2. XÂY DỰNG BẢNG: Tạo bảng tỷ hiệu cho các điểm đã chọn.
        3. TẠO ĐA THỨC: Xây dựng đa thức Newton tiến x(y).
        4. TÍNH GIÁ TRỊ: Thay y0 vào đa thức để có giá trị x cuối cùng.
    """
    # Bước 1: Chọn deg+1 điểm y gần y0 nhất
    index = pickPoints(y_all, y0, deg+1)
    index.sort() # Sắp xếp chỉ số để bảng tỷ hiệu hiển thị theo thứ tự y tăng dần
    y_chosen = [y_all[i] for i in index]  # Danh sách y được chọn
    x_chosen = [x_all[i] for i in index]  # Danh sách x tương ứng
    
    # Bước 2: Xây dựng bảng tỷ hiệu chia cho x(y)
    BTH = buildBTH(y_chosen, x_chosen, deg)
    
    # Bước 3: Tạo đa thức nội suy Newton tiến x(y)
    f = nsNewtonTien(y_chosen, deg, BTH)
    
    # Bước 4: Tính giá trị x tại y0 bằng cách thay t = y0 vào đa thức
    value = f.subs(Symbol('t'), y0)
    
    return f, value, BTH, y_chosen, x_chosen

# ========================================
# HÀM CHÍNH
# ========================================
def main():
    """
    🎯 MỤC ĐÍCH: ĐIỀU KHIỂN LUỒNG CHƯƠNG TRÌNH - Giao diện người dùng hoàn chỉnh.
    """
    # Bước 1: Đọc dữ liệu từ file
    y, x, n = inputData()
    
    # Sắp xếp các điểm theo y để vẽ đồ thị cho đẹp
    sorted_points = sorted(zip(y, x))
    y_sorted, x_sorted = zip(*sorted_points)

    # Bước 2: Nhập thông tin từ người dùng
    y0 = float(input("Mời nhập giá trị y cần tìm x: "))
    
    # Phân tích các khoảng đơn điệu
    # Trong nội suy ngược, y là biến độc lập, x là biến phụ thuộc
    # Nhưng để tìm khoảng đơn điệu, ta xét sự thay đổi của y theo x
    intervals = find_monotonic_intervals(x_sorted, y_sorted)
    if not intervals:
        print("Không thể phân tích dữ liệu thành các khoảng đơn điệu.")
        return
        
    # Tìm khoảng phù hợp cho y0
    selected_interval = None
    for interval in intervals:
        min_y, max_y = min(interval['y']), max(interval['y'])
        if min_y <= y0 <= max_y:
            selected_interval = interval
            break
            
    if selected_interval is None:
        print(f"Không tìm thấy khoảng đơn điệu nào chứa giá trị y = {y0}")
        return

    # Lấy dữ liệu từ khoảng đã chọn để nội suy
    x_interval = selected_interval['x']
    y_interval = selected_interval['y']
    
    print(f"\nGiá trị y={y0} thuộc khoảng {selected_interval['type']} "
          f"với các điểm y trong đoạn [{min(y_interval):.4f}, {max(y_interval):.4f}].")
    print("Sử dụng các điểm dữ liệu sau để nội suy:")
    for i in range(len(x_interval)):
        print(f"({x_interval[i]}, {y_interval[i]})")

    n_interval = len(y_interval) - 1
    
    try:
        deg = int(input(f"\nMời nhập bậc đa thức (<= {n_interval}): "))
        if (deg <= 0 or deg > n_interval):
            print(f"Bậc đa thức không hợp lệ. Tự động chọn bậc lớn nhất là {n_interval}.")
            deg = n_interval
    except:
        print(f"Bậc đa thức không hợp lệ. Tự động chọn bậc lớn nhất là {n_interval}.")
        deg = n_interval
    
    # Bước 3: Thực hiện nội suy ngược Newton trên khoảng đã chọn
    # Lưu ý: đầu vào của estimate là (y, x, ...) vì ta nội suy x = g(y)
    f, v, BTH, y_chosen, x_chosen = estimate(y_interval, x_interval, y0, deg)
    
    # Bước 4: Hiển thị kết quả
    print("\n========================================")
    print("      BẢNG TỶ HIỆU CHIA (CHO HÀM NGƯỢC x(y))")
    print("========================================")
    
    # In tiêu đề của bảng
    header = "y_i".ljust(10) + "x_i".ljust(15)
    for i in range(1, deg + 1):
        header += f"Bậc {i}".ljust(15)
    print(header)
    print("-" * len(header))

    # In nội dung của bảng theo dạng tam giác vuông
    for i in range(deg + 1):
        row_str = f"{y_chosen[i]:<10.4f}{BTH[i, 0]:<15.4f}"
        # Với mỗi hàng i, ta in các tỷ hiệu trên đường chéo đi lên
        # BTH[i,0], BTH[i-1,1], BTH[i-2,2], ..., BTH[0,i]
        for j in range(1, i + 1):
            row_str += f"{BTH[i-j, j]:<15.4f}"
        print(row_str)
        
    print("========================================\n")

# In ra các hệ số tỷ sai phân đã chọn
    print("\n========================================")
    print("Các hệ số tỷ sai phân đã chọn (cho đa thức Newton ngược):")
    selected_divided_differences = [BTH[0, j] for j in range(deg + 1)]
    for i, coeff in enumerate(selected_divided_differences):
        print(f"Hệ số bậc {i}: {coeff:.8f}")
    print("========================================\n")

    # Hiển thị Bảng Tính Tích (BTT) cho các đa thức cơ sở w_i(y)
    print("\n========================================")
    print("      BẢNG TÍNH TÍCH (CHO HÀM NGƯỢC)")
    print("========================================")
    
    # w_i(y) = (y - y_0)...(y - y_{i-1})
    # w được lưu dưới dạng hệ số [a_0, a_1, ...] (bậc thấp đến cao)
    w = [1.0]  # w_0(y) = 1
    table_w = []
    # Thêm w_0 vào bảng. Đệm bằng 0 để đủ độ dài deg+1
    table_w.append(w + [0.0] * deg)

    # Tính các w_i(y) = w_{i-1}(y) * (y - y_{i-1})
    for i in range(1, deg + 1):
        w = hoocneNhan(w, y_chosen[i-1])
        # Đệm và thêm vào bảng
        table_w.append(w + [0.0] * (deg - i))
    
    # In bảng hệ số w(y) dạng table đẹp
    col_width = 12
    total_cols = deg + 2
    sep = '+' + '+'.join(['-' * col_width for _ in range(total_cols)]) + '+'

    # Header (bậc cao đến thấp)
    deg_labels = [f"y^{d}" for d in reversed(range(deg + 1))]
    header_cells = ["y_k"] + deg_labels
    header = "|" + "|".join(f"{s:^{col_width}}" for s in header_cells) + "|"

    
    print(sep)
    print(header)
    print(sep)
    
    
    # In dòng đầu tiên cho w_0(y) = 1
    row = f"|{'Bắt đầu':^{col_width}}|"
    for coef in reversed(table_w[0]): # Đảo ngược hệ số để in (cao -> thấp)
        row += f"|{coef:^{col_width}.4f}"
    row += "|"
    print(row)
    print(sep)
    
    # In các dòng tiếp theo
    for idx in range(1, len(table_w)):
        row_coef = table_w[idx]
        row = f"|{y_chosen[idx-1]:^{col_width}.4f}|"
        for coef in reversed(row_coef): # Đảo ngược hệ số để in (cao -> thấp)
            row += f"|{coef:^{col_width}.4f}"
        row += "|"
        print(row)
        print(sep)
        
    print("========================================\n")
     
    # In đa thức nội suy
    print("Đa thức nội suy ngược x(y) tìm được:")
    t = Symbol('y')
    pretty_f = printing.pretty(expand(f.subs(Symbol('t'), t)))
    print(pretty_f)
    
    print("\nGiá trị x cần tính tại y =", y0, " là: ", v)
    
    # Bước 5: Vẽ đồ thị để minh họa
    # Tạo các điểm y trên khoảng đã chọn để vẽ đường cong liên tục
    yy = np.linspace(min(y_chosen), max(y_chosen), 400)
    fx = [f.subs(Symbol('t'), yyy) for yyy in yy]  # Tính giá trị x tương ứng

    # Thiết lập và vẽ đồ thị
    plt.figure(figsize=(10, 6))
    plt.plot(fx, yy, 'b-', linewidth=2, label=f'Đa thức nội suy trên khoảng được chọn') # Vẽ x theo y
    plt.scatter(x, y, marker='*', color='red', s=100, label='Tất cả điểm dữ liệu')  # Các điểm dữ liệu gốc
    plt.scatter([v], [y0], marker='o', color='green', s=150, label=f'Điểm nội suy x({y0}) ≈ {v:.4f}')  # Điểm được tính
    
    # Thiết lập labels và legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Nội suy ngược Newton')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Lưu đồ thị ra file
    plt.savefig("Noi_suy_Nguoc/graph.png", dpi=300, bbox_inches='tight')
    print("\nĐồ thị đã được lưu vào file 'Noi_suy_Nguoc/graph.png'")

# ========================================
# ĐIỂM KHỞI CHẠY CHƯƠNG TRÌNH
# ========================================
if __name__=='__main__':
    main()
