#!/usr/bin/env python
# coding: utf-8

# ========================================
# PHƯƠNG PHÁP NỘI SUY NEWTON SỬ DỤNG BẢNG TỶ HIỆU
# ========================================
# Chương trình thực hiện nội suy Newton để tìm đa thức xấp xỉ 
# và tính giá trị hàm tại một điểm bất kỳ
# 
# Đặc điểm:
# - Không yêu cầu các mốc nội suy phải cách đều
# - Sử dụng bảng tỷ hiệu chia (divided difference table)
# - Có thể chọn các điểm gần x0 nhất để cải thiện độ chính xác
# ========================================

import numpy as np
import matplotlib.pyplot as plt
import sys
from sympy import *
from sympy import init_printing
init_printing()

# ========================================
# MỤC ĐÍCH CỦA CÁC HÀM TRONG CHƯƠNG TRÌNH
# ========================================
"""
TỔNG QUAN CÁC HÀM VÀ MỤC ĐÍCH SỬ DỤNG:

1. inputData()
   - MỤC ĐÍCH: Đọc và xử lý dữ liệu đầu vào từ file
   - CHỨC NĂNG: Lấy các cặp (x, y) từ file mnb.txt, loại bỏ trùng lặp
   - KẾT QUẢ: Trả về danh sách x, y và bậc đa thức tối đa

2. buildBTH(x, y, n)
   - MỤC ĐÍCH: Xây dựng bảng tỷ hiệu chia Newton
   - CHỨC NĂNG: Tính toán các tỷ hiệu chia từ bậc 0 đến bậc n
   - KẾT QUẢ: Ma trận chứa tất cả tỷ hiệu cần thiết cho nội suy

3. nsNewtonTien(x, y, n)
   - MỤC ĐÍCH: Tạo đa thức nội suy Newton tiến
   - CHỨC NĂNG: Xây dựng đa thức từ điểm đầu tiên x[0]
   - KẾT QUẢ: Biểu thức symbolic của đa thức Newton tiến

4. nsNewtonLui(x, y, n)
   - MỤC ĐÍCH: Tạo đa thức nội suy Newton lùi
   - CHỨC NĂNG: Xây dựng đa thức từ điểm cuối cùng x[n]
   - KẾT QUẢ: Biểu thức symbolic của đa thức Newton lùi

5. pickPoints(x, x0, num)
   - MỤC ĐÍCH: Chọn các điểm tối ưu cho nội suy
   - CHỨC NĂNG: Tìm num điểm gần x0 nhất để cải thiện độ chính xác
   - KẾT QUẢ: Danh sách chỉ số của các điểm được chọn

6. estimate(x, y, x0, deg)
   - MỤC ĐÍCH: Thực hiện nội suy và tính giá trị tại x0
   - CHỨC NĂNG: Tích hợp toàn bộ quy trình nội suy Newton
   - KẾT QUẢ: Đa thức nội suy và giá trị ước tính tại x0

7. main()
   - MỤC ĐÍCH: Điều khiển luồng chương trình chính
   - CHỨC NĂNG: Nhập dữ liệu, thực hiện nội suy, hiển thị kết quả và vẽ đồ thị
   - KẾT QUẢ: Kết quả nội suy hoàn chỉnh với đồ thị minh họa

LƯU Ý QUAN TRỌNG:
- Các hàm 3, 4: Chọn Newton tiến khi x0 gần đầu khoảng, Newton lùi khi x0 gần cuối
- Hàm 5: Việc chọn điểm gần x0 giúp giảm sai số nội suy đáng kể
- Hàm 6: Là hàm trung tâm kết hợp tất cả các bước nội suy
"""

# ========================================
# HÀM NHẬP DỮ LIỆU TỪ FILE
# ========================================
def inputData():
    """
    🎯 MỤC ĐÍCH: Đọc và chuẩn bị dữ liệu cho quá trình nội suy Newton
    
    📁 INPUT: File 'mnb.txt' chứa các cặp giá trị (x, y) cách nhau bởi dấu cách
    
    📊 OUTPUT:
        x: danh sách các giá trị x (hoành độ) - không trùng lặp
        y: danh sách các giá trị y (tung độ) tương ứng
        n: bậc tối đa của đa thức nội suy (= số điểm - 1)
    
    🔍 CHỨC NĂNG:
        - Đọc từng dòng trong file input
        - Tách và chuyển đổi giá trị x, y thành số thực
        - Tự động loại bỏ các điểm có x trùng lặp
        - Trả về dữ liệu sạch sẵn sàng cho nội suy
    
    ⚠️ LƯU Ý: File input phải có format: "x_value y_value" trên mỗi dòng
    """
    x = []  # Danh sách lưu các giá trị x
    y = []  # Danh sách lưu các giá trị y tương ứng
    
    # Mở file và đọc dữ liệu
    with open('Noi_suy_Newton/input.txt','r+') as f:
        for line in f.readlines():
            # Tách giá trị x và y từ mỗi dòng (cách nhau bởi dấu cách)
            xt = float(line.split(' ')[0])  # Giá trị x
            yt = float(line.split(' ')[1])  # Giá trị y
            
            # Kiểm tra xem giá trị x đã tồn tại chưa (tránh trùng lặp)
            check = True
            for x_check in x:
                if x_check == xt:
                    check = False
                    print(f"x[{xt}] da ton tai")
                    break
            
            # Nếu chưa tồn tại thì thêm vào danh sách
            if check:
                x.append(xt)
                y.append(yt)
                
    # Kiểm tra nếu tất cả các điểm x là cách đều thì không cho phép chạy (chỉ dùng cho nội suy bất kỳ)
    is_evenly_spaced = True
    dx0 = x[1] - x[0] if len(x) > 1 else None
    for i in range(2, len(x)):
        if abs((x[i] - x[i-1]) - dx0) > 1e-6:
            is_evenly_spaced = False
            break
    if is_evenly_spaced and len(x) > 2:
        print("Các điểm cách đều, vui lòng nhập lại dữ liệu cho nội suy newton bất kỳ.")
        sys.exit()
    
    return x, y, len(x)-1  # Trả về x, y và bậc của đa thức

def hoocneNhan(A,xk):
    A.append(1)
    for i in range(len(A)-2,0,-1):
        A[i] = A[i - 1] - A[i] * xk
    A[0] = - A[0] * xk
    return A

# ========================================
# HÀM XÂY DỰNG BẢNG TỶ HIỆU CHIA
# ========================================
def buildBTH(x, y, n):
    """
    🎯 MỤC ĐÍCH: Xây dựng bảng tỷ hiệu chia - nền tảng của phương pháp Newton
    
    📊 INPUT:
        x: danh sách các giá trị x (hoành độ) - không cần cách đều
        y: danh sách các giá trị y (tung độ) tương ứng
        n: bậc của đa thức nội suy (thường = len(x) - 1)
    
    📋 OUTPUT:
        BTH: ma trận (n+1)×(n+1) chứa tất cả tỷ hiệu chia
             - Cột 0: f[x_i] (tỷ hiệu bậc 0)
             - Cột j: f[x_i, x_{i+1}, ..., x_{i+j}] (tỷ hiệu bậc j)
    
    🧮 CÔNG THỨC TỶ HIỆU:
        f[x_i, ..., x_{i+k}] = (f[x_{i+1}, ..., x_{i+k}] - f[x_i, ..., x_{i+k-1}]) / (x_{i+k} - x_i)
    
    🔍 CHỨC NĂNG:
        - Khởi tạo cột đầu với các giá trị y
        - Tính tỷ hiệu bậc cao dần từ trái sang phải
        - Tạo ra "kim tự tháp" các tỷ hiệu chia
    
    💡 Ý NGHĨA: Mỗi phần tử BTH[0,j] là hệ số của đa thức Newton
    """
    # Khởi tạo ma trận bảng tỷ hiệu với kích thước (n+1) x (n+1)
    BTH = np.zeros([n+1, n+1])
    
    # Gán cột đầu tiên của bảng = các giá trị y (tỷ hiệu bậc 0)
    for i in range(n+1):
        BTH[i, 0] = y[i]  # f[x_i] = y_i
    
    # Xây dựng các cột tỷ hiệu bậc cao hơn
    for j in range(1, n+1):         # j: bậc của tỷ hiệu (1, 2, 3, ...)
        for i in range(n+1-j):      # i: chỉ số hàng (giảm dần theo bậc)
            # Áp dụng công thức tỷ hiệu chia
            BTH[i, j] = (BTH[i+1, j-1] - BTH[i, j-1]) / (x[i+j] - x[i])
            
    
    
    return BTH


# ========================================
# HÀM NỘI SUY NEWTON TIẾN
# ========================================
def nsNewtonTien(x, n, BTH):
    """
    Xây dựng đa thức nội suy Newton tiến từ Bảng Tỷ Hiệu có sẵn
    
    Args:
        x: danh sách các giá trị x (hoành độ)
        n: bậc của đa thức nội suy
        BTH: Bảng tỷ hiệu đã được tính toán
    
    Returns:
        f: đa thức nội suy Newton dưới dạng biểu thức symbolic
        
    Công thức Newton tiến:
        P(x) = f[x₀] + (x-x₀)f[x₀,x₁] + (x-x₀)(x-x₁)f[x₀,x₁,x₂] + ...
    """
    # Khởi tạo biến symbolic và đa thức ban đầu
    t = Symbol('t')
    f = BTH[0, 0]  # Hệ số tự do = f[x₀]
    
    # Khởi tạo biến tích (x-x₀)
    var = (t - x[0])
    
    # Xây dựng từng số hạng của đa thức Newton
    for i in range(1, n+1):
        # Thêm số hạng: var * f[x₀, x₁, ..., x_i]
        f += var * BTH[0, i]
        # Cập nhật biến tích: var = (x-x₀)(x-x₁)...(x-x_i)
        var = var * (t - x[i])
    
    return f

# ========================================
# HÀM NỘI SUY NEWTON LÙI
# ========================================
def nsNewtonLui(x, n, BTH):
    """
    Xây dựng đa thức nội suy Newton lùi từ Bảng Tỷ Hiệu có sẵn
    
    Args:
        x: danh sách các giá trị x (hoành độ)
        n: bậc của đa thức nội suy
        BTH: Bảng tỷ hiệu đã được tính toán
    
    Returns:
        f: đa thức nội suy Newton dưới dạng biểu thức symbolic
        
    Công thức Newton lùi:
        P(x) = f[x_n] + (x-x_n)f[x_{n-1},x_n] + (x-x_n)(x-x_{n-1})f[x_{n-2},x_{n-1},x_n] + ...
    """
    # Khởi tạo biến symbolic và đa thức ban đầu
    t = Symbol('t')
    f = BTH[n, 0]  # Hệ số tự do = f[x_n]
    
    # Khởi tạo biến tích (x-x_n)
    var = (t - x[n])
    
    # Xây dựng từng số hạng của đa thức Newton (từ cuối về đầu)
    for i in range(1, n+1):
        # Thêm số hạng: var * f[x_{n-i}, ..., x_n]
        f += var * BTH[n-i, i]
        # Cập nhật biến tích: var = (x-x_n)(x-x_{n-1})...(x-x_{n-i})
        var = var * (t - x[n-i])
    
    return f


# ========================================
# HÀM CHỌN ĐIỂM GẦN NHẤT
# ========================================
def pickPoints(x, x0, num):
    """
    Chọn ra num điểm gần x0 nhất từ danh sách các điểm cho trước
    
    Args:
        x: danh sách các giá trị x
        x0: điểm cần tính giá trị nội suy
        num: số lượng điểm muốn chọn
        
    Returns:
        index: danh sách chỉ số của các điểm được chọn (sắp xếp theo độ gần)
        
    Mục đích: Chọn các điểm gần x0 nhất để cải thiện độ chính xác của nội suy
    """
    if num > len(x):
        raise Exception('Số điểm yêu cầu vượt quá số điểm có sẵn! Mời nhập lại')
    else:
        # Tính khoảng cách từ x0 đến tất cả các điểm
        hieu = [abs(x[i] - x0) for i in range(len(x))]
        
        # Sắp xếp các chỉ số theo thứ tự khoảng cách tăng dần
        # enumerate(hieu) tạo ra các cặp (chỉ số, khoảng cách)
        # sorted() sắp xếp theo khoảng cách (key=lambda t:t[1])
        # [i[0] for i in ...] lấy ra chỉ số từ các cặp đã sắp xếp
        index = [i[0] for i in sorted(enumerate(hieu), key=lambda t:t[1])]
        
        # Trả về num điểm gần nhất
        return index[:num]

# ========================================
# HÀM ƯỚC TÍNH GIÁ TRỊ NỘI SUY
# ========================================
def estimate(x, y, x0, deg, choice):
    """
    🎯 MỤC ĐÍCH: HÀM TRUNG TÂM - Thực hiện toàn bộ quá trình nội suy Newton
    
    📊 INPUT:
        x: danh sách tất cả các giá trị x có sẵn
        y: danh sách tất cả các giá trị y tương ứng
        x0: điểm cần tính giá trị (có thể nằm trong hoặc ngoài khoảng dữ liệu)
        deg: bậc của đa thức nội suy mong muốn
        
    🎯 OUTPUT:
        f: đa thức nội suy Newton dưới dạng biểu thức symbolic
        value: giá trị số thực ước tính tại điểm x0
        BTH: Bảng tỷ hiệu chia được sử dụng
        x1: Danh sách các điểm x được chọn để nội suy
    
    🚀 QUY TRÌNH 4 BƯỚC:
        1. CHỌN ĐIỂM THÔNG MINH: Lấy deg+1 điểm gần x0 nhất
        2. XÂY DỰNG BẢNG: Tạo bảng tỷ hiệu chia cho các điểm đã chọn
        3. TẠO ĐA THỨC: Xây dựng đa thức Newton tiến từ bảng tỷ hiệu
        4. TÍNH GIÁ TRỊ: Thay x0 vào đa thức để có kết quả cuối cùng
    
    🎛️ TỐI ƯU HÓA:
        - Chọn điểm gần x0 → giảm sai số nội suy
        - Sử dụng Newton tiến → ổn định tính toán
        - Kết hợp symbolic math → kết quả chính xác
    
    💡 SỬ DỤNG: Đây là hàm chính để gọi khi muốn thực hiện nội suy Newton
    """
    # Bước 1: Chọn deg+1 điểm gần x0 nhất
    index = pickPoints(x, x0, deg+1)
    index.sort() # Sắp xếp chỉ số để bảng tỷ hiệu hiển thị theo thứ tự x tăng dần
    x1 = [x[i] for i in index]  # Danh sách x được chọn
    y1 = [y[i] for i in index]  # Danh sách y tương ứng
    
    # Bước 2: Xây dựng bảng tỷ hiệu chia
    BTH = buildBTH(x1, y1, deg)
    
    # Bước 3: Tạo đa thức nội suy Newton tiến hoặc lùi
    if choice == 1:
        f = nsNewtonTien(x1, deg, BTH)
    elif choice == 2:
        f = nsNewtonLui(x1, deg, BTH)
    else:
        print("Phương pháp nội suy không hợp lệ")
        return None, None, None, None
    
    # Bước 4: Tính giá trị tại x0 bằng cách thay t = x0 vào đa thức
    value = f.subs(Symbol('t'), x0)
    
    return f, value, BTH, x1

# ========================================
# HÀM CHÍNH
# ========================================
def main():
    """
    🎯 MỤC ĐÍCH: ĐIỀU KHIỂN LUỒNG CHƯƠNG TRÌNH - Giao diện người dùng hoàn chỉnh
    
    🎮 CHỨC NĂNG CHÍNH:
        - Giao diện tương tác với người dùng
        - Điều phối tất cả các hàm con
        - Hiển thị kết quả và visualization
        - Xử lý input/output hoàn chỉnh
    
    📋 QUY TRÌNH 5 BƯỚC:
        1. 📁 ĐỌC DỮ LIỆU: Gọi inputData() để load file mnb.txt
        2. ⌨️ NHẬP THÔNG TIN: Cho phép user nhập x0 và bậc đa thức
        3. 🧮 THỰC HIỆN NỘI SUY: Gọi estimate() để tính toán
        4. 📊 HIỂN THỊ KẾT QUẢ: In ra đa thức và giá trị tại x0
        5. 📈 VẼ ĐỒ THỊ: Tạo visualization với matplotlib
    
    🎨 VISUALIZATION:
        - Điểm dữ liệu gốc (đỏ, dạng sao)
        - Đường cong nội suy (xanh dương)
        - Điểm tính toán x0 (xanh lá)
        - Grid, legend và labels đầy đủ
    
    💾 OUTPUT FILE:
        - mygraph.png: Đồ thị chất lượng cao (300 DPI)
        - Console: Đa thức symbolic và giá trị số
    
    🎯 ĐỐI TƯỢNG SỬ DỤNG: Sinh viên, giảng viên, người học nội suy
    """
    # Bước 1: Đọc dữ liệu từ file
    x, y, n = inputData()
    print("Mời chọn phương pháp nội suy: \n1. Newton tiến\n2. Newton lùi(NOT SUPPORTED YET)\n")
    choice = 1
    # Bước 2: Nhập thông tin từ người dùng
    x0 = float(input("Mời nhập giá trị cần tính: "))
    try:
        deg = int(input(f"Mời nhập bậc đa thức (< bậc lớn nhất: {n+1}): "))
        deg -=1
        if (deg <= 0):
            print("Bậc đa thức không hợp lệ tự động chọn bậc lớn nhất")
            sys.exit()
    except:
        print("Bậc đa thức không hợp lệ tự động chọn bậc lớn nhất")
        deg = n
    
    # Bước 3: Thực hiện nội suy Newton
    f, v, BTH, x_chosen = estimate(x, y, x0, deg, choice)
    
    # Bước 4: Hiển thị kết quả
    print("\n========================================")
    print("      BẢNG TỶ HIỆU CHIA (BTH)          ")
    print("========================================")
    
    # In tiêu đề của bảng
    header = "x_i".ljust(10) + "y_i".ljust(15)
    for i in range(1, deg + 1):
        header += f"Bậc {i}".ljust(15)
    print(header)
    print("-" * len(header))

    # In nội dung của bảng
    for i in range(deg + 1):
        row_str = f"{x_chosen[i]:<10.4f}{BTH[i, 0]:<15.4f}"
        for j in range(1, i + 1):
            row_str += f"{BTH[i - j, j]:<15.4f}"
        print(row_str)
    print("========================================\n")
     
    # Bước 4: Hiển thị Bảng Tính Tích (BTT) cho các đa thức cơ sở w_i(x)
    print("\n========================================")
    print("      BẢNG TÍNH TÍCH (PRODUCT TABLE)          ")
    print("========================================")
    
    # w_i(x) được tính dựa trên các điểm đã chọn (x_chosen) và bậc (deg)
    w = [1.0]  # Bắt đầu với w_0(x) = 1, hệ số từ bậc thấp đến cao
    table_w = []
    # Thêm w_0 vào bảng. Coeffs are reversed for printing (high to low deg)
    # and padded to have deg+1 elements
    table_w.append([0.0] * (deg + 1 - len(w)) + w[::-1])

    # Tính các w_i(x) = w_{i-1}(x) * (x - x_{i-1}), CHÚ Ý phải nhân đủ tới phần tử cuối cùng (tức là tới w_deg)
    # Sửa: phải chạy từ i=deg xuống 1 để lấy đủ từ x_chosen[deg],...,x_chosen[1]
    for i in reversed(range(1, deg+1)):
        w = hoocneNhan(w, x_chosen[i])
        # Đệm và thêm vào bảng, đảo ngược hệ số để in
        table_w.append([0.0] * (deg + 1 - len(w)) + w[::-1])
    
    # In bảng hệ số w(x) dạng table đẹp
    col_width = 12
    total_cols = deg + 2  # 1 cột cho x_k, các cột còn lại cho hệ số x^i
    sep = '+' + '+'.join(['-' * col_width for _ in range(total_cols)]) + '+'

    # Header
    deg_labels = [f"x^{d}" for d in reversed(range(deg + 1))]
    header_cells = ["x_k"] + deg_labels
    header = "|" + "|".join(f"{s:^{col_width}}" for s in header_cells) + "|"

    print(sep)
    print(header)
    print(sep)

    # In dòng đầu tiên cho w_0(x) = 1 (không nhân với x_k nào)
    row = f"|{'Bắt đầu':^{col_width}}|"
    for coef in table_w[0]:
        row += f"|{coef:^{col_width}.4f}"
    row += "|"
    print(row)
    print(sep)
    
    # In các dòng tiếp theo, mỗi dòng là kết quả của việc nhân với (x - x_k)
    # In đủ các w_1 đến w_deg
    for idx in range(1, len(table_w)):
        row_coef = table_w[idx]
        # x_k được dùng để tính w_idx là x_chosen[deg-idx+1]
        row = f"|{x_chosen[deg-idx+1]:^{col_width}.4f}|"
        for coef in row_coef:
            row += f"|{coef:^{col_width}.4f}"
        row += "|"
        print(row)
        print(sep)
        
    print("========================================\n")
    
    # In đa thức nội suy theo dạng cột
    from sympy import Poly, Symbol

    print("Đa thức nội suy (theo dạng cột):")
    t = Symbol('t')
    poly = Poly(simplify(f), t)
    coefs = poly.all_coeffs()  # hệ số từ bậc cao xuống thấp
    deg_poly = poly.degree()
    col_width_deg = 8
    col_width_coef = 16

    print(f"{'Bậc':^{col_width_deg}} | {'Hệ số':^{col_width_coef}}")
    print("-" * (col_width_deg + 3 + col_width_coef))
    for i, a in enumerate(coefs):
        print(f"{deg_poly - i:^{col_width_deg}} | {a:^{col_width_coef}}")
    print("-" * (col_width_deg + 3 + col_width_coef))
    print("Giá trị cần tính tại ", x0, " là: ", v)
    
    # Bước 5: Vẽ đồ thị để minh họa
    # Tạo các điểm để vẽ đường cong liên tục
    xx = np.linspace(x[0], x[-1], 100)
    fx = [f.subs(Symbol('t'), xxx) for xxx in xx]  # Tính giá trị đa thức tại các điểm

    # Thiết lập và vẽ đồ thị
    plt.figure()
    plt.scatter(x, y, marker='*', color='red', s=100, label='Điểm dữ liệu')  # Các điểm dữ liệu gốc
    plt.plot(xx, fx, 'b-', linewidth=2, label='Đa thức nội suy Newton')      # Đường cong nội suy
    plt.scatter([x0], [v], marker='o', color='green', s=150, label=f'Điểm tính toán x₀={x0}')  # Điểm được tính
    
    # Thiết lập labels và legend
    plt.xlabel('X (Hoành độ)')
    plt.ylabel('Y (Tung độ)')
    plt.title('Nội suy Newton - Đa thức xấp xỉ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Lưu đồ thị ra file
    plt.savefig("Noi_suy_Newton/mygraph.png", dpi=300, bbox_inches='tight')
    print("Đồ thị đã được lưu vào file 'Noi_suy_Newton/mygraph.png'")

# ========================================
# ĐIỂM KHỞI CHẠY CHƯƠNG TRÌNH
# ========================================
if __name__=='__main__':
    """
    🚀 CÁCH SỬ DỤNG CHƯƠNG TRÌNH:
    
    1. 📄 CHUẨN BỊ DỮ LIỆU:
       - Tạo file 'mnb.txt' trong thư mục 'Noi_suy_Newton/'
       - Format mỗi dòng: "x_value y_value" (cách nhau bởi dấu cách)
       - Ví dụ: 
         1.0 2.5
         2.0 4.1
         3.0 7.8
    
    2. ▶️ CHẠY CHƯƠNG TRÌNH:
       python newton_batky.py
    
    3. ⌨️ NHẬP THÔNG TIN:
       - Nhập giá trị x0 cần tính
       - Nhập bậc đa thức mong muốn (< số điểm - 1)
    
    4. 📊 XEM KẾT QUẢ:
       - Đa thức nội suy được hiển thị trên console
       - Giá trị tại x0 được tính toán
       - Đồ thị được lưu vào 'mygraph.png'
    
    🎓 MỤC ĐÍCH HỌC TẬP:
       - Hiểu thuật toán nội suy Newton
       - Thực hành với bảng tỷ hiệu chia
       - So sánh Newton tiến vs Newton lùi
       - Tối ưu hóa bằng cách chọn điểm gần nhất
    """
    main()