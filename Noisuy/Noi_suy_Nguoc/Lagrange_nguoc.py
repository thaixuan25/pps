#
#_nội suy hàm ngược_
#
import numpy as np
import matplotlib.pyplot as plt
import os

# Hàm hoocne nhân đa thức với (x-xk)
def hoocneNhan(A,xk):
    A.append(1)
    for i in range(len(A)-2,0,-1):
        A[i] = A[i - 1] - A[i] * xk
    A[0] = - A[0] * xk
    return A

# Hàm hoocne chia đa thức cho (x-xk)
def hoocneChia(A,xk):
    B = np.ones(len(A) - 1)
    for i in range(len(B) - 2,-1,-1):
        B[i] = A[i + 1] + B[i + 1] * xk
    return B

# Hàm tính giá trị đa thức tại xt
def PolyCoefficients(xt, coeffs):
    """ Trả về giá trị của đa thức tại xt với các hệ số coeffs.
    Các hệ số phải theo thứ tự tăng dần (x**0 đến x**o).
    """
    o = len(coeffs)
    yt = 0
    if isinstance(xt, (list, np.ndarray)):
        yt = np.zeros_like(xt, dtype=float)

    for i in range(o):
        yt += coeffs[i] * xt ** i
    return yt

# Đọc dữ liệu dạng cột từ file input.txt
def read_data(filename):
    x_points = []
    y_points = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if not line.strip():
                    continue  # bỏ qua dòng trắng
                parts = line.strip().split()
                if len(parts) < 2:
                    continue  # bỏ qua dòng không đủ hai giá trị
                x_points.append(float(parts[0]))
                y_points.append(float(parts[1]))
        return x_points, y_points
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {filename}")
        return None, None
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None, None

def find_monotonic_intervals(x_points, y_points):
    """
    Phân tích dữ liệu và chia thành các khoảng đơn điệu (đồng biến hoặc nghịch biến).
    Trả về một danh sách các khoảng, mỗi khoảng là một dictionary chứa 
    các điểm x, y và kiểu đơn điệu của khoảng đó.
    """
    if len(y_points) < 2:
        return []

    intervals = []
    # Tìm các điểm uốn (turning points)
    turning_point_indices = [0]
    for i in range(1, len(y_points) - 1):
        # bỏ qua các điểm nằm ngang
        prev_diff = y_points[i] - y_points[i-1]
        next_diff = y_points[i+1] - y_points[i]
        if prev_diff * next_diff < 0:
            turning_point_indices.append(i)
    turning_point_indices.append(len(y_points) - 1)

    # Tạo các khoảng dựa trên điểm uốn
    for i in range(len(turning_point_indices) - 1):
        start_idx = turning_point_indices[i]
        end_idx = turning_point_indices[i+1]
        
        interval_x = x_points[start_idx : end_idx + 1]
        interval_y = y_points[start_idx : end_idx + 1]

        if len(interval_y) < 2:
            continue
        
        # Xác định kiểu đơn điệu
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

# Main function
def main():
    # Lấy đường dẫn tới thư mục chứa script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Tạo đường dẫn tới file input.txt
    input_file = os.path.join(script_dir, 'input.txt')

    x_points, y_points = read_data(input_file)
    
    if len(x_points) != len(y_points):
        print("Lỗi: số lượng điểm x và y không bằng nhau.")
        return
        
    # In ra các điểm dữ liệu
    print("Các điểm dữ liệu (x, y):")
    for i in range(len(x_points)):
        print(f"({x_points[i]}, {y_points[i]})")
        
    # Phân tích các khoảng đơn điệu
    intervals = find_monotonic_intervals(x_points, y_points)
    if not intervals:
        print("Không thể phân tích dữ liệu thành các khoảng đơn điệu.")
        return

    # Nhập giá trị y cần nội suy
    try:
        y_val = float(input("\nNhập giá trị y cần tìm x: "))

        # Tìm khoảng phù hợp cho y_val
        selected_interval = None
        for interval in intervals:
            min_y, max_y = min(interval['y']), max(interval['y'])
            if min_y <= y_val <= max_y:
                selected_interval = interval
                break
        
        if selected_interval is None:
            print(f"Không tìm thấy khoảng đơn điệu nào chứa giá trị y = {y_val}")
            return

        # Lấy dữ liệu từ khoảng đã chọn để nội suy
        x_points_interval = selected_interval['x']
        y_points_interval = selected_interval['y']
        
        print(f"Giá trị y={y_val} thuộc khoảng {selected_interval['type']} "
              f"với các điểm y trong đoạn [{min(y_points_interval):.4f}, {max(y_points_interval):.4f}].")
        print("Sử dụng các điểm dữ liệu sau để nội suy:")
        for i in range(len(x_points_interval)):
            print(f"({x_points_interval[i]}, {y_points_interval[i]})")

    except Exception as e:
        print(f"Giá trị nhập vào không hợp lệ hoặc đã có lỗi xảy ra: {e}")
        return

    # Nội suy ngược: coi y là biến, x là hàm.
    # Sử dụng dữ liệu từ khoảng đã chọn
    x = np.asarray(y_points_interval)
    y = np.asarray(x_points_interval)
    n = len(x)

    print("\nBảng nội suy (vai trò của x và y đã được hoán đổi):")
    width = 10  # độ rộng cột
    # In header của table
    print('-' * ((n + 1) * width))
    print(f"{'':<{width}}", end="")  # cột trống đầu tiên
    for i in range(n):
        print(f"{f'y[{i}]':^{width}}", end="")
    print()
    print('-' * ((n + 1) * width))
    # In dòng giá trị y
    print(f"{'y':<{width}}", end="")
    for i in range(n):
        print(f"{x[i]:^{width}.4f}", end="")
    print()
    print('-' * ((n + 1) * width))
    # In các dòng giá trị hiệu (y_j - y_i), cột đầu tên y[i]
    for i in range(n):
        print(f"{x[i]:<{width}.4f}", end="")
        for j in range(n):
            if i != j:
                print(f"{(x[j] - x[i]):^{width}.4f}", end="")
            else:
                print(f"{1:^{width}.4f}", end="")
        print()
    print('-' * ((n + 1) * width))

    # tinh D
    D = []
    for i in range(n):
        D.append(1)
        for j in range(n):
            if (i != j):
                D[i] *= (x[i] - x[j])
    D = np.asarray(D)
    for i in range(n):
        print(f"x[{i}]/D[{i}] = {y[i] / D[i]:^{width}.4f}")
    print('-' * ((n + 1) * width))

    # tinh w
    print("\nBẢNG TÍNH TÍCH (đa thức w(y)):")
    w = [1]
    table_w = []
    table_w.append([0] * (n + 1 - len(w)) + w[::-1])
    for i in range(n):
        w = hoocneNhan(w, x[i])
        table_w.append([0] * (n + 1 - len(w)) + w[::-1])
    w = np.asarray(w)

    # In bảng hệ số w(y) dạng table đẹp
    col_width = 12
    total_cols = n + 2  # 1 for "Bước", rest for y^i
    sep = '+' + '+'.join(['-' * col_width for _ in range(total_cols)]) + '+'

    # Header
    deg_labels = [f"y^{deg}" for deg in reversed(range(n + 1))]
    header_cells = ["Bước"] + deg_labels
    header = "|" + "|".join(f"{s:^{col_width}}" for s in header_cells) + "|"

    print(sep)
    print(header)
    print(sep)

    for idx, row_coef in enumerate(table_w):
        row = f"|{idx:^{col_width}}"
        for coef in row_coef:
            row += f"|{coef:^{col_width}.4f}"
        row += "|"
        print(row)
        print(sep)
        
    # tinh wi
    print("\nBảng các hệ số của đa thức wi(y):")
    wi = []
    for i in range(n):
        wi.append(hoocneChia(w, x[i]))
    wi = np.asarray(wi)
    # Header
    header = f" {'':<7} |"
    for i in reversed(range(n)):
        header += f"   y^{i:<5} |"
    print(header)
    print("-" * len(header))

    # Rows
    for i in range(n):
        row_header = f"w_{i}(y)"
        row = f" {row_header:<7} |"
        for j in reversed(range(n)):
            row += f" {wi[i, j]:>9.4f} |"
        print(row)

    # tinh P(y)
    py = np.zeros(n)
    for i in range(n):
        for j in range(n):
            py[i] += wi[j, i] * y[j] / D[j]

    print("\nĐa thức nội suy ngược P(y):")
    poly_str = "x = P(y) = "
    is_first = True
    for i in range(n - 1, -1, -1):
        coef = py[i]
        if abs(coef) < 1e-9:
            continue

        # Dấu của hệ số
        sign = ""
        if is_first:
            if coef < 0:
                sign = "- "
        else:
            if coef > 0:
                sign = " + "
            else:
                sign = " - "
        
        coef = abs(coef)

        # Hệ số và biến
        term = ""
        if abs(coef - 1.0) < 1e-9 and i > 0:
             term = "" # không in hệ số 1.0
        else:
             term = f"{coef:.4f}"

        if i > 1:
            poly_str += f"{sign}{term}y^{i}"
        elif i == 1:
            poly_str += f"{sign}{term}y"
        else: # i == 0
            poly_str += f"{sign}{term}"

        is_first = False
    
    if is_first: # Trường hợp tất cả hệ số bằng 0
        poly_str += "0.0"

    print(poly_str)


    # Tính giá trị x tương ứng
    x_val = PolyCoefficients(y_val, py)
    print(f"\nVới y = {y_val}, giá trị x tương ứng là: {x_val}")

    # Vẽ đồ thị để minh họa
    plt.figure(figsize=(10, 6))
    
    # Tạo một dải các giá trị y để vẽ đường cong nội suy
    y_plot = np.linspace(min(y_points) - 1, max(y_points) + 1, 400)
    # Tính toán x_plot cho toàn bộ đa thức để thấy đường cong tổng thể
    # Để làm điều này, chúng ta cần tính lại đa thức với TOÀN BỘ điểm dữ liệu
    # Tạm thời chỉ vẽ đa thức nội suy trên khoảng đã chọn
    y_plot_interval = np.linspace(min(y_points_interval), max(y_points_interval), 100)
    x_plot_interval = PolyCoefficients(y_plot_interval, py)


    plt.plot(x_plot_interval, y_plot_interval, label=f'Đa thức nội suy trên khoảng được chọn')
    # Vẽ toàn bộ các điểm dữ liệu gốc
    plt.plot(x_points, y_points, 'ro', label='Tất cả các điểm dữ liệu')
    plt.plot(x_val, y_val, 'go', markersize=10, label=f'Điểm nội suy ({x_val:.4f}, {y_val})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Nội suy ngược bằng phương pháp hàm ngược (Lagrange)')
    plt.legend()
    plt.grid(True)
    plt.savefig("Noi_suy_Nguoc/graph.png", dpi=300, bbox_inches='tight')
    print("Đồ thị đã được lưu vào file 'Noi_suy_Nguoc/graph.png'")

if __name__ == "__main__":
    main()
