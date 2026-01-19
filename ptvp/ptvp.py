import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# ==========================================
# 0. THÔNG TIN BÀI THI (IN CÔNG THỨC QUAN TRỌNG)
# ==========================================
def print_exam_info(problem_id, t_span, y0, h, labels):
    print("\n" + "=" * 70)
    print("THÔNG TIN CẦN THIẾT (CÔNG THỨC + CÁC BƯỚC QUAN TRỌNG)")
    print("=" * 70)

    # Thông tin bài toán
    print(f"- Bài toán: y' = F(t, y),  t ∈ [{t_span[0]}, {t_span[1]}],  y(t0)=y0")
    print(f"- t0 = {t_span[0]},  y0 = {y0} (theo thứ tự {labels}),  bước h = {h}")
    print(f"- Lưới đều: t_n = t0 + n·h")

    # In hệ phương trình cụ thể (nếu có)
    if problem_id == "f":
        print("\n- Hệ phương trình (bài f):")
        print("  x' = 0.4·x·(1 - x/20) + 0.4·y - 0.3·x·z")
        print("  y' = 0.7·y·(1 - y/25) - 0.4·y - 0.4·y·z")
        print("  z' = -0.3·z + 0.35·(x + y)·z")

    # Công thức các phương pháp
    print("\n- Công thức phương pháp:")
    print("  (1) Euler hiện (Explicit Euler):")
    print("      y_{n+1} = y_n + h·F(t_n, y_n)")
    print("  (2) Euler ẩn (Implicit Euler):")
    print("      y_{n+1} = y_n + h·F(t_{n+1}, y_{n+1})")
    print("      ⇔ Giải phương trình: G(z) = z - y_n - h·F(t_{n+1}, z) = 0")
    print("  (3) Hình thang (Trapezoidal / Crank–Nicolson):")
    print("      y_{n+1} = y_n + (h/2)·(F(t_n, y_n) + F(t_{n+1}, y_{n+1}))")
    print("      ⇔ Giải phương trình: H(z) = z - y_n - (h/2)·(F(t_n, y_n) + F(t_{n+1}, z)) = 0")

    print("\n- Ghi chú khi giải ẩn:")
    print("  Dùng nghiệm dự đoán (guess) từ Euler hiện để tăng tốc hội tụ fsolve.")
    print("=" * 70 + "\n")

# ==========================================
# 1. ĐỊNH NGHĨA CÁC BÀI TOÁN (INPUT DATA)
# ==========================================
def get_problem_config(problem_id):
    """
    Trả về hàm F(t, y), khoảng t, giá trị đầu y0, và bước h
    Lưu ý: y luôn được xử lý dưới dạng mảng (vector)
    """
    if problem_id == 'f':
        # Hệ 3 phương trình ví dụ:
        #   x' = 0.4*x*(1 - x/20) + 0.4*y - 0.3*x*z
        #   y' = 0.7*y*(1 - y/25) - 0.4*y - 0.4*y*z
        #   z' = -0.3*z + 0.35*(x + y)*z
        #
        # HƯỚNG DẪN THAY BÀI KHÁC:
        # - B1: Sửa lại 3 công thức dx, dy, dz theo đề bài mới.
        #       Nhớ dùng đúng thứ tự biến (x, y, z) hoặc đổi tên cho dễ hiểu.
        # - B2: Nếu số ẩn ≠ 3 (ví dụ chỉ có 1 ẩn y hoặc 2 ẩn x,y), bạn:
        #       + Giảm/tăng số biến trong Y (ví dụ: y = Y[0], z = Y[1], ...).
        #       + Trả về np.array([...]) với đúng số phương trình.
        # - B3: Sửa `t_span`, `y0`, `h`, và `labels` cho khớp đề:
        #       + t_span = (t0, T): khoảng thời gian cần giải.
        #       + y0 = np.array([...]): giá trị đầu cho từng biến.
        #       + h: bước thời gian (càng nhỏ càng chính xác, nhưng tốn thời gian).
        #       + labels: tên các ẩn để in bảng/đồ thị cho dễ nhìn.
        #   B4: Nếu có hàm là đạo hàm bậc cao thì nên đặt giá trị trung gian như y'' thì đặt y' = u tù đỏ y'' thành u' = .... (Và tìm giá trị ban đầu của u)
        def func(t, Y):
            x, y, z = Y
            dx = 0.4*x*(1 - x/20) + 0.4*y - 0.3*x*z
            dy = 0.7*y*(1 - y/25) - 0.4*y - 0.4*y*z
            dz = -0.3*z + 0.35*(x + y)*z
            return np.array([dx, dy, dz])
        t_span, y0, h = (0, 1500), np.array([12.0, 18.0, 8.0]), 0.1
        labels = ['x', 'y', 'z']
    
    else:
        return None
    
    return func, t_span, y0, h, labels

# ==========================================
# 2. BỘ GIẢI TỔNG QUÁT (SOLVERS)
# ==========================================

def _build_grid(t_span, h):
    # +h/1000 để lấy cả điểm cuối (tránh lỗi do sai số float)
    t_values = np.arange(t_span[0], t_span[1] + h / 1000, h)
    return t_values


def _should_explain_step(i, n_steps, explain_steps, explain_first_n, explain_last_n):
    if not explain_steps:
        return False
    if i < explain_first_n:
        return True
    if explain_last_n > 0 and i >= (n_steps - 1 - explain_last_n):
        return True
    return False


# ==========================================
# 2b. HÀM KHUẾCH ĐẠI & MIỀN ỔN ĐỊNH
#    (cho bài test: y' = λ y,  z = h·λ)
# ==========================================

def amplification_factor(method, z):
    """
    Trả về R(z) cho bài toán test y' = λy với z = h·λ.
    - Euler hiện:     R(z) = 1 + z
    - Euler ẩn:       R(z) = 1 / (1 - z)
    - Hình thang:     R(z) = (1 + z/2) / (1 - z/2)
    """
    if method == "euler_hien":
        return 1 + z
    elif method == "euler_an":
        return 1.0 / (1.0 - z)
    elif method == "hinh_thang":
        return (1.0 + z / 2.0) / (1.0 - z / 2.0)
    else:
        raise ValueError("Method không hợp lệ khi tính R(z)")


def analyze_real_stability_interval(method, z_min=-5.0, z_max=1.0, n_points=2001):
    """
    Xấp xỉ miền ổn định trên trục thực (z ∈ R) sao cho |R(z)| ≤ 1.
    Dùng để kiểm tra lại công thức lý thuyết và in cho bài thi.
    """
    zs = np.linspace(z_min, z_max, n_points)
    Rs = amplification_factor(method, zs)
    stable = np.abs(Rs) <= 1.0 + 1e-10

    intervals = []
    in_segment = False
    start = None
    for i, is_stable in enumerate(stable):
        if is_stable and not in_segment:
            in_segment = True
            start = zs[i]
        elif not is_stable and in_segment:
            in_segment = False
            end = zs[i - 1]
            intervals.append((start, end))
    if in_segment:
        intervals.append((start, zs[-1]))

    return intervals


def print_stability_info():
    """
    In thông tin miền ổn định cho 3 phương pháp trên bài test y' = λy.
    """
    print("\n" + "=" * 70)
    print("PHÂN TÍCH MIỀN ỔN ĐỊNH (BÀI TEST y' = λy,  z = h·λ)")
    print("=" * 70)

    print("- Bài test: y' = λy (λ có thể là số phức).")
    print("- Đặt z = h·λ. Khi đó nghiệm rời rạc có dạng: y_{n+1} = R(z)·y_n")
    print("- Tiêu chuẩn ổn định tuyệt đối: |R(z)| ≤ 1")

    methods = [
        ("Euler hiện", "euler_hien"),
        ("Euler ẩn", "euler_an"),
        ("Hình thang", "hinh_thang"),
    ]

    for name, key in methods:
        print(f"\n- {name}:")
        if key == "euler_hien":
            print("  B1) Công thức phương pháp:")
            print("      y_{n+1} = y_n + h·f(t_n, y_n)")
            print("      Với f = λy ⇒ y_{n+1} = y_n + h·λ·y_n = (1 + hλ)·y_n")
            print("  B2) Đặt z = hλ ⇒ R(z) = 1 + z")
            print("  B3) Điều kiện ổn định: |R(z)| ≤ 1 ⇔ |1 + z| ≤ 1")
            print("      ⇒ Miền ổn định là đĩa: tâm -1, bán kính 1 (trong mặt phẳng phức).")
            print("      Trên trục thực: |1 + z| ≤ 1 ⇔ -2 ≤ z ≤ 0")
        elif key == "euler_an":
            print("  B1) Công thức phương pháp:")
            print("      y_{n+1} = y_n + h·f(t_{n+1}, y_{n+1})")
            print("      Với f = λy ⇒ y_{n+1} = y_n + h·λ·y_{n+1}")
            print("      ⇒ (1 - hλ)·y_{n+1} = y_n ⇒ y_{n+1} = 1/(1 - hλ) · y_n")
            print("  B2) Đặt z = hλ ⇒ R(z) = 1 / (1 - z)")
            print("  B3) Điều kiện ổn định: |R(z)| ≤ 1 ⇔ |1/(1 - z)| ≤ 1 ⇔ |1 - z| ≥ 1")
            print("      ⇒ Miền ổn định là phía ngoài (và trên) đường tròn: tâm 1, bán kính 1.")
            print("      ⇒ A-stable: ổn định với mọi z có Re(z) ≤ 0.")
            print("      Trên trục thực: z ≤ 0 đều thỏa |1 - z| = 1 - z ≥ 1.")
        else:
            print("  B1) Công thức hình thang (Crank–Nicolson):")
            print("      y_{n+1} = y_n + (h/2)·(f(t_n,y_n) + f(t_{n+1},y_{n+1}))")
            print("      Với f = λy ⇒ y_{n+1} = y_n + (h/2)·λ·(y_n + y_{n+1})")
            print("      ⇒ (1 - hλ/2)·y_{n+1} = (1 + hλ/2)·y_n")
            print("      ⇒ y_{n+1} = (1 + hλ/2)/(1 - hλ/2) · y_n")
            print("  B2) Đặt z = hλ ⇒ R(z) = (1 + z/2) / (1 - z/2)")
            print("  B3) Điều kiện ổn định: |R(z)| ≤ 1")
            print("      ⇔ |1 + z/2| ≤ |1 - z/2|  (nhân chéo vì mẫu ≠ 0)")
            print("      ⇔ Re(z) ≤ 0  (tính chất chuẩn: CN là A-stable)")
            print("      ⇒ Miền ổn định là nửa mặt phẳng trái (kể cả trục ảo).")
            print("      Trên trục thực: z ≤ 0 đều ổn định.")

        intervals = analyze_real_stability_interval(key, z_min=-10, z_max=5, n_points=15001)
        if intervals:
            print("  Miền ổn định (xấp xỉ) trên trục thực (z ∈ R, |R(z)| ≤ 1):")
            for a, b in intervals:
                print(f"    z ∈ [{a:.4f}, {b:.4f}]")
        else:
            print("  Không tìm thấy đoạn ổn định hữu hạn trên trục thực trong khoảng khảo sát.")

        print("  Ghi nhớ: miền ổn định là tập z sao cho |R(z)| ≤ 1.")

    print("=" * 70 + "\n")


def solve_euler_hien(func, t_span, y0, h, labels, explain_steps=False, explain_first_n=3, explain_last_n=0):
    """
    Euler hiện:
      y_{n+1} = y_n + h·F(t_n, y_n)
    """
    t_values = _build_grid(t_span, h)
    n_steps = len(t_values)

    y_values = np.zeros((n_steps, len(y0)))
    y_values[0] = y0

    print(f"Đang giải bằng Euler Hiện ({n_steps} bước)...")
    # Sửa: Lấy nhãn cho header nếu có, nếu không thì để mặc định
    print(f"{'t':>8} | " + " | ".join(f"{name:>10}" for name in labels))
    for i in range(n_steps - 1):
        t_curr = t_values[i]
        y_curr = y_values[i]

        f_curr = func(t_curr, y_curr)
        y_next = y_curr + h * f_curr
        y_values[i + 1] = y_next

        if (i < 5 or i >= n_steps - 5):
            print(f"{t_curr:>8.4f} | " + " | ".join(f"{y_curr[j]:>10.4f}" for j in range(len(y_curr))))
    return t_values, y_values


def solve_euler_an(func, t_span, y0, h, labels, explain_steps=False, explain_first_n=3, explain_last_n=0):
    """
    Euler ẩn:
      y_{n+1} = y_n + h·F(t_{n+1}, y_{n+1})
    ⇔ Giải G(z)=z - y_n - h·F(t_{n+1}, z)=0
    """
    t_values = _build_grid(t_span, h)
    n_steps = len(t_values)

    y_values = np.zeros((n_steps, len(y0)))
    y_values[0] = y0

    print(f"Đang giải bằng Euler Ẩn ({n_steps} bước)...")
    # In công thức tổng hợp cố định, không phụ thuộc tham số/biến
    print("\n[Euler ẩn] Công thức tổng quát (cố định):")
    print("  y_{n+1} = y_n + h·F(t_{n+1}, y_{n+1})")
    print("  ⇔ Giải G(z) = z - y_n - h·F(t_{n+1}, z) = 0")
    print("  Guess gợi ý: z0 = y_n + h·F(t_n, y_n) (Euler hiện)")
    print("  Dùng solver (fsolve/Newton) để tìm z = y_{n+1}")
    print(f"{'t':>8} | " + " | ".join(f"{name:>10}" for name in labels))
    for i in range(n_steps - 1):
        t_curr = t_values[i]
        t_next = t_values[i + 1]
        y_curr = y_values[i]

        def equation(z):
            return z - y_curr - h * func(t_next, z)

        # Dự đoán nghiệm ban đầu bằng Euler hiện để fsolve chạy nhanh
        f_curr = func(t_curr, y_curr)
        guess = y_curr + h * f_curr
        y_next = fsolve(equation, guess)
        y_values[i + 1] = y_next        
        if (i < 5 or i >= n_steps - 5):
            print(f"{t_curr:>8.4f} | " + " | ".join(f"{y_curr[j]:>10.4f}" for j in range(len(y_curr))))

    return t_values, y_values


def solve_hinh_thang(func, t_span, y0, h, labels, explain_steps=False, explain_first_n=3, explain_last_n=0):
    """
    Hình thang (Crank–Nicolson):
      y_{n+1} = y_n + (h/2)·(F(t_n,y_n) + F(t_{n+1},y_{n+1}))
    ⇔ Giải H(z)=z - y_n - (h/2)·(F(t_n,y_n) + F(t_{n+1},z))=0
    """
    t_values = _build_grid(t_span, h)
    n_steps = len(t_values)

    y_values = np.zeros((n_steps, len(y0)))
    y_values[0] = y0

    print(f"Đang giải bằng Hình thang ({n_steps} bước)...")
    # In công thức tổng hợp cố định, không phụ thuộc tham số/biến
    print("\n[Hình thang] Công thức tổng quát (cố định):")
    print("  y_{n+1} = y_n + (h/2)·(F(t_n,y_n) + F(t_{n+1},y_{n+1}))")
    print("  ⇔ Giải H(z) = z - y_n - (h/2)·(F(t_n,y_n) + F(t_{n+1}, z)) = 0")
    print("  Guess gợi ý: z0 = y_n + h·F(t_n, y_n) (Euler hiện)")
    print("  Dùng solver (fsolve/Newton) để tìm z = y_{n+1}")
    print(f"{'t':>8} | " + " | ".join(f"{name:>10}" for name in labels))
    for i in range(n_steps - 1):
        t_curr = t_values[i]
        t_next = t_values[i + 1]
        y_curr = y_values[i]

        f_curr = func(t_curr, y_curr)

        def equation(z):
            return z - y_curr - (h / 2) * (f_curr + func(t_next, z))

        guess = y_curr + h * f_curr
        y_next = fsolve(equation, guess)
        y_values[i + 1] = y_next

        if (i < 5 or i >= n_steps - 5):
            print(f"{t_curr:>8.4f} | " + " | ".join(f"{y_curr[j]:>10.4f}" for j in range(len(y_curr))))
            

    return t_values, y_values

# ==========================================
# 3. CHƯƠNG TRÌNH CHÍNH (MAIN)
# ==========================================

# --- CẤU HÌNH ---
# Bạn đổi chữ cái ở đây để giải bài khác: 'a', 'b', 'c', 'd', 'e', 'f'
CHOICE = 'f'  
# Bật/tắt in công thức “đi thi”
PRINT_EXAM_INFO = True
# Bật/tắt in công thức ngay trong các bước lặp (nhiều output)
EXPLAIN_ITERATIONS = True
# Chỉ in chi tiết cho N bước đầu và M bước cuối để tránh ngập màn hình
EXPLAIN_FIRST_N = 3
EXPLAIN_LAST_N = 0
# Bật/tắt in phân tích miền ổn định cho 3 phương pháp (bài test y' = λy)
ANALYZE_STABILITY = True
# ----------------

data = get_problem_config(CHOICE)
if data:
    func, t_span, y0, h, labels = data

    if PRINT_EXAM_INFO:
        print_exam_info(CHOICE, t_span, y0, h, labels)

    if ANALYZE_STABILITY:
        print_stability_info()
    
    # Giải bằng cả 3 phương pháp để so sánh
    t1, y1 = solve_euler_hien(
        func, t_span, y0, h, labels,
        explain_steps=EXPLAIN_ITERATIONS,
        explain_first_n=EXPLAIN_FIRST_N,
        explain_last_n=EXPLAIN_LAST_N
    )
    t2, y2 = solve_euler_an(
        func, t_span, y0, h, labels,
        explain_steps=EXPLAIN_ITERATIONS,
        explain_first_n=EXPLAIN_FIRST_N,
        explain_last_n=EXPLAIN_LAST_N
    )
    t3, y3 = solve_hinh_thang(
        func, t_span, y0, h, labels,
        explain_steps=EXPLAIN_ITERATIONS,
        explain_first_n=EXPLAIN_FIRST_N,
        explain_last_n=EXPLAIN_LAST_N
    )
    
    # Vẽ đồ thị: mỗi phương pháp một hình, trong đó chứa tất cả các biến
    methods_plot = [
        ("Euler Hiện", t1, y1),
        ("Euler Ẩn", t2, y2),
        ("Hình thang", t3, y3),
    ]

    for method_name, t_vals, y_vals in methods_plot:
        plt.figure(figsize=(12, 6))
        for j, label in enumerate(labels):
            plt.plot(t_vals, y_vals[:, j], label=label)

        plt.title(f"Kết quả bài toán ({CHOICE}) - {method_name}")
        plt.xlabel("Thời gian t")
        plt.ylabel("Giá trị các ẩn")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # In vài giá trị cuối để kiểm tra
    print(f"\nGiá trị cuối cùng tại t={t_span[1]}:")
    print(f"Euler Hiện: {y1[-1]}")
    print(f"Euler Ẩn  : {y2[-1]}")
    print(f"Hình thang: {y3[-1]}")

else:
    print("Mã bài toán không hợp lệ!")