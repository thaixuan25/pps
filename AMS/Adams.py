import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


# ==========================================
# 0. KHUÔN MẪU DÙNG TRONG BÀI THI
#    (CHỈ CẦN SỬA Ở PHẦN NÀY)
# ==========================================

def ode_function(t, Y):
    """
    Hàm f(t, Y) trong bài toán Cauchy:
        Y' = f(t, Y)
    - Với 1 ẩn:  Y là mảng 1 phần tử, ví dụ: y = Y[0]
    - Với nhiều ẩn: ví dụ: x, y, z = Y

    TODO: SỬA HÀM NÀY CHO ĐÚNG ĐỀ BÀI.
    Ví dụ minh họa (1 ẩn):
        y' = y - t^2 + 1
    """
    # Ví dụ: y' = y - t^2 + 1
    x, y, z = Y
    dx = 0.4*x*(1 - x/20) + 0.4*y - 0.3*x*z
    dy = 0.7*y*(1 - y/25) - 0.4*y - 0.4*y*z
    dz = -0.3*z + 0.35*(x + y)*z
    return np.array([dx, dy, dz], dtype=float)


# Khoảng thời gian và điều kiện đầu (SỬA CHO ĐÚNG ĐỀ)
T_SPAN = (0, 1500)       # (t0, T)
Y0 = np.array([12.0, 18.0, 8.0])         # giá trị đầu: [y0] hoặc [x0, y0, z0, ...]
H = 0.1                     # bước h
LABELS = ["x", "y", "z"]               # tên các ẩn để in bảng

# Chọn phương pháp:
#   "ab1", "ab2", "ab3", "ab4", "ab5", "ab6" = Adams-Bashforth bậc 1-6 (Dùng đến cấp 3 thôi vì cấp 4 trở lên ko hội tụ)
#   "am1", "am2", "am3", "am4", "am5", "am6" = Adams-Moulton bậc 1-6 (Nên dùng vì tính ổn cực ổn từ cấp 3, 4)
#   "pc1"... "pc6" = Predictor–Corrector AB(s)–AM(s) (PECE): dự báo bằng AB, hiệu chỉnh bằng AM (thường 1 lần sửa)
METHOD = "am4"

# Cấu hình in giải thích từng bước (phục vụ bài thi)
EXPLAIN_STEPS = True         # True: in công thức & các bước tính toán
EXPLAIN_FIRST_N = 5          # số bước đầu cần in chi tiết
PLOT_RESULT = True           # True: vẽ đồ thị nghiệm theo t
PC_CORRECTIONS = 1           # số lần hiệu chỉnh trong chế độ "pc*" (1 = PECE tiêu chuẩn)


# ==========================================
# 1. CÁC HỆ SỐ CHO ADAMS-BASHFORTH VÀ ADAMS-MOULTON
# ==========================================

def get_adams_bashforth_coeffs(s):
    """
    Trả về các hệ số beta cho công thức Adams-Bashforth s bước.
    
    Công thức tổng quát:
        y_{n+1} = y_n + h * sum_{i=0}^{s-1} beta_i * f_{n-i}
    
    Args:
        s: số bước (1, 2, 3, 4, ...)
    
    Returns:
        beta: mảng các hệ số [beta_0, beta_1, ..., beta_{s-1}]
    """
    if s == 1:
        # AB1 (Euler hiện)
        return np.array([1.0])
    elif s == 2:
        # AB2: y_{n+1} = y_n + h/2*(3*f_n - f_{n-1})
        return np.array([3/2, -1/2])
    elif s == 3:
        # AB3: y_{n+1} = y_n + h/12*(23*f_n - 16*f_{n-1} + 5*f_{n-2})
        return np.array([23/12, -16/12, 5/12])
    elif s == 4:
        # AB4: y_{n+1} = y_n + h/24*(55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})
        return np.array([55/24, -59/24, 37/24, -9/24])
    elif s == 5:
        # AB5: y_{n+1} = y_n + h/720*(1901*f_n - 2713*f_{n-1} + 15487*f_{n-2} - 586*f_{n-3} + 6737*f_{n-4} - 263*f_{n-5})
        # Hệ số đúng: [1901/720, -1387/360, 109/30, -637/360, 251/720]
        return np.array([1901/720, -1387/360, 109/30, -637/360, 251/720])
    elif s == 6:
        # AB6: y_{n+1} = y_n + h/1440*(475*f_n - 1427*f_{n-1} + 798*f_{n-2} - 482*f_{n-3} + 173*f_{n-4} - 27*f_{n-5})
        return np.array([4277/1440, -2641/480, 4991/720, -3649/720, 959/480, -95/288])
    else:
        raise ValueError(f"Chưa hỗ trợ Adams-Bashforth bậc {s}. Chỉ hỗ trợ s = 1, 2, 3, 4, 5, 6.")


def get_adams_moulton_coeffs(s):
    """
    Trả về các hệ số beta cho công thức Adams-Moulton s bước.
    
    Công thức tổng quát:
        y_{n+1} = y_n + h * sum_{i=0}^{s} beta_i * f_{n+1-i}
    
    Args:
        s: số bước (1, 2, 3, 4, ...)
    
    Returns:
        beta: mảng các hệ số [beta_0, beta_1, ..., beta_s]
              trong đó beta_0 là hệ số của f_{n+1}
    """
    # Lưu ý: với cách ký hiệu đang dùng, "s bước" nghĩa là sử dụng (s+1) giá trị f:
    #   {f_{n+1}, f_n, f_{n-1}, ..., f_{n+1-s}}
    # nên mảng beta luôn có độ dài s+1.
    if s == 0:
        return np.array([1.0])
    elif s == 1:
        # AM1 (hình thang / Crank–Nicolson, bậc 2):
        #   y_{n+1} = y_n + h/2 * (f_{n+1} + f_n)
        #   ⇒ beta_0 = 1/2, beta_1 = 1/2
        return np.array([1/2, 1/2])
    elif s == 2:
        # AM2 (bậc 3):
        #   y_{n+1} = y_n + h/12 * (5 f_{n+1} + 8 f_n - f_{n-1})
        #   ⇒ beta = [5/12, 8/12, -1/12]
        return np.array([5/12, 8/12, -1/12])
    elif s == 3:
        # AM3 (bậc 4):
        #   y_{n+1} = y_n + h/24 * (9 f_{n+1} + 19 f_n - 5 f_{n-1} + f_{n-2})
        #   ⇒ beta = [9/24, 19/24, -5/24, 1/24]
        return np.array([9/24, 19/24, -5/24, 1/24])
    elif s == 4:
        # AM4 (bậc 5, 4 bước):
        #   y_{n+1} = y_n + h/720 * (251 f_{n+1} + 646 f_n
        #                            - 264 f_{n-1} + 106 f_{n-2} - 19 f_{n-3})
        #   ⇒ beta = [251/720, 646/720, -264/720, 106/720, -19/720]
        return np.array([251/720, 646/720, -264/720, 106/720, -19/720])
    elif s == 5:
        # AM5 (bậc 6, 5 bước):
        #   y_{n+1} = y_n + h/1440 * (475 f_{n+1} + 1427 f_n - 798 f_{n-1} + 482 f_{n-2} - 173 f_{n-3} + 27 f_{n-4})
        #   ⇒ beta = [475/1440, 1427/1440, -798/1440, 482/1440, -173/1440, 27/1440]
        return np.array([475/1440, 1427/1440, -798/1440, 482/1440, -173/1440, 27/1440])
    elif s == 6:
        # AM6 (bậc 7, 6 bước):
        #   y_{n+1} = y_n + h/12096 * (1901 f_{n+1} + 6511 f_n - 4641 f_{n-1} + 2520 f_{n-2} - 840 f_{n-3} + 168 f_{n-4} - 18 f_{n-5})
        #   ⇒ beta = [1901/12096, 6511/12096, -4641/12096, 2520/12096, -840/12096, 168/12096, -18/12096]
        return np.array([
            1901/12096,
            6511/12096,
            -4641/12096,
            2520/12096,
            -840/12096,
            168/12096,
            -18/12096,
        ])
    else:
        raise ValueError(f"Chưa hỗ trợ Adams-Moulton bậc {s}. Chỉ hỗ trợ s = 1, 2, 3, 4, 5, 6.")


# ==========================================
# 2. CÁC HÀM HỖ TRỢ
# ==========================================

def _to_array(y):
    """
    Đảm bảo y luôn là vector 1D numpy.
    Trả thêm cờ is_scalar để khi trả kết quả có thể ép về dạng số nếu cần.
    """
    if np.isscalar(y):
        return np.array([y], dtype=float), True
    y_arr = np.asarray(y, dtype=float).ravel()
    return y_arr, False


def rk4_step(f, t, y, h):
    """
    Thực hiện 1 bước Runge-Kutta bậc 4 (dùng để khởi tạo cho Adams).
    """
    y, is_scalar = _to_array(y)
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)
    y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_next[0] if is_scalar else y_next


# ==========================================
# 3. PHƯƠNG PHÁP ADAMS-BASHFORTH (HIỆN)
# ==========================================

def adams_bashforth_step(f, t_values, y_values, f_values, n, s, h):
    """
    Thực hiện 1 bước Adams-Bashforth s bước.
    
    Args:
        f: hàm f(t, y)
        t_values: mảng các thời điểm
        y_values: mảng các giá trị y đã tính
        f_values: mảng các giá trị f đã tính
        n: chỉ số bước hiện tại (tính y_{n+1})
        s: số bước của phương pháp
        h: bước thời gian
    
    Returns:
        y_{n+1}
    """
    beta = get_adams_bashforth_coeffs(s)
    
    # Công thức: y_{n+1} = y_n + h * sum_{i=0}^{s-1} beta_i * f_{n-i}
    sum_term = np.zeros_like(y_values[n])
    for i in range(s):
        if n - i >= 0:
            sum_term += beta[i] * f_values[n - i]
    
    y_next = y_values[n] + h * sum_term
    return y_next


def solve_adams_bashforth(f, t_span, y0, h, s, explain_steps=False, explain_first_n=3):
    """
    Giải bài toán y' = f(t, y) bằng phương pháp Adams-Bashforth s bước.
    
    Args:
        f: hàm f(t, y)
        t_span: (t0, T) - khoảng thời gian
        y0: giá trị ban đầu
        h: bước thời gian
        s: số bước (1, 2, 3, 4)
        explain_steps: có in chi tiết các bước không
        explain_first_n: số bước đầu in chi tiết
    
    Returns:
        t_values: mảng các thời điểm
        y_values: mảng các giá trị y
    """
    t0, T = t_span
    y0_arr, is_scalar = _to_array(y0)
    
    # Xây lưới thời gian
    t_values = np.arange(t0, T + h / 1000.0, h, dtype=float)
    n_steps = len(t_values)
    
    y_values = np.zeros((n_steps, len(y0_arr)), dtype=float)
    f_values = np.zeros((n_steps, len(y0_arr)), dtype=float)
    
    y_values[0] = y0_arr
    f_values[0] = f(t0, y0_arr)
    
    # Khởi tạo các giá trị ban đầu bằng RK4
    if explain_steps:
        print("\n" + "="*70)
        print(f"KHỞI TẠO CÁC GIÁ TRỊ BAN ĐẦU BẰNG RK4 (cần {s-1} giá trị)")
        print("="*70)
    print(f"Sử dụng công thức RK4 thường dùng cho {s-1} bước đầu tiên:")
    print("  y_{n+1} = y_n + h/6 * (k1 + 2k2 + 2k3 + k4)")
    print("  với:")
    print("    k1 = f(t_n, y_n)")
    print("    k2 = f(t_n + h/2, y_n + h/2 * k1)")
    print("    k3 = f(t_n + h/2, y_n + h/2 * k2)")
    print("    k4 = f(t_n + h, y_n + h * k3)")
    for i in range(min(s - 1, n_steps - 1)):
        if explain_steps and i < explain_first_n:
            print(f"\nBước {i+1}: Tính y_{i+1} bằng RK4")
            print(f"  t_{i} = {t_values[i]:.4f}, y_{i} = {y_values[i]}")
        
        y_values[i + 1] = rk4_step(f, t_values[i], y_values[i], h)
        f_values[i + 1] = f(t_values[i + 1], y_values[i + 1])
        
        if explain_steps and i < explain_first_n:
            print(f"  t_{i+1} = {t_values[i+1]:.4f}, y_{i+1} = {y_values[i+1]}")
            print(f"  f_{i+1} = f(t_{i+1}, y_{i+1}) = {f_values[i+1]}")
    
    # In công thức Adams-Bashforth
    if explain_steps:
        print("\n" + "="*70)
        print(f"CÔNG THỨC ADAMS-BASHFORTH {s} BƯỚC")
        print("="*70)
        beta = get_adams_bashforth_coeffs(s)
        formula = f"y_{{n+1}} = y_n + h * ("
        terms = []
        for i in range(s):
            if beta[i] >= 0 and i > 0:
                terms.append(f"+ {beta[i]:.6f}*f_{{n-{i}}}")
            else:
                terms.append(f"{beta[i]:.6f}*f_{{n-{i}}}")
        formula += "".join(terms) + ")"
        print(formula)
        print("="*70 + "\n")
    
    # Áp dụng Adams-Bashforth cho các bước còn lại
    for i in range(s - 1, n_steps - 1):
        if explain_steps and i < explain_first_n + s - 1:
            print(f"\nBước {i+1}: Tính y_{i+1} bằng Adams-Bashforth {s} bước")
            print(f"  t_{i} = {t_values[i]:.4f}, y_{i} = {y_values[i]}")
            beta = get_adams_bashforth_coeffs(s)
            print(f"  Các giá trị f đã có:")
            for j in range(s):
                print(f"    f_{{n-{j}}} = f_{i-j} = {f_values[i-j]}")
            print(f"  Tính: y_{i+1} = y_{i} + h * (", end="")
            sum_parts = []
            for j in range(s):
                term = f"{beta[j]:.6f}*f_{i-j}"
                sum_parts.append(term)
            print(" + ".join(sum_parts) + ")")
        
        y_values[i + 1] = adams_bashforth_step(f, t_values, y_values, f_values, i, s, h)
        f_values[i + 1] = f(t_values[i + 1], y_values[i + 1])
        
        if explain_steps and i < explain_first_n + s - 1:
            print(f"  Kết quả: y_{i+1} = {y_values[i+1]}")
            print(f"  f_{i+1} = f(t_{i+1}, y_{i+1}) = {f_values[i+1]}")
    
    if is_scalar:
        return t_values, y_values[:, 0]
    return t_values, y_values


# ==========================================
# 4. PHƯƠNG PHÁP ADAMS-MOULTON (ẨN)
# ==========================================

def adams_moulton_step(f, t_values, y_values, f_values, n, s, h, tol=1e-8, max_iter=10):
    """
    Thực hiện 1 bước Adams-Moulton s bước (phương pháp ẩn).
    
    Công thức: y_{n+1} = y_n + h * sum_{i=0}^{s} beta_i * f_{n+1-i}
    Trong đó beta_0 là hệ số của f_{n+1} (ẩn).
    
    Giải bằng phương pháp lặp Newton hoặc dự đoán-sửa.
    """
    beta = get_adams_moulton_coeffs(s)
    
    # Dự đoán ban đầu bằng Adams-Bashforth cùng bậc
    y_predict = adams_bashforth_step(f, t_values, y_values, f_values, n, s, h)
    
    # Lặp để giải phương trình ẩn
    y_next = y_predict.copy()
    t_next = t_values[n + 1]
    
    for iteration in range(max_iter):
        f_next = f(t_next, y_next)
        
        # Tính phần đã biết (không chứa f_{n+1})
        sum_known = np.zeros_like(y_values[n])
        for i in range(1, s + 1):
            if n + 1 - i >= 0:
                sum_known += beta[i] * f_values[n + 1 - i]
        
        # Công thức: y_{n+1} = y_n + h * (beta_0 * f_{n+1} + sum_known)
        y_new = y_values[n] + h * (beta[0] * f_next + sum_known)
        
        # Kiểm tra hội tụ
        error = np.linalg.norm(y_new - y_next)
        if error < tol:
            break
        
        y_next = y_new
    
    return y_next


def solve_adams_moulton(f, t_span, y0, h, s, explain_steps=False, explain_first_n=3, tol=1e-8, max_iter=10):
    """
    Giải bài toán y' = f(t, y) bằng phương pháp Adams-Moulton s bước.
    
    Args:
        f: hàm f(t, y)
        t_span: (t0, T) - khoảng thời gian
        y0: giá trị ban đầu
        h: bước thời gian
        s: số bước (1, 2, 3, 4)
        explain_steps: có in chi tiết các bước không
        explain_first_n: số bước đầu in chi tiết
    
    Returns:
        t_values: mảng các thời điểm
        y_values: mảng các giá trị y
    """
    t0, T = t_span
    y0_arr, is_scalar = _to_array(y0)
    
    # Xây lưới thời gian
    t_values = np.arange(t0, T + h / 1000.0, h, dtype=float)
    n_steps = len(t_values)
    
    y_values = np.zeros((n_steps, len(y0_arr)), dtype=float)
    f_values = np.zeros((n_steps, len(y0_arr)), dtype=float)
    
    y_values[0] = y0_arr
    f_values[0] = f(t0, y0_arr)
    
    # Khởi tạo các giá trị ban đầu bằng RK4
    if explain_steps:
        print("\n" + "="*70)
        print(f"KHỞI TẠO CÁC GIÁ TRỊ BAN ĐẦU BẰNG RK4 (cần {s-1} giá trị)")
        print("="*70)
    print(f"Sử dụng công thức RK4 thường dùng cho {s-1} bước đầu tiên:")
    print("  y_{n+1} = y_n + h/6 * (k1 + 2k2 + 2k3 + k4)")
    print("  với:")
    print("    k1 = f(t_n, y_n)")
    print("    k2 = f(t_n + h/2, y_n + h/2 * k1)")
    print("    k3 = f(t_n + h/2, y_n + h/2 * k2)")
    print("    k4 = f(t_n + h, y_n + h * k3)")
    for i in range(min(s - 1, n_steps - 1)):
        if explain_steps and i < explain_first_n:
            print(f"\nBước {i+1}: Tính y_{i+1} bằng RK4")
            print(f"  t_{i} = {t_values[i]:.4f}, y_{i} = {y_values[i]}")
        
        y_values[i + 1] = rk4_step(f, t_values[i], y_values[i], h)
        f_values[i + 1] = f(t_values[i + 1], y_values[i + 1])
        
        if explain_steps and i < explain_first_n:
            print(f"  t_{i+1} = {t_values[i+1]:.4f}, y_{i+1} = {y_values[i+1]}")
            print(f"  f_{i+1} = f(t_{i+1}, y_{i+1}) = {f_values[i+1]}")
    
    # In công thức Adams-Moulton
    if explain_steps:
        print("\n" + "="*70)
        print(f"CÔNG THỨC ADAMS-MOULTON {s} BƯỚC")
        print("="*70)
        beta = get_adams_moulton_coeffs(s)
        formula = f"y_{{n+1}} = y_n + h * ("
        terms = []
        for i in range(s + 1):
            if i == 0:
                terms.append(f"{beta[0]:.6f}*f_{{n+1}}")
            else:
                if beta[i] >= 0:
                    terms.append(f"+ {beta[i]:.6f}*f_{{n+1-{i}}}")
                else:
                    terms.append(f"{beta[i]:.6f}*f_{{n+1-{i}}}")
        formula += "".join(terms) + ")"
        print(formula)
        print("(Lưu ý: f_{n+1} phụ thuộc vào y_{n+1}, nên đây là phương trình ẩn)")
        print("="*70 + "\n")
    
    # Áp dụng Adams-Moulton cho các bước còn lại
    for i in range(s - 1, n_steps - 1):
        if explain_steps and i < explain_first_n + s - 1:
            print(f"\nBước {i+1}: Tính y_{i+1} bằng Adams-Moulton {s} bước")
            print(f"  t_{i} = {t_values[i]:.4f}, y_{i} = {y_values[i]}")
            beta = get_adams_moulton_coeffs(s)
            print(f"  Dự đoán ban đầu bằng Adams-Bashforth:")
            y_predict = adams_bashforth_step(f, t_values, y_values, f_values, i, s, h)
            print(f"    y_{{n+1}}^(0) = {y_predict}")
            print(f"  Lặp để giải phương trình ẩn...")
        
        y_values[i + 1] = adams_moulton_step(
            f, t_values, y_values, f_values, i, s, h, tol=tol, max_iter=max_iter
        )
        f_values[i + 1] = f(t_values[i + 1], y_values[i + 1])
        
        if explain_steps and i < explain_first_n + s - 1:
            print(f"  Kết quả: y_{i+1} = {y_values[i+1]}")
            print(f"  f_{i+1} = f(t_{i+1}, y_{i+1}) = {f_values[i+1]}")
    
    if is_scalar:
        return t_values, y_values[:, 0]
    return t_values, y_values


# ==========================================
# 5. HÀM TỔNG QUÁT GIẢI BÀI TOÁN
# ==========================================

def solve_adams(f, t_span, y0, h, method, explain_steps=False, explain_first_n=3):
    """
    Giải bài toán y' = f(t, y) bằng phương pháp Adams.
    
    Args:
        f: hàm f(t, y)
        t_span: (t0, T) - khoảng thời gian
        y0: giá trị ban đầu
        h: bước thời gian
        method: "ab1", "ab2", "ab3", "ab4", "am1", "am2", "am3", "am4"
        explain_steps: có in chi tiết các bước không
        explain_first_n: số bước đầu in chi tiết
    
    Returns:
        t_values: mảng các thời điểm
        y_values: mảng các giá trị y
    """
    method = method.lower()
    
    if method.startswith("ab"):
        s = int(method[2])
        return solve_adams_bashforth(f, t_span, y0, h, s, explain_steps, explain_first_n)
    elif method.startswith("am"):
        s = int(method[2])
        return solve_adams_moulton(f, t_span, y0, h, s, explain_steps, explain_first_n)
    elif method.startswith("pc"):
        # Predictor–Corrector AB(s)–AM(s) kiểu PECE:
        # - Dự báo bằng AB(s)
        # - Hiệu chỉnh bằng AM(s) với số vòng lặp = PC_CORRECTIONS (mặc định 1 lần)
        s = int(method[2])
        return solve_adams_moulton(
            f,
            t_span,
            y0,
            h,
            s,
            explain_steps=explain_steps,
            explain_first_n=explain_first_n,
            tol=0.0,
            max_iter=max(1, int(PC_CORRECTIONS)),
        )
    else:
        raise ValueError(f"Phương pháp '{method}' không hợp lệ. Chọn 'ab1'-'ab4' hoặc 'am1'-'am4'.")


# ==========================================
# 6. HÀM VẼ ĐỒ THỊ VÀ IN KẾT QUẢ
# ==========================================

def print_results_table(t_values, y_values, labels, n_print=1):
    """
    In bảng kết quả.
    """
    print("\n" + "="*70)
    print("BẢNG KẾT QUẢ")
    print("="*70)
    
    n_steps = len(t_values)
    step = max(1, n_print)
    
    # In header
    header = f"{'n':<5} {'t_n':<12}"
    for label in labels:
        header += f" {label:>15}"
    print(header)
    print("-" * 70)
    
    # Kiểm tra shape của y_values
    is_1d = y_values.ndim == 1
    
    # In các dòng định kỳ (mặc định khoảng n_steps // n_print)
    for i in range(0, n_steps, step):
        if i <= 5 or i >= n_steps - 5:
            row = f"{i:<5} {t_values[i]:<12.6f}"
            if is_1d:
                row += f" {y_values[i]:>15.8f}"
            else:
                for j in range(len(labels)):
                    row += f" {y_values[i, j]:>15.8f}"
            print(row)
    
    # In dòng cuối
    if (n_steps - 1) % step != 0:
        i = n_steps - 1
        row = f"{i:<5} {t_values[i]:<12.6f}"
        if is_1d:
            row += f" {y_values[i]:>15.8f}"
        else:
            for j in range(len(labels)):
                row += f" {y_values[i, j]:>15.8f}"
        print(row)
    
    print("="*70)


def plot_results(t_values, y_values, labels):
    """
    Vẽ đồ thị kết quả.
    """
    plt.figure(figsize=(10, 6))
    
    if len(labels) == 1:
        plt.plot(t_values, y_values, 'b-', linewidth=2, label=labels[0])
        plt.ylabel(labels[0], fontsize=12)
    else:
        for i, label in enumerate(labels):
            plt.plot(t_values, y_values[:, i], linewidth=2, label=label)
        plt.ylabel('Giá trị', fontsize=12)
        plt.legend()
    
    plt.xlabel('t', fontsize=12)
    plt.title('Nghiệm của phương trình vi phân bằng phương pháp Adams', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# 7. CHƯƠNG TRÌNH CHÍNH
# ==========================================

if __name__ == "__main__":
    print("="*70)
    print("GIẢI PHƯƠNG TRÌNH VI PHÂN BẰNG PHƯƠNG PHÁP ADAMS")
    print("="*70)
    print(f"\nBài toán: y' = f(t, y)")
    print(f"Khoảng: t ∈ [{T_SPAN[0]}, {T_SPAN[1]}]")
    print(f"Điều kiện đầu: y({T_SPAN[0]}) = {Y0}")
    print(f"Bước h = {H}")
    print(f"Phương pháp: {METHOD.upper()}")
    
    # Giải bài toán
    t_values, y_values = solve_adams(
        ode_function, 
        T_SPAN, 
        Y0, 
        H, 
        METHOD,
        explain_steps=EXPLAIN_STEPS,
        explain_first_n=EXPLAIN_FIRST_N
    )
    
    # In bảng kết quả
    print_results_table(t_values, y_values, LABELS)
    
    # Vẽ đồ thị
    if PLOT_RESULT:
        plot_results(t_values, y_values, LABELS)
