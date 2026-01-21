import numpy as np
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
    x, y = Y
    dx = y
    dy = ((4*t**3 -6*t**2 + 2.25*t)*x - 2*x*y - 39 * np.exp(-t**2)*(1.25*t**2 - 2.75*t -2))/(2*x)
    return np.array([dx, dy], dtype=float)


# Khoảng thời gian và điều kiện đầu (SỬA CHO ĐÚNG ĐỀ)
T_SPAN = (0, 3)       # (t0, T)
Y0 = np.array([39, 39])         # giá trị đầu: [y0] hoặc [x0, y0, z0, ...]
H = 0.01                     # bước h
LABELS = ["x", "y"]               # tên các ẩn để in bảng

# Chọn phương pháp:
#   "rk1" = Euler hiện, "rk2", "rk3", "rk4"
#   "rk_custom_2" = RK2 tự chọn alpha, "rk_custom_3" = RK3 tự chọn alpha
METHOD = "rk4"

# Cấu hình cho RK2 / RK3 tùy chỉnh từ alpha (chỉ dùng nếu METHOD là rk_custom_*)
CUSTOM_S = 3                 # 2 hoặc 3
CUSTOM_ALPHA = [0.0, 1/3, 2/3]    # ví dụ RK2 với alpha = [0, c2]

# Cấu hình in giải thích từng bước (phục vụ bài thi)
EXPLAIN_STEPS = True         # True: in công thức & k1,k2,... cho vài bước đầu
EXPLAIN_FIRST_N = 3          # số bước đầu cần in chi tiết
PRINT_CUSTOM_DERIVATION = True  # True: in cách tìm hệ số cho RK tùy chỉnh
ANALYZE_STABILITY = True     # True: vừa giải vừa in miền ổn định tuyệt đối trên trục thực
PLOT_RESULT = True           # True: vẽ đồ thị nghiệm theo t


# ==========================================
# 1. CÁC BƯỚC RUNGE–KUTTA CỐ ĐỊNH BƯỚC
#    (RK1, RK2, RK3, RK4 cổ điển)
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


def rk_step(f, t, y, h, method="rk4"):
    """
    Thực hiện 1 bước Runge–Kutta cho bài toán y' = f(t, y).

    - f  : hàm f(t, y) trả về cùng kích thước với y
    - t  : thời điểm hiện tại
    - y  : trạng thái hiện tại (scalar hoặc vector)
    - h  : bước thời gian
    - method: "rk1", "rk2", "rk3", "rk4"
    """
    y, is_scalar = _to_array(y)
    method = method.lower()

    k1 = f(t, y)

    if method == "rk1":  # Euler hiện (bậc 1)
        y_next = y + h * k1

    elif method == "rk2":  # Midpoint / Heun tùy theo cách chọn
        # Ở đây dùng RK2 dạng "midpoint":
        #   k1 = f(t_n, y_n)
        #   k2 = f(t_n + h/2, y_n + h/2·k1)
        #   y_{n+1} = y_n + h·k2
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        y_next = y + h * k2

    elif method == "rk3":
        # RK3 cổ điển:
        #   k1 = f(t, y)
        #   k2 = f(t + h/2, y + h/2·k1)
        #   k3 = f(t + h,   y - h·k1 + 2h·k2)
        #   y_{n+1} = y + h/6 · (k1 + 4k2 + k3)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + h, y - h * k1 + 2 * h * k2)
        y_next = y + (h / 6.0) * (k1 + 4 * k2 + k3)

    elif method == "rk4":
        # RK4 cổ điển (phổ biến nhất trong báo cáo)
        #   k1 = f(t, y)
        #   k2 = f(t + h/2, y + h/2·k1)
        #   k3 = f(t + h/2, y + h/2·k2)
        #   k4 = f(t + h,   y + h·k3)
        #   y_{n+1} = y + h/6 · (k1 + 2k2 + 2k3 + k4)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y_next = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    else:
        raise ValueError("method phải là một trong: 'rk1', 'rk2', 'rk3', 'rk4'")

    return y_next[0] if is_scalar else y_next


def _explain_rk_step(f, t, y, h, method, step_index):
    """
    In ra công thức RK và giá trị k1, k2, ... cho một bước (phục vụ bài thi),
    đồng thời trả lại y_{n+1}.
    """
    print(f"\nBƯỚC {step_index}: t_{step_index} = {t:.4f}, Y_{step_index} = {y}")
    method = method.lower()

    if method == "rk1":
        print("Euler hiện (RK1):  Y_{n+1} = Y_n + h·f(t_n, Y_n)")
        k1 = f(t, y)
        print(f"  k1 = f(t_n, Y_n) = f({t:.4f}, {y}) = {k1}")
        y_next = y + h * k1
        print(f"  ⇒ Y_{step_index+1} = Y_{step_index} + h·k1 = {y_next}")
        return y_next

    if method == "rk2":
        print("RK2 (midpoint):")
        print("  k1 = f(t_n, Y_n)")
        print("  k2 = f(t_n + h/2, Y_n + h/2·k1)")
        print("  Y_{n+1} = Y_n + h·k2")
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        print(f"  k1 = f({t:.4f}, {y}) = {k1}")
        print(f"  k2 = f({t + 0.5*h:.4f}, {y + 0.5*h*k1}) = {k2}")
        y_next = y + h * k2
        print(f"  ⇒ Y_{step_index+1} = Y_{step_index} + h·k2 = {y_next}")
        return y_next

    if method == "rk3":
        print("RK3 cổ điển:")
        print("  k1 = f(t_n, Y_n)")
        print("  k2 = f(t_n + h/2, Y_n + h/2·k1)")
        print("  k3 = f(t_n + h,   Y_n - h·k1 + 2h·k2)")
        print("  Y_{n+1} = Y_n + h/6 · (k1 + 4k2 + k3)")
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + h, y - h * k1 + 2 * h * k2)
        print(f"  k1 = f({t:.4f}, {y}) = {k1}")
        print(f"  k2 = f({t + 0.5*h:.4f}, {y + 0.5*h*k1}) = {k2}")
        print(f"  k3 = f({t + h:.4f}, {y - h*k1 + 2*h*k2}) = {k3}")
        y_next = y + (h / 6.0) * (k1 + 4 * k2 + k3)
        print(f"  ⇒ Y_{step_index+1} = Y_{step_index} + h/6·(k1 + 4k2 + k3) = {y_next}")
        return y_next

    if method == "rk4":
        print("RK4 cổ điển:")
        print("  k1 = f(t_n, Y_n)")
        print("  k2 = f(t_n + h/2, Y_n + h/2·k1)")
        print("  k3 = f(t_n + h/2, Y_n + h/2·k2)")
        print("  k4 = f(t_n + h,   Y_n + h·k3)")
        print("  Y_{n+1} = Y_n + h/6 · (k1 + 2k2 + 2k3 + k4)")
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        print(f"  k1 = f({t:.4f}, {y}) = {k1}")
        print(f"  k2 = f({t + 0.5*h:.4f}, {y + 0.5*h*k1}) = {k2}")
        print(f"  k3 = f({t + 0.5*h:.4f}, {y + 0.5*h*k2}) = {k3}")
        print(f"  k4 = f({t + h:.4f}, {y + h*k3}) = {k4}")
        y_next = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        print(f"  ⇒ Y_{step_index+1} = Y_{step_index} + h/6·(k1 + 2k2 + 2k3 + k4) = {y_next}")
        return y_next

    # Nếu method khác, fallback sang rk_step thường
    return rk_step(f, t, y, h, method=method)


def solve_rk(f, t_span, y0, h, method="rk4", explain_steps=False, explain_first_n=3):
    """
    Giải bài toán Cauchy y' = f(t, y),  t ∈ [t0, T],  y(t0) = y0
    bằng công thức Runge–Kutta bậc 1–4 với bước cố định.

    - f      : hàm f(t, y)
    - t_span : (t0, T)
    - y0     : giá trị đầu (scalar hoặc vector)
    - h      : bước thời gian
    - method : "rk1", "rk2", "rk3", "rk4"

    Trả về:
    - t_values : mảng các thời điểm
    - y_values : mảng (N, m), trong đó m là số ẩn
    """
    t0, T = t_span
    y0_arr, is_scalar = _to_array(y0)

    # Xây lưới thời gian (lấy thêm chút để chắc chắn chứa T do sai số float)
    t_values = np.arange(t0, T + h / 1000.0, h, dtype=float)
    n_steps = len(t_values)

    y_values = np.zeros((n_steps, len(y0_arr)), dtype=float)
    y_values[0] = y0_arr

    for i in range(n_steps - 1):
        t = t_values[i]
        y = y_values[i]
        if explain_steps and i < explain_first_n:
            y_next = _explain_rk_step(f, t, y, h, method, i)
        else:
            y_next = rk_step(f, t, y, h, method=method)
        y_values[i + 1] = y_next

    if is_scalar:
        return t_values, y_values[:, 0]
    return t_values, y_values


# ==========================================
# 1b. RK TÙY CHỈNH TỪ HỆ SỐ alpha, beta, gamma
#     (Explicit RK / Butcher tableau)
#     - alpha_i  ~ c_i  (thời điểm nội suy trong 1 bước)
#     - beta_ij  ~ a_ij (ma trận hệ số cho tổ hợp k)
#     - gamma_i  ~ b_i  (trọng số để cộng ra y_{n+1})
# ==========================================

def _validate_explicit_rk_coeffs(alpha, beta, gamma):
    alpha = np.asarray(alpha, dtype=float).ravel()
    gamma = np.asarray(gamma, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float)

    s = len(alpha)
    if s < 1:
        raise ValueError("alpha phải có ít nhất 1 phần tử (số nấc s >= 1).")
    if len(gamma) != s:
        raise ValueError("gamma phải có cùng độ dài với alpha (đều là s).")
    if beta.shape != (s, s):
        raise ValueError("beta phải có kích thước (s, s).")

    # Explicit RK: beta_ij = 0 với j >= i (ma trận tam giác dưới, chéo 0)
    if np.any(np.triu(beta, 0) != 0):
        raise ValueError(
            "Đây không phải explicit RK: beta phải tam giác dưới với phần trên & đường chéo bằng 0."
        )

    # Thường alpha_1 = 0; không bắt buộc nhưng khuyến nghị
    return alpha, beta, gamma


def rk_step_custom(f, t, y, h, alpha, beta, gamma):
    """
    1 bước Runge–Kutta explicit theo hệ số (alpha, beta, gamma):

    - alpha: (s,)  với alpha[i] = c_{i+1}
    - beta : (s,s) với beta[i,j] = a_{i+1,j+1}, chỉ dùng j < i
    - gamma: (s,)  với gamma[i] = b_{i+1}

    Công thức:
      k_i = f(t + alpha_i*h, y + h * sum_{j=1..i-1} beta_{i,j} * k_j)
      y_{n+1} = y_n + h * sum_{i=1..s} gamma_i * k_i
    """
    y, is_scalar = _to_array(y)
    alpha, beta, gamma = _validate_explicit_rk_coeffs(alpha, beta, gamma)

    s = len(alpha)
    k = []
    for i in range(s):
        y_stage = y.copy()
        if i > 0:
            acc = 0.0
            for j in range(i):
                acc = acc + beta[i, j] * k[j]
            y_stage = y_stage + h * acc
        k_i = f(t + alpha[i] * h, y_stage)
        k.append(k_i)

    y_next = y + h * sum(gamma[i] * k[i] for i in range(s))
    return y_next[0] if is_scalar else y_next


def solve_rk_custom(f, t_span, y0, h, alpha, beta, gamma):
    """
    Giải y' = f(t,y) bằng công thức explicit RK do bạn tự định nghĩa từ (alpha,beta,gamma).
    """
    t0, T = t_span
    y0_arr, is_scalar = _to_array(y0)

    t_values = np.arange(t0, T + h / 1000.0, h, dtype=float)
    n_steps = len(t_values)

    y_values = np.zeros((n_steps, len(y0_arr)), dtype=float)
    y_values[0] = y0_arr

    for i in range(n_steps - 1):
        t = t_values[i]
        y = y_values[i]
        y_values[i + 1] = rk_step_custom(f, t, y, h, alpha, beta, gamma)

    if is_scalar:
        return t_values, y_values[:, 0]
    return t_values, y_values


# ==========================================
# 1c. "BUILDER" TỰ SINH (beta, gamma) TỪ s & alpha
#     - Mục tiêu: bạn nhập số nấc s và các alpha_i (c_i)
#       => tự ra công thức RK explicit bậc cao nhất có thể.
#
#     Lưu ý quan trọng:
#     - Với s=2: có họ RK2 bậc 2 tham số hóa bởi alpha2 (c2).
#     - Với s=3: có thể dựng RK3 bậc 3 từ alpha2, alpha3 với giả thiết
#       cấu trúc đơn giản: a21=c2, a32=c3, còn lại 0.
#     - Với s>=4: chỉ nhập alpha là CHƯA ĐỦ để xác định một RK bậc 4 duy nhất
#       (cần thêm ràng buộc/cấu trúc hoặc nhập thêm tham số).
# ==========================================

def build_explicit_rk_from_alpha(s, alpha):
    """
    Sinh (alpha, beta, gamma) cho RK explicit từ đầu vào:
    - s: số nấc
    - alpha: danh sách/array các alpha_i (c_i)

    Trả về bộ (alpha, beta, gamma) dùng trực tiếp với rk_step_custom/solve_rk_custom.
    """
    alpha = np.asarray(alpha, dtype=float).ravel()
    if len(alpha) != s:
        raise ValueError("Độ dài alpha phải đúng bằng số nấc s.")

    if s == 2:
        # RK2 bậc 2 tổng quát (explicit 2-stage):
        #   c = [0, c2]
        #   a21 = c2
        #   Điều kiện bậc 2:
        #      (1) b1 + b2 = 1
        #      (2) b2*c2   = 1/2
        #   ⇒ b2 = 1/(2c2), b1 = 1 - b2
        c1, c2 = alpha
        if abs(c1) > 1e-12:
            raise ValueError("RK2: alpha1 nên bằng 0.")
        if c2 == 0:
            raise ValueError("RK2: alpha2 (c2) phải khác 0.")

        beta = np.zeros((2, 2), dtype=float)
        beta[1, 0] = c2
        gamma2 = 1.0 / (2.0 * c2)
        gamma1 = 1.0 - gamma2
        gamma = np.array([gamma1, gamma2], dtype=float)

        if PRINT_CUSTOM_DERIVATION:
            print("\n[RK2 tùy chỉnh] DẪN CÔNG THỨC TỪ alpha:")
            print(f"  c = [0, c2] với c2 = {c2}")
            print("  Chọn a21 = c2, các a_ij khác = 0.")
            print("  Điều kiện bậc 2:")
            print("    (1) b1 + b2 = 1")
            print("    (2) b2*c2   = 1/2")
            print("  ⇒ b2 = 1/(2c2), b1 = 1 - b2")
            print(f"  ⇒ b1 = {gamma1}, b2 = {gamma2}")

        return alpha, beta, gamma

    if s == 3:
        # RK3 bậc 3 với cấu trúc đơn giản:
        #   c = [0, c2, c3]
        #   a21 = c2
        #   a31 = 0, a32 = c3
        #
        # Điều kiện bậc 3 (Butcher):
        #   (1) sum b = 1
        #   (2) sum b c = 1/2
        #   (3) sum b c^2 = 1/3
        #   (4) b^T A c = 1/6
        #
        # Với A như trên: A c có thành phần thứ 3 là a32*c2 = c3*c2
        # => b3*c2*c3 = 1/6 => b3 = 1/(6 c2 c3)
        c1, c2, c3 = alpha
        if abs(c1) > 1e-12:
            raise ValueError("RK3: alpha1 nên bằng 0.")
        if c2 == 0 or c3 == 0:
            raise ValueError("RK3: alpha2, alpha3 phải khác 0.")
        if c2 == c3:
            raise ValueError("RK3: nên chọn alpha2 != alpha3 để hệ xác định ổn định hơn.")

        beta = np.zeros((3, 3), dtype=float)
        beta[1, 0] = c2
        beta[2, 1] = c3  # a32

        b3 = 1.0 / (6.0 * c2 * c3)

        # Giải b1, b2 từ:
        #   b1 + b2 + b3 = 1
        #   b2*c2 + b3*c3 = 1/2
        #   b2*c2^2 + b3*c3^2 = 1/3
        #
        # (từ 2) => b2 = (1/2 - b3*c3)/c2
        b2 = (0.5 - b3 * c3) / c2
        # kiểm tra thêm điều kiện (3), nếu lệch quá lớn thì báo
        lhs3 = b2 * c2**2 + b3 * c3**2
        if abs(lhs3 - (1.0 / 3.0)) > 5e-8:
            raise ValueError(
                "RK3: bộ alpha này không thỏa điều kiện bậc 3 với cấu trúc builder hiện tại. "
                "Hãy thử alpha khác (ví dụ [0, 1/2, 1]) hoặc yêu cầu mình mở rộng builder."
            )

        b1 = 1.0 - b2 - b3
        gamma = np.array([b1, b2, b3], dtype=float)

        if PRINT_CUSTOM_DERIVATION:
            print("\n[RK3 tùy chỉnh] DẪN CÔNG THỨC TỪ alpha:")
            print(f"  c = [0, c2, c3] với c2 = {c2}, c3 = {c3}")
            print("  Chọn ma trận A:")
            print("    a21 = c2, a31 = 0, a32 = c3, các a_ij khác = 0.")
            print("  Điều kiện bậc 3:")
            print("    (1) b1 + b2 + b3 = 1")
            print("    (2) b1*c1 + b2*c2 + b3*c3 = 1/2")
            print("    (3) b1*c1^2 + b2*c2^2 + b3*c3^2 = 1/3")
            print("    (4) b^T A c = 1/6")
            print("  Với cấu trúc A, (4) ⇒ b3*c2*c3 = 1/6 ⇒ b3 = 1/(6 c2 c3).")
            print("  Thế vào (2) ⇒ b2 = (1/2 - b3*c3)/c2.")
            print("  Thế tiếp vào (1) ⇒ b1 = 1 - b2 - b3.")
            print(f"  ⇒ b1 = {b1}, b2 = {b2}, b3 = {b3}")

        return alpha, beta, gamma

    raise NotImplementedError(
        "Hiện builder chỉ hỗ trợ s=2 (RK2 bậc 2) và s=3 (RK3 bậc 3) từ alpha. "
        "Nếu bạn muốn s=4 (bậc 4) theo alpha tuỳ ý, cần nhập thêm ràng buộc/đặt cấu trúc A "
        "hoặc cho thêm tham số để giải hệ điều kiện Butcher."
    )


# ==========================================
# 2. MIỀN ỔN ĐỊNH TUYỆT ĐỐI CHO CÁC CÔNG THỨC RK
#    (BÀI TEST y' = λy, z = h·λ, y_{n+1} = R(z)·y_n)
# ==========================================

def stability_function_rk(method, z, alpha=None, beta=None, gamma=None):
    """
    Tính hàm khuếch đại R(z) cho bài test y' = λy với z = h·λ.
    - method: "rk1", "rk2", "rk3", "rk4" hoặc "custom"
    - z: số thực hoặc mảng numpy (trục thực)

    Ý tưởng: đặt h = 1, f(t,y) = z·y, y0 = 1 ⇒ 1 bước RK cho kết quả y1 = R(z).
    """
    zs = np.asarray(z, dtype=float)
    scalar_input = np.isscalar(z)
    if scalar_input:
        zs = zs.reshape(1)

    R = np.zeros_like(zs, dtype=float)
    for i, zi in enumerate(zs):
        def f_test(t, y):
            return zi * y

        y0 = np.array([1.0])
        if method in ("rk1", "rk2", "rk3", "rk4"):
            y1 = rk_step(f_test, 0.0, y0, 1.0, method=method)
        elif method == "custom":
            if alpha is None or beta is None or gamma is None:
                raise ValueError("Cần truyền alpha, beta, gamma cho method='custom'.")
            y1 = rk_step_custom(f_test, 0.0, y0, 1.0, alpha, beta, gamma)
        else:
            raise ValueError("method phải là 'rk1', 'rk2', 'rk3', 'rk4' hoặc 'custom'.")

        R[i] = y1[0]

    return R[0] if scalar_input else R


def analyze_real_stability_interval_for_method(
    method,
    z_min=-10.0,
    z_max=5.0,
    n_points=2001,
    alpha=None,
    beta=None,
    gamma=None,
):
    """
    Tìm gần đúng các đoạn trên trục thực mà |R(z)| ≤ 1 (miền ổn định tuyệt đối trên R).
    Dùng được cho cả RK cổ điển (rk1..rk4) và RK tùy chỉnh (method='custom', kèm alpha,beta,gamma).
    """
    zs = np.linspace(z_min, z_max, n_points)
    Rs = stability_function_rk(method, zs, alpha=alpha, beta=beta, gamma=gamma)
    stable = np.abs(Rs) <= 1.0 + 1e-10

    intervals = []
    in_seg = False
    start = None
    for i, ok in enumerate(stable):
        if ok and not in_seg:
            in_seg = True
            start = zs[i]
        elif not ok and in_seg:
            in_seg = False
            end = zs[i - 1]
            intervals.append((start, end))
    if in_seg:
        intervals.append((start, zs[-1]))

    return intervals


def print_stability_analysis_detailed(method, alpha=None, beta=None, gamma=None, intervals=None):
    """
    In đầy đủ phần phân tích miền ổn định tuyệt đối như một phần bài thi:
    - Phương pháp (bài test y' = λy)
    - Công thức tính R(z) cụ thể
    - Các bước xác định miền ổn định
    - Kết quả cuối cùng
    """
    print("\n" + "=" * 70)
    print("PHÂN TÍCH MIỀN ỔN ĐỊNH TUYỆT ĐỐI")
    print("=" * 70)
    
    # 1. Phương pháp
    print("\n1. PHƯƠNG PHÁP:")
    print("   Xét bài toán test: y' = λy với λ ∈ ℝ (hoặc λ ∈ ℂ)")
    print("   Đặt z = h·λ, trong đó h là bước thời gian.")
    print("   Áp dụng công thức Runge-Kutta cho bài toán này, ta có:")
    print("     y_{n+1} = R(z) · y_n")
    print("   trong đó R(z) là hàm khuếch đại (amplification function).")
    print("   Miền ổn định tuyệt đối là tập các z sao cho |R(z)| ≤ 1.")
    
    # 2. Công thức R(z) cụ thể
    print("\n2. CÔNG THỨC TÍNH R(z):")
    
    if method == "rk1":
        print("   Với Euler hiện (RK1):")
        print("     y_{n+1} = y_n + h·f(t_n, y_n)")
        print("     Với f(t,y) = λy:")
        print("     y_{n+1} = y_n + h·λ·y_n = (1 + hλ)·y_n")
        print("     ⇒ R(z) = 1 + z")
        print("   Điều kiện ổn định: |R(z)| ≤ 1 ⇔ |1 + z| ≤ 1")
        print("   Trên trục thực: |1 + z| ≤ 1 ⇔ -2 ≤ z ≤ 0")
        
    elif method == "rk2":
        print("   Với RK2 (midpoint):")
        print("     k1 = f(t_n, y_n) = λ·y_n")
        print("     k2 = f(t_n + h/2, y_n + h/2·k1) = λ·(y_n + h/2·λ·y_n) = λ·(1 + z/2)·y_n")
        print("     y_{n+1} = y_n + h·k2 = y_n + h·λ·(1 + z/2)·y_n")
        print("     ⇒ y_{n+1} = (1 + z + z²/2)·y_n")
        print("     ⇒ R(z) = 1 + z + z²/2")
        print("   Điều kiện ổn định: |R(z)| ≤ 1")
        
    elif method == "rk3":
        print("   Với RK3 cổ điển:")
        print("     k1 = λ·y_n")
        print("     k2 = λ·(y_n + h/2·k1) = λ·(1 + z/2)·y_n")
        print("     k3 = λ·(y_n - h·k1 + 2h·k2) = λ·(1 - z + 2z·(1 + z/2))·y_n")
        print("     y_{n+1} = y_n + h/6·(k1 + 4k2 + k3)")
        print("     Sau khi tính toán:")
        print("     ⇒ R(z) = 1 + z + z²/2 + z³/6")
        print("   Điều kiện ổn định: |R(z)| ≤ 1")
        
    elif method == "rk4":
        print("   Với RK4 cổ điển:")
        print("     k1 = λ·y_n")
        print("     k2 = λ·(y_n + h/2·k1) = λ·(1 + z/2)·y_n")
        print("     k3 = λ·(y_n + h/2·k2) = λ·(1 + z/2 + z²/4)·y_n")
        print("     k4 = λ·(y_n + h·k3) = λ·(1 + z + z²/2 + z³/4)·y_n")
        print("     y_{n+1} = y_n + h/6·(k1 + 2k2 + 2k3 + k4)")
        print("     Sau khi tính toán:")
        print("     ⇒ R(z) = 1 + z + z²/2 + z³/6 + z⁴/24")
        print("   Điều kiện ổn định: |R(z)| ≤ 1")
        
    elif method == "custom":
        if alpha is None or beta is None or gamma is None:
            print("   [Cần thông tin về công thức RK tùy chỉnh]")
        else:
            s = len(alpha)
            print(f"   Với RK{s} tùy chỉnh:")
            print(f"     alpha (c) = {alpha}")
            print(f"     gamma (b) = {gamma}")
            print("     Áp dụng công thức:")
            for i in range(s):
                if i == 0:
                    print(f"       k{i+1} = f(t_n, Y_n) = λ·y_n")
                else:
                    terms = []
                    for j in range(i):
                        if abs(beta[i, j]) > 1e-10:
                            if abs(beta[i, j] - 1.0) < 1e-10:
                                terms.append(f"h·k{j+1}")
                            else:
                                terms.append(f"{beta[i, j]}·h·k{j+1}")
                    if terms:
                        sum_str = " + ".join(terms)
                        print(f"       k{i+1} = f(t_n + {alpha[i]}·h, Y_n + {sum_str})")
                    else:
                        print(f"       k{i+1} = f(t_n + {alpha[i]}·h, Y_n)")
            
            gamma_str = " + ".join([f"{gamma[i]}·k{i+1}" for i in range(s)])
            print(f"       y_{{n+1}} = y_n + h·({gamma_str})")
            print("     Sau khi tính toán với f(t,y) = λy, ta được R(z) là đa thức bậc", s)
            print("   Điều kiện ổn định: |R(z)| ≤ 1")
    
    # 3. Các bước xác định miền ổn định
    print("\n3. XÁC ĐỊNH MIỀN ỔN ĐỊNH TRÊN TRỤC THỰC:")
    print("   Ta khảo sát |R(z)| ≤ 1 với z ∈ ℝ.")
    print("   Cách làm:")
    print("     - Tính R(z) tại nhiều điểm z trên trục thực")
    print("     - Kiểm tra điều kiện |R(z)| ≤ 1")
    print("     - Xác định các đoạn [a, b] mà |R(z)| ≤ 1")
    
    # 4. Kết quả
    print("\n4. KẾT QUẢ:")
    if intervals:
        print("   Miền ổn định tuyệt đối trên trục thực (z ∈ ℝ):")
        for a, b in intervals:
            print(f"     |R(z)| ≤ 1 với z ∈ [{a:.4f}, {b:.4f}]")
        
        # Tính độ dài tổng
        total_length = sum(b - a for a, b in intervals)
        print(f"\n   Tổng độ dài các đoạn ổn định: {total_length:.4f}")
    else:
        print("   Không tìm được đoạn ổn định hữu hạn trên trục thực")
        print("   trong khoảng khảo sát [-10, 5].")
        print("   (Có thể phương pháp là A-stable hoặc miền ổn định nằm ngoài khoảng này)")
    
    print("=" * 70)


# ==========================================
# 3. HÀM MAIN ĐƠN GIẢN CHO BÀI THI
# ==========================================

def main():
    """
    Quy trình dùng trong bài thi:
      1) Sửa ode_function, T_SPAN, Y0, H, LABELS, METHOD, CUSTOM_* ở đầu file.
      2) Chạy file: python RK.py
      3) Đọc bảng giá trị (có thể chép vào bài làm).
    """
    f = ode_function
    t_span = T_SPAN
    y0 = Y0
    h = H

    # Chọn solver
    method = METHOD.lower()
    alpha = beta = gamma = None

    if method in ("rk1", "rk2", "rk3", "rk4"):
        t_values, y_values = solve_rk(
            f,
            t_span,
            y0,
            h,
            method=method,
            explain_steps=EXPLAIN_STEPS,
            explain_first_n=EXPLAIN_FIRST_N,
        )
        method_name = method.upper()
    elif method == "rk_custom_2" or method == "rk_custom_3":
        s = CUSTOM_S
        alpha_input = CUSTOM_ALPHA
        alpha, beta, gamma = build_explicit_rk_from_alpha(s, alpha_input)

        # In Butcher tableau và công thức tổng quát cho RK tùy chỉnh
        print("\nCÔNG THỨC RK TÙY CHỈNH (Butcher tableau):")
        print(f"  alpha (c)  = {alpha}")
        print("  beta (A)   =")
        print(beta)
        print(f"  gamma (b)  = {gamma}")

        if s == 2:
            c2 = alpha[1]
            a21 = beta[1, 0]
            b1, b2 = gamma
            print("\nDạng cụ thể (RK2):")
            print("  k1 = f(t_n, Y_n)")
            print(f"  k2 = f(t_n + c2·h, Y_n + a21·h·k1) với c2 = {c2}, a21 = {a21}")
            print(f"  Y_{{n+1}} = Y_n + h·({b1}·k1 + {b2}·k2)")
        elif s == 3:
            c2, c3 = alpha[1], alpha[2]
            a21, a31, a32 = beta[1, 0], beta[2, 0], beta[2, 1]
            b1, b2, b3 = gamma
            print("\nDạng cụ thể (RK3):")
            print("  k1 = f(t_n, Y_n)")
            print(f"  k2 = f(t_n + c2·h, Y_n + a21·h·k1) với c2 = {c2}, a21 = {a21}")
            print(f"  k3 = f(t_n + c3·h, Y_n + a31·h·k1 + a32·h·k2) với c3 = {c3}, a31 = {a31}, a32 = {a32}")
            print(f"  Y_{{n+1}} = Y_n + h·({b1}·k1 + {b2}·k2 + {b3}·k3)")

        t_values, y_values = solve_rk_custom(f, t_span, y0, h, alpha, beta, gamma)
        method_name = f"RK{CUSTOM_S}-tùy-chỉnh"
    else:
        raise ValueError("METHOD không hợp lệ. Hãy chọn: rk1, rk2, rk3, rk4, rk_custom_2, rk_custom_3.")

    # In thông tin tóm tắt bài toán
    print("\n" + "=" * 60)
    print("GIẢI PHƯƠNG TRÌNH VI PHÂN BẰNG CÔNG THỨC RUNGE–KUTTA")
    print("=" * 60)
    print(f"Phương pháp: {method_name}")
    print(f"Khoảng thời gian: t ∈ [{t_span[0]}, {t_span[1]}],  bước h = {h}")
    print(f"Giá trị đầu: Y0 = {y0}")
    print("=" * 60)

    # Nếu bật, phân tích luôn miền ổn định tuyệt đối trên trục thực (bài test y' = λy)
    if ANALYZE_STABILITY:
        if method in ("rk1", "rk2", "rk3", "rk4"):
            key = method  # dùng trực tiếp
            intervals = analyze_real_stability_interval_for_method(key, z_min=-10, z_max=5, n_points=5001)
            print_stability_analysis_detailed(key, intervals=intervals)
        else:
            key = "custom"
            intervals = analyze_real_stability_interval_for_method(
                key, z_min=-10, z_max=5, n_points=5001, alpha=alpha, beta=beta, gamma=gamma
            )
            print_stability_analysis_detailed(key, alpha=alpha, beta=beta, gamma=gamma, intervals=intervals)

    # In bảng kết quả (vài bước đầu + vài bước cuối)
    n, m = y_values.shape if y_values.ndim == 2 else (len(y_values), 1)
    if m == 1 and not np.isscalar(Y0):
        m = len(Y0)

    # Tiêu đề cột
    if not LABELS or len(LABELS) != m:
        labels = [f"y{i+1}" for i in range(m)]
    else:
        labels = LABELS

    header = f"{'n':>4} | {'t_n':>10} | " + " | ".join(f"{name:>12}" for name in labels)
    print(header)
    print("-" * len(header))

    def print_row(idx):
        t = t_values[idx]
        y = y_values[idx]
        if np.isscalar(y):
            y_list = [y]
        else:
            y_list = list(y)
        row = f"{idx:4d} | {t:10.4f} | " + " | ".join(f"{val:12.6f}" for val in y_list)
        print(row)

    max_print = 10  # số dòng tối đa in chi tiết
    if len(t_values) <= max_print:
        for i in range(len(t_values)):
            print_row(i)
    else:
        # In vài bước đầu và vài bước cuối (tránh bảng quá dài)
        for i in range(5):
            print_row(i)
        print("   ...")
        for i in range(len(t_values) - 5, len(t_values)):
            print_row(i)

    # In giá trị cuối cùng (hay dùng để so sánh/ghi kết quả)
    print("\nGiá trị cuối cùng:")
    print(f"t_N = {t_values[-1]:.4f}")
    print(f"Y_N = {y_values[-1]}")

    # Vẽ đồ thị nghiệm theo t (nếu bật)
    if PLOT_RESULT:
        plt.figure(figsize=(10, 5))
        if y_values.ndim == 1:
            plt.plot(t_values, y_values, label=labels[0] if labels else "y")
        else:
            for i in range(m):
                plt.plot(t_values, y_values[:, i], label=labels[i])
        plt.xlabel("t")
        plt.ylabel("Giá trị nghiệm")
        plt.title(f"Đồ thị nghiệm theo thời gian (phương pháp {method_name})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
