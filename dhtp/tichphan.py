from sympy import sympify, symbols
from sympy import *
import math

def print_exam_cheatsheet():
    """In nhanh công thức + bước làm quan trọng (dùng cho bài thi)."""
    print("\n" + "="*60)
    print("TÓM TẮT CÔNG THỨC & CÁC BƯỚC LÀM (DÙNG CHO BÀI THI)")
    print("="*60)
    print("\n1) DỮ LIỆU & KÝ HIỆU")
    print("  - Cho ∫[a,b] f(x) dx")
    print("  - Chia [a,b] thành N khoảng: x_i = a + i*h, i=0..N;  h = (b-a)/N")
    print("  - f_i = f(x_i)")
    print("  - Điểm giữa: x_{i+1/2} = a + (i+1/2)h, i=0..N-1;  f_{i+1/2} = f(x_{i+1/2})")

    print("\n2) CÔNG THỨC ĐIỂM GIỮA (COMPOSITE MIDPOINT)")
    print("  - I ≈ h * Σ_{i=0}^{N-1} f(x_{i+1/2})")
    print("  - Sai số: |E| ≤ (M2/24) * (b-a) * h^2,  M2 = max_{[a,b]} |f''(x)|")

    print("\n3) CÔNG THỨC HÌNH THANG (COMPOSITE TRAPEZOIDAL)")
    print("  - I ≈ h/2 * [f0 + 2(f1+...+f_{N-1}) + fN]")
    print("  - Sai số: |E| ≤ (M2/12) * (b-a) * h^2")

    print("\n4) CÔNG THỨC SIMPSON (COMPOSITE SIMPSON)")
    print("  - Điều kiện: N chẵn")
    print("  - I ≈ h/3 * [f0 + 4(f1+f3+...+f_{N-1}) + 2(f2+f4+...+f_{N-2}) + fN]")
    print("  - Sai số: |E| ≤ (M4/180) * (b-a) * h^4,  M4 = max |f^(4)(x)|")

    print("\n5) NEWTON–COTES TỔNG HỢP (COMPOSITE NEWTON–COTES) BẬC m")
    print("  - Ý nghĩa: m là bậc công thức nhỏ (m=4/5/6 trong đề); mỗi panel dùng (m+1) hệ số")
    print("  - Điều kiện ghép: N phải chia hết cho m  ⇒  số panel = N/m")
    print("  - Độ dài 1 panel: L = m*h")
    print("  - Nếu H0..Hm là hệ trọng số cho bậc m (đề cho), thì với panel p:")
    print("      I_p ≈ L * Σ_{j=0..m} H_j * f(x_{p*m + j})")
    print("    Tổng: I ≈ Σ_{p=0..N/m-1} I_p")
    print("  - Ghi chú quan trọng:")
    print("      + Hệ trọng số chỉ phụ thuộc m (không phụ thuộc f)")
    print("      + N là tổng số khoảng chia của bài (x0→x40 ⇒ N=40), khác với m")

    print("\n6) SAI SỐ NEWTON–COTES TỔNG HỢP (CHẶN TRÊN)")
    print("  - Với bậc m chẵn: dùng đạo hàm bậc (m+2)")
    print("  - Với bậc m lẻ : dùng đạo hàm bậc (m+1)")
    print("  - Ý tưởng: Sai số toàn đoạn ≤ (số panel) * (sai số 1 panel)")
    print("  - Khi chạy code, chương trình sẽ in M = max|f^(k)(x)| và chặn trên |E| tương ứng.")

    print("\n7) TRÌNH BÀY BÀI THI (GỢI Ý BƯỚC LÀM)")
    print("  - B1: Xác định N (từ đề: b = x_N), tính h = (b-a)/N")
    print("  - B2: Lập các điểm x_i (và x_{i+1/2} nếu dùng midpoint)")
    print("  - B3: Tính các giá trị f(x_i) (và f(x_{i+1/2}))")
    print("  - B4: Thế vào công thức (midpoint / hình thang / Simpson)")
    print("  - B5: Newton–Cotes bậc m: kiểm tra N % m == 0, rồi ghép theo từng panel")
    print("  - B6: Ước lượng sai số bằng công thức chặn trên (dùng M2/M4 hoặc M_k)")


# Công thức điểm giữa (Midpoint Rule)

def midpoint(A_mid):
    """ Tính gần đúng tích phân xác định bằng công thức điểm giữa """
    print("\n" + "="*60)
    print("PHƯƠNG PHÁP ĐIỂM GIỮA (MIDPOINT RULE)")
    print("="*60)
    print(f"\nCông thức: ∫[a,b] f(x)dx ≈ h * [f(x₀.₅) + f(x₁.₅) + f(x₂.₅) + ... + f(xₙ₋₁.₅)]")
    print(f"Với h = (b-a)/n = ({b}-{a})/{n} = {h}")
    print(f"\nCác điểm giữa:")
    sum_mid = 0
    for i in range(n):
        xi_mid = a + (i + 0.5)*h
        print(f"  x{i+0.5} = {a} + ({i} + 0.5)*{h} = {xi_mid:.6f}  →  f(x{i+0.5}) = {A_mid[i]:.6f}")
        sum_mid += A_mid[i]
    
    result = h * sum_mid
    print(f"\nCác bước tính:")
    print(f"  Tổng các giá trị tại điểm giữa: Σf(xᵢ.₅) = {sum_mid:.6f}")
    print(f"  Kết quả = h * Σf(xᵢ.₅) = {h:.6f} * {sum_mid:.6f} = {result:.6f}")
    print(f"\nTích phân bằng công thức điểm giữa      : {result:.10f}")

def midpoint_error():
    """ Sai số của công thức điểm giữa """
    m = max(f, 2)
    error = m/24*(b-a)*(h**2)
    print(f"\nCông thức sai số: |E| ≤ M₂/24 * (b-a) * h²")
    print(f"  Với M₂ = max|f''(x)| trên [{a}, {b}] = {m:.10f}")
    print(f"  |E| ≤ {m:.10f}/24 * ({b}-{a}) * ({h})²")
    print(f"     ≤ {m/24:.10f} * {b-a:.6f} * {h**2:.10f}")
    print(f"     ≤ {error:.10f}")
    print(f"Sai số công thức điểm giữa              : {error:.10f}")

# Công thức điểm giữa (Midpoint Rule)

def midpoint(A_mid):
    """ Tính gần đúng tích phân xác định bằng công thức điểm giữa """
    print("\n" + "="*60)
    print("PHƯƠNG PHÁP ĐIỂM GIỮA (MIDPOINT RULE)")
    print("="*60)
    print(f"\nCông thức: ∫[a,b] f(x)dx ≈ h * [f(x₀.₅) + f(x₁.₅) + f(x₂.₅) + ... + f(xₙ₋₁.₅)]")
    print(f"Với h = (b-a)/n = ({b}-{a})/{n} = {h}")
    print(f"\nCác điểm giữa:")
    sum_mid = 0
    for i in range(n):
        xi_mid = a + (i + 0.5)*h
        print(f"  x{i+0.5} = {a} + ({i} + 0.5)*{h} = {xi_mid:.6f}  →  f(x{i+0.5}) = {A_mid[i]:.6f}")
        sum_mid += A_mid[i]
    
    result = h * sum_mid
    print(f"\nCác bước tính:")
    print(f"  Tổng các giá trị tại điểm giữa: Σf(xᵢ.₅) = {sum_mid:.6f}")
    print(f"  Kết quả = h * Σf(xᵢ.₅) = {h:.6f} * {sum_mid:.6f} = {result:.6f}")
    print(f"\nTích phân bằng công thức điểm giữa      : {result:.10f}")

def midpoint_error():
    """ Sai số của công thức điểm giữa """
    m = max(f, 2)
    error = m/24*(b-a)*(h**2)
    print(f"\nCông thức sai số: |E| ≤ M₂/24 * (b-a) * h²")
    print(f"  Với M₂ = max|f''(x)| trên [{a}, {b}] = {m:.10f}")
    print(f"  |E| ≤ {m:.10f}/24 * ({b}-{a}) * ({h})²")
    print(f"     ≤ {m/24:.10f} * {b-a:.6f} * {h**2:.10f}")
    print(f"     ≤ {error:.10f}")
    print(f"Sai số công thức điểm giữa              : {error:.10f}")

# Công thức hình thang

def trapezoidal(A):
    """ Tính gần đúng tích phân xác định bằng công thức hình thang """
    print("\n" + "="*60)
    print("PHƯƠNG PHÁP HÌNH THANG (TRAPEZOIDAL RULE)")
    print("="*60)
    print(f"\nCông thức: ∫[a,b] f(x)dx ≈ h/2 * [f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]")
    print(f"Với h = (b-a)/n = ({b}-{a})/{n} = {h}")
    print(f"\nCác điểm chia:")
    for i in range(n+1):
        xi = a + i*h
        print(f"  x{i} = {a} + {i}*{h} = {xi:.6f}  →  f(x{i}) = {A[i]:.6f}")
    
    print(f"\nCác bước tính:")
    print(f"  Tổng đầu cuối: f(x₀) + f(xₙ) = {A[0]:.6f} + {A[n]:.6f} = {A[0] + A[n]:.6f}")
    sum_middle = sum(A[1:n])
    print(f"  Tổng các điểm giữa (nhân 2): 2 * ({' + '.join([f'{A[i]:.6f}' for i in range(1, n)])}) = 2 * {sum_middle:.6f} = {2*sum_middle:.6f}")
    
    trape = 1/2*(A[0] + A[n])
    for i in range(1, n):
        trape = trape + A[i]
    print(f"\n  Kết quả = h/2 * [f(x₀) + 2Σf(xᵢ) + f(xₙ)]")
    print(f"           = {h}/2 * [{A[0]:.6f} + 2*{sum_middle:.6f} + {A[n]:.6f}]")
    print(f"           = {h/2:.6f} * {trape:.6f}")
    print(f"           = {trape*h:.6f}")
    print(f"\nTích phân bằng công thức hình thang      : {trape*h:.10f}")

def trapezoidal_error():
    """ Sai số của công thức hình thang """
    m = max(f, 2)
    error = m/12*(b-a)*(h**2)
    print(f"\nCông thức sai số: |E| ≤ M₂/12 * (b-a) * h²")
    print(f"  Với M₂ = max|f''(x)| trên [{a}, {b}] = {m:.10f}")
    print(f"  |E| ≤ {m:.10f}/12 * ({b}-{a}) * ({h})²")
    print(f"     ≤ {m/12:.10f} * {b-a:.6f} * {h**2:.10f}")
    print(f"     ≤ {error:.10f}")
    print(f"Sai số công thức hình thang              : {error:.10f}")

def trpezoidal_intervals():
    """ Số khoảng chia để thỏa mãn sai số cho trước trong công thức hình thang"""
    print("\n" + "="*60)
    print("TÍNH SỐ KHOẢNG CHIA CẦN THIẾT - CÔNG THỨC HÌNH THANG")
    print("="*60)
    print(f"\nYêu cầu: |E| ≤ ε = {eps}")
    print(f"Khoảng tích phân: [{a}, {b}]")
    
    print(f"Đặt f(x) = {f}")
    
    g = Derivative(f, (x, 2), evaluate=True)
    print(f"\nĐạo hàm cấp 2: f''(x) = {g}")
    print(f"  Dạng rút gọn: f''(x) = {simplify(g)}")
    
    m = max(f, 2)
    print(f"\nCông thức: n ≥ √[M₂ * (b-a)³ / (12 * ε)]")
    print(f"  Với M₂ = max|f''(x)| trên [{a}, {b}] = {m:.10f}")
    
    n_calc = math.floor((abs(((m*(b-a)**3)*(1/12)*(1/eps))))**(1/2))+1
    print(f"\nCác bước tính:")
    print(f"  M₂ * (b-a)³ = {m:.10f} * ({b}-{a})³ = {m:.10f} * {(b-a)**3:.6f} = {m*(b-a)**3:.10f}")
    print(f"  12 * ε = 12 * {eps} = {12*eps:.10f}")
    print(f"  M₂ * (b-a)³ / (12 * ε) = {m*(b-a)**3:.10f} / {12*eps:.10f} = {m*(b-a)**3/(12*eps):.10f}")
    print(f"  √[...] = {math.sqrt(m*(b-a)**3/(12*eps)):.6f}")
    print(f"  n ≥ {math.sqrt(m*(b-a)**3/(12*eps)):.6f}")
    print(f"  n = {n_calc} (làm tròn lên)")
    print(f"\nMax của f''(x) là: {m:.10f}")
    print(f"Số khoảng chia cần thiết (Hình thang)    : {n_calc}")


# Công thức Simpson

def simpson(A):
    """ Tính gần đúng tích phân xác định bằng công thức Simpson """
    print("\n" + "="*60)
    print("PHƯƠNG PHÁP SIMPSON")
    print("="*60)
    print(f"\nCông thức: ∫[a,b] f(x)dx ≈ h/3 * [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 4f(xₙ₋₁) + f(xₙ)]")
    print(f"Với h = (b-a)/n = ({b}-{a})/{n} = {h}")
    print(f"Lưu ý: n phải là số chẵn (n = {n} {'✓' if n%2==0 else '✗'})")
    
    print(f"\nCác điểm chia:")
    for i in range(n+1):
        xi = a + i*h
        coeff = 1 if (i == 0 or i == n) else (4 if i%2 == 1 else 2)
        print(f"  x{i} = {xi:.6f}  →  f(x{i}) = {A[i]:.6f}  (hệ số: {coeff})")
    
    simp_odd = 0
    simp_even = 0
    odd_indices = [i for i in range(1, n, 2)]
    even_indices = [i for i in range(2, n, 2)]
    
    for i in range(1, n, 2):
        simp_odd += A[i]
    for i in range(2, n, 2):
        simp_even += A[i]
    
    print(f"\nCác bước tính:")
    print(f"  Tổng đầu cuối: f(x₀) + f(xₙ) = {A[0]:.6f} + {A[n]:.6f} = {A[0] + A[n]:.6f}")
    if odd_indices:
        odd_str = ' + '.join([f'f(x{i})' for i in odd_indices])
        print(f"  Tổng điểm lẻ (nhân 4): 4 * ({odd_str}) = 4 * ({' + '.join([f'{A[i]:.6f}' for i in odd_indices])}) = 4 * {simp_odd:.6f} = {4*simp_odd:.6f}")
    if even_indices:
        even_str = ' + '.join([f'f(x{i})' for i in even_indices])
        print(f"  Tổng điểm chẵn (nhân 2): 2 * ({even_str}) = 2 * ({' + '.join([f'{A[i]:.6f}' for i in even_indices])}) = 2 * {simp_even:.6f} = {2*simp_even:.6f}")
    
    simp = h/3*(A[0]+A[n]) + h/3*4*simp_odd + h/3*2*simp_even
    print(f"\n  Kết quả = h/3 * [f(x₀) + 4Σf(xᵢ lẻ) + 2Σf(xᵢ chẵn) + f(xₙ)]")
    print(f"           = {h}/3 * [{A[0]:.6f} + 4*{simp_odd:.6f} + 2*{simp_even:.6f} + {A[n]:.6f}]")
    print(f"           = {h/3:.6f} * [{A[0] + A[n]:.6f} + {4*simp_odd:.6f} + {2*simp_even:.6f}]")
    print(f"           = {simp:.6f}")
    print(f"\nTích phân bằng công thức Simpson         : {simp:.10f}")

def simpson_error():
    """ Sai số của công thức Simpson """
    m = max(f, 4)
    error = m/180*(b-a)*(h**4)
    print(f"\nCông thức sai số: |E| ≤ M₄/180 * (b-a) * h⁴")
    print(f"  Với M₄ = max|f⁽⁴⁾(x)| trên [{a}, {b}] = {m:.10f}")
    print(f"  |E| ≤ {m:.10f}/180 * ({b}-{a}) * ({h})⁴")
    print(f"     ≤ {m/180:.10f} * {b-a:.6f} * {h**4:.10f}")
    print(f"     ≤ {error:.10f}")
    print(f"Sai số công thức Simpson                 : {error:.10f}")

def simpson_intervals():
    """ Số khoảng chia để thỏa mãn sai số cho trước trong công thức Simpson"""
    print("\n" + "="*60)
    print("TÍNH SỐ KHOẢNG CHIA CẦN THIẾT - CÔNG THỨC SIMPSON")
    print("="*60)
    print(f"\nYêu cầu: |E| ≤ ε = {eps}")
    print(f"Khoảng tích phân: [{a}, {b}]")
    print(f"Lưu ý: n phải là số chẵn")
    
    print(f"Đặt f(x) = {f}")
    
    g = Derivative(f, (x, 4), evaluate=True)
    print(f"\nĐạo hàm cấp 4: f⁽⁴⁾(x) = {g}")
    print(f"  Dạng rút gọn: f⁽⁴⁾(x) = {simplify(g)}")
    
    m = max(f, 4)
    print(f"\nCông thức: n ≥ ⁴√[M₄ * (b-a)⁵ / (180 * ε)]")
    print(f"  Với M₄ = max|f⁽⁴⁾(x)| trên [{a}, {b}] = {m:.10f}")
    
    n = math.floor((abs((m*(b-a)**5)*(1/180)*(1/eps)))**(1/4))+1
    print(f"\nCác bước tính:")
    print(f"  M₄ * (b-a)⁵ = {m:.10f} * ({b}-{a})⁵ = {m:.10f} * {(b-a)**5:.6f} = {m*(b-a)**5:.10f}")
    print(f"  180 * ε = 180 * {eps} = {180*eps:.10f}")
    print(f"  M₄ * (b-a)⁵ / (180 * ε) = {m*(b-a)**5:.10f} / {180*eps:.10f} = {m*(b-a)**5/(180*eps):.10f}")
    print(f"  ⁴√[...] = {(m*(b-a)**5/(180*eps))**(1/4):.6f}")
    print(f"  n ≥ {(m*(b-a)**5/(180*eps))**(1/4):.6f}")
    print(f"  n = {n} (làm tròn lên)")
    
    if n % 2 == 1:
        n_final = n+1
        print(f"  Vì n = {n} là số lẻ, cần làm tròn lên số chẵn gần nhất")
        print(f"  n = {n_final} (số chẵn)")
    else:
        n_final = n+2
        print(f"  Vì n = {n} là số chẵn nhưng cần đảm bảo đủ lớn")
        print(f"  n = {n_final} (làm tròn lên)")
    
    print(f"\nMax của f⁽⁴⁾(x) là: {m:.10f}")
    print(f"Số khoảng chia cần thiết (Simpson)       : {n_final}")


# Công thức Newton - Cotes

def multiply_horner(A, i) -> list:
    """ Nhân một đa thức với (x-i) """
    A.append(0)
    for j in range(len(A)-1,0,-1):
        A[j] = A[j] - A[j - 1] * i
    return A

def devide_horner(A, i) -> list:
    """ Chia một đa thức với (x-i) """
    X = A.copy()
    X.pop()
    for j in range(1, len(X)):
        X[j] = i*X[j-1] + X[j]
    return X

def poly_integral(A, a, b) -> float:
    I = 0
    """ Tính tích phân xác định của đa thức """
    for j in range(0, len(A)):
        if (A[j] == 0):
            continue
        else:
            A[j] = A[j]/(len(A)-j)     
        I = I + A[j]*(b**(len(A)-j)-a**(len(A)-j))
    return I

def cotez_coef(i, verbose=False) -> float:
    """ Tính hệ số Cotez H_i"""
    # D là tích các đa thức (t-j), j từ 0 đến n, nhưng để tiết kiệm thời gian, tính 
    # nó một lần duy nhất ở dưới.
    # Bước 1: Chia D cho (t-i) để được đa thức X = D/(t-i)
    X = devide_horner(D, i)
    
    # Bước 2: Tính tích phân của đa thức X từ 0 đến n
    integral_value = poly_integral(X, 0, n)
    
    # Bước 3: Tính hệ số Cotes theo công thức
    # H_i = (1/n) * [(-1)^(n-i) / (i! * (n-i)!)] * ∫₀ⁿ [D/(t-i)] dt
    coeff = ((-1)**(n-i))/(math.factorial(i)*math.factorial(n-i))
    h = (1/n) * coeff * integral_value
    
    if verbose:
        print(f"    Bước 1: Chia D cho (t-{i}) → đa thức X")
        print(f"    Bước 2: Tính ∫₀ⁿ X dt = {integral_value:.10f}")
        print(f"    Bước 3: Hệ số = (1/{n}) * [(-1)^({n}-{i}) / ({i}! * {n-i}!)] * {integral_value:.10f}")
        print(f"           = (1/{n}) * [{coeff:.10f}] * {integral_value:.10f}")
        print(f"           = {h:.10f}")
    
    return h

def show_standard_weights():
    """ Hiển thị bảng hệ trọng số chuẩn cho n = 1-6 """
    print("\n" + "="*60)
    print("BẢNG HỆ TRỌNG SỐ NEWTON-COTES CHUẨN")
    print("="*60)
    print("\nCác hệ trọng số đã được chuẩn hóa cho các giá trị n phổ biến:")
    
    standard_weights = {
        1: [0.5, 0.5],  # Hình thang
        2: [1/6, 4/6, 1/6],  # Simpson
        3: [1/8, 3/8, 3/8, 1/8],  # Simpson 3/8
        4: [7/90, 32/90, 12/90, 32/90, 7/90],
        5: [19/288, 25/96, 25/144, 25/144, 25/96, 19/288],
        6: [41/840, 9/35, 9/280, 34/105, 9/280, 9/35, 41/840]
    }
    
    for n_val, weights in standard_weights.items():
        print(f"\nn = {n_val}:")
        for i, w in enumerate(weights):
            print(f"  H{i} = {w:.10f} ({w})")
    
    print("\n⚠️ LƯU Ý QUAN TRỌNG:")
    print("  - Các hệ số này đã được chuẩn hóa, có thể dùng trực tiếp.")
    print("  - Các hệ trọng số CHỈ phụ thuộc vào n, KHÔNG phụ thuộc vào hàm f(x)!")
    print("  - Với cùng một n, các hệ số giống nhau cho MỌI hàm f(x).")
    return standard_weights

def calculate_weights_for_n(n_val):
    """ Tính hệ trọng số Newton-Cotes cho một giá trị n bất kỳ """
    print("\n" + "="*60)
    print(f"TÍNH HỆ TRỌNG SỐ NEWTON-COTES CHO n = {n_val}")
    print("="*60)
    
    print(f"\n" + "="*60)
    print("MỐI QUAN HỆ GIỮA HỆ TRỌNG SỐ VÀ SỐ KHOẢNG CHIA")
    print("="*60)
    print(f"\n✓ HỆ TRỌNG SỐ PHỤ THUỘC HOÀN TOÀN VÀO SỐ KHOẢNG CHIA n:")
    print(f"  • Với mỗi giá trị n khác nhau → có một bộ hệ trọng số khác nhau")
    print(f"  • n = 1 → có 2 hệ số: H₀, H₁")
    print(f"  • n = 2 → có 3 hệ số: H₀, H₁, H₂")
    print(f"  • n = 3 → có 4 hệ số: H₀, H₁, H₂, H₃")
    print(f"  • n = {n_val} → có {n_val+1} hệ số: H₀, H₁, ..., H{n_val}")
    print(f"\n✓ CÔNG THỨC TÍNH CHỨA n:")
    print(f"  Hᵢ = (1/n) * [(-1)ⁿ⁻ⁱ / (i! * (n-i)!)] * ∫₀ⁿ [Πⱼ≠ᵢ(t-j)/(i-j)] dt")
    print(f"  → Tất cả các thành phần đều phụ thuộc vào n:")
    print(f"    • (1/n) - chia cho n")
    print(f"    • (-1)ⁿ⁻ⁱ - lũy thừa của n")
    print(f"    • (n-i)! - giai thừa của n")
    print(f"    • ∫₀ⁿ - tích phân từ 0 đến n")
    print(f"    • D = (t-0)(t-1)...(t-n) - tích n+1 nhân tử")
    print(f"\n✓ KHÔNG PHỤ THUỘC VÀO HÀM f(x):")
    print(f"  → Với cùng một n, các hệ trọng số giống nhau cho MỌI hàm f(x)")
    print(f"  → Bạn có thể tính một lần và dùng lại cho nhiều hàm khác nhau")
    print(f"\nBước 1: Tính tích D = (t-0)(t-1)...(t-{n_val})")
    
    # Tính tích D = (t-0)(t-1)...(t-n)
    D_temp = [1]
    for i in range(0, n_val+1):
        multiply_horner(D_temp, i)
    
    print(f"  Đã tính xong đa thức D (bậc {n_val+1})")
    
    # Tính các hệ số
    Hs = []
    print(f"\nBước 2: Tính các hệ trọng số H₀, H₁, ..., H{n_val}:")
    
    for i in range(0, n_val+1):
        # Chia D cho (t-i)
        X = devide_horner(D_temp.copy(), i)
        
        # Tính tích phân
        integral_value = poly_integral(X, 0, n_val)
        
        # Tính hệ số
        coeff = ((-1)**(n_val-i))/(math.factorial(i)*math.factorial(n_val-i))
        h = (1/n_val) * coeff * integral_value
        Hs.append(h)
        
        print(f"\n  Tính H{i}:")
        print(f"    Chia D cho (t-{i}) → đa thức X")
        print(f"    Tính ∫₀^{n_val} X dt = {integral_value:.10f}")
        print(f"    Hệ số = (1/{n_val}) * [(-1)^({n_val}-{i}) / ({i}! * {n_val-i}!)] * {integral_value:.10f}")
        print(f"           = (1/{n_val}) * [{coeff:.10f}] * {integral_value:.10f}")
        print(f"           = {h:.10f}")
    
    # Hiển thị kết quả
    print(f"\n" + "="*60)
    print(f"KẾT QUẢ: HỆ TRỌNG SỐ CHO n = {n_val}")
    print("="*60)
    print(f"\n✓ Các hệ trọng số này ÁP DỤNG CHO MỌI HÀM f(x) với n = {n_val}")
    print(f"\nCác hệ trọng số:")
    for i, h in enumerate(Hs):
        print(f"  H{i} = {h:.15f}")
    
    print(f"\nDạng mảng Python:")
    print(f"  [{', '.join([f'{h:.15f}' for h in Hs])}]")
    
    print(f"\nDạng rút gọn (làm tròn 10 chữ số):")
    print(f"  [{', '.join([f'{round(h, 10)}' for h in Hs])}]")
    
    # Kiểm tra tổng
    total = sum(Hs)
    print(f"\nTổng các hệ số = {total:.15f}", end="")
    if abs(total - 1.0) < 0.0001:
        print(" ✓ (đúng, tổng = 1)")
    else:
        print(f" (lưu ý: tổng nên bằng 1)")
    
    # Thử tìm dạng phân số gần đúng
    print(f"\nDạng phân số gần đúng:")
    try:
        from fractions import Fraction
        for i, h in enumerate(Hs):
            frac = Fraction(h).limit_denominator(10000)
            print(f"  H{i} ≈ {frac} ({float(frac):.10f})")
    except:
        pass
    
    return Hs

def input_manual_weights():
    """ Nhập hệ trọng số thủ công từ người dùng - cho phép nhập số lượng tùy ý """
    print(f"\nNhập hệ trọng số thủ công cho n = {n}:")
    print(f"Bạn có thể nhập số lượng hệ số tùy ý (không nhất thiết phải đúng {n+1} hệ số)")
    print(f"Ví dụ với n=2 (Simpson): 0.166667 0.666667 0.166667")
    print(f"Hoặc chỉ nhập một số hệ số: 0.166667 0.666667")
    print(f"Hoặc dạng phân số: 1/6 4/6 1/6")
    
    while True:
        try:
            weights_input = input(f"\nNhập các hệ số cách nhau bởi dấu cách (có thể nhập ít hơn hoặc nhiều hơn {n+1}): ").strip()
            if not weights_input:
                print("Lỗi: Không được để trống.")
                continue
                
            weights_str = weights_input.split()
            Hs = []
            
            for w_str in weights_str:
                # Kiểm tra nếu là phân số
                if '/' in w_str:
                    parts = w_str.split('/')
                    if len(parts) == 2:
                        num = float(parts[0])
                        den = float(parts[1])
                        Hs.append(num / den)
                    else:
                        raise ValueError(f"Phân số không hợp lệ: {w_str}")
                else:
                    Hs.append(float(w_str))
            
            num_weights = len(Hs)
            print(f"\nBạn đã nhập {num_weights} hệ số.")
            
            # Xử lý các trường hợp
            if num_weights < n+1:
                print(f"  → Đã nhập {num_weights} hệ số, thiếu {n+1 - num_weights} hệ số so với yêu cầu ({n+1} hệ số)")
                print(f"\nBạn muốn:")
                print(f"  (1) Tự động tính phần còn lại ({n+1 - num_weights} hệ số)")
                print(f"  (2) Chỉ dùng {num_weights} hệ số đã nhập (sẽ cập nhật n = {num_weights-1})")
                print(f"  (3) Nhập lại")
                choice = input("Chọn (1/2/3): ").strip()
                
                if choice == '3':
                    continue
                elif choice == '2':
                    # Chỉ dùng số lượng đã nhập
                    print(f"\nSẽ sử dụng {num_weights} hệ số đã nhập (n = {num_weights-1})")
                    print(f"Hệ trọng số đã nhập:")
                    for i in range(num_weights):
                        print(f"  H{i} = {Hs[i]:.10f}")
                    
                    # Kiểm tra tổng
                    total = sum(Hs)
                    if abs(total - 1.0) > 0.01:
                        print(f"\nCảnh báo: Tổng các hệ số = {total:.10f} (thường nên bằng 1)")
                    
                    confirm = input("\nXác nhận các hệ số này đúng? (y/n): ").strip().lower()
                    if confirm == 'y':
                        return Hs
                    else:
                        continue
                else:
                    # Tự động tính phần còn lại
                    print(f"\nĐang tính tự động {n+1 - num_weights} hệ số còn lại...")
                    # Tính tích D
                    D_temp = [1]
                    for i in range(0, n+1):
                        multiply_horner(D_temp, i)
                    
                    # Tính các hệ số còn lại
                    for i in range(num_weights, n+1):
                        X = devide_horner(D_temp.copy(), i)
                        integral_value = poly_integral(X, 0, n)
                        coeff = ((-1)**(n-i))/(math.factorial(i)*math.factorial(n-i))
                        h = (1/n) * coeff * integral_value
                        Hs.append(h)
                        print(f"  Đã tính H{i} = {h:.10f}")
                    
                    print(f"\nĐã có đủ {n+1} hệ số:")
                    
            elif num_weights > n+1:
                print(f"  → Đã nhập {num_weights} hệ số, nhiều hơn yêu cầu ({n+1} hệ số)")
                print(f"  → Sẽ chỉ lấy {n+1} hệ số đầu tiên, bỏ qua {num_weights - (n+1)} hệ số cuối")
                Hs = Hs[:n+1]
                print(f"\nCác hệ số sẽ sử dụng:")
            else:
                print(f"  → Đã nhập đúng {n+1} hệ số ✓")
            
            # Hiển thị kết quả
            print(f"\nHệ trọng số sẽ sử dụng ({len(Hs)} hệ số):")
            for i in range(len(Hs)):
                print(f"  H{i} = {Hs[i]:.10f}")
            
            # Kiểm tra tổng các hệ số (nên bằng 1)
            total = sum(Hs)
            if abs(total - 1.0) > 0.01:
                print(f"\nCảnh báo: Tổng các hệ số = {total:.10f} (thường nên bằng 1)")
            else:
                print(f"\n✓ Tổng các hệ số = {total:.10f} (đúng)")
            
            confirm = input("\nXác nhận các hệ số này đúng? (y/n): ").strip().lower()
            if confirm == 'y':
                return Hs
            else:
                print("Vui lòng nhập lại.")
        except ValueError as e:
            print(f"Lỗi: {e}. Vui lòng nhập các số hợp lệ.")
        except Exception as e:
            print(f"Lỗi: {e}")
    
def newton_cotez(Hs_manual=None, use_composite=True) -> float:
    """ Tính gần đúng tích phân xác định bằng công thức Newton - Cotez 
    
    Args:
        Hs_manual: Hệ trọng số cho trước (nếu None thì tự tính)
        use_composite: True nếu dùng Composite khi n > 6, False để tính Naive (không khuyến khích)
    """
    global n, h, A, D
    
    # Nếu n > 6 và không có hệ số cho trước, chỉ dùng Composite
    if n > 6 and Hs_manual is None and use_composite:
        print("\n" + "="*60)
        print("PHƯƠNG PHÁP NEWTON-COTES TỔNG HỢP (COMPOSITE)")
        print("="*60)
        print(f"\n⚠️ LƯU Ý: n = {n} > 6")
        print(f"  → Không tính hệ trọng số Naive cho n lớn (không ổn định)")
        print(f"  → Chỉ sử dụng phương pháp Composite (ghép nhiều công thức nhỏ)")
        
        # Hiển thị bảng hệ trọng số chuẩn
        standard_weights = show_standard_weights()
        
        print(f"\nChọn bậc công thức nhỏ để ghép:")
        print(f"  (1) n = 1 (Hình thang)")
        print(f"  (2) n = 2 (Simpson)")
        print(f"  (3) n = 3 (Simpson 3/8)")
        print(f"  (4) n = 4 (Boole)")
        print(f"  (5) n = 5")
        print(f"  (6) n = 6")
        print(f"  (7) Tự động chọn (ưu tiên Simpson n=2 nếu n chẵn)")
        
        choice = input("Chọn (1-7, mặc định là 7): ").strip()
        
        if choice == '1':
            n_small = 1
            Hs_small = standard_weights[1]
        elif choice == '2':
            n_small = 2
            Hs_small = standard_weights[2]
        elif choice == '3':
            n_small = 3
            Hs_small = standard_weights[3]
        elif choice == '4':
            n_small = 4
            Hs_small = standard_weights[4]
        elif choice == '5':
            n_small = 5
            Hs_small = standard_weights[5]
        elif choice == '6':
            n_small = 6
            Hs_small = standard_weights[6]
        else:
            # Tự động chọn
            if n % 2 == 0:
                n_small = 2  # Ưu tiên Simpson
                Hs_small = standard_weights[2]
                print(f"\nTự động chọn: n_small = 2 (Simpson) vì n = {n} chẵn")
            elif n % 3 == 0:
                n_small = 3  # Simpson 3/8
                Hs_small = standard_weights[3]
                print(f"\nTự động chọn: n_small = 3 (Simpson 3/8) vì n = {n} chia hết cho 3")
            else:
                n_small = 1  # Hình thang
                Hs_small = standard_weights[1]
                print(f"\nTự động chọn: n_small = 1 (Hình thang)")
        
        return composite_newton_cotes(n_small, Hs_small, auto_mode=True)
    
    print("\n" + "="*60)
    print("PHƯƠNG PHÁP NEWTON-COTES")
    print("="*60)
    
    E = 0
    Hs = [1]*(n+1)
    actual_n = n  # Số khoảng chia thực tế sẽ dùng
    
    if Hs_manual is not None:
        # Sử dụng hệ trọng số đã nhập thủ công hoặc từ bảng chuẩn
        Hs = Hs_manual
        num_weights = len(Hs)
        
        # Kiểm tra số lượng hệ trọng số
        if num_weights < n+1:
            print(f"\n⚠️ CẢNH BÁO: Bạn đã cung cấp {num_weights} hệ trọng số, nhưng n = {n} (cần {n+1} hệ số)")
            print(f"  → Sẽ sử dụng {num_weights} hệ số đầu tiên, cập nhật n = {num_weights-1}")
            actual_n = num_weights - 1
            # Cập nhật h và A tương ứng
            h = (b-a)/actual_n
            A = [f.subs(x, a+i*h) for i in range(actual_n+1)]
            print(f"  → Bước chia mới: h = ({b}-{a})/{actual_n} = {h:.10f}")
            print(f"  → Số điểm mới: {actual_n+1} điểm")
        elif num_weights > n+1:
            print(f"\n⚠️ CẢNH BÁO: Bạn đã cung cấp {num_weights} hệ trọng số, nhiều hơn yêu cầu ({n+1})")
            print(f"  → Sẽ chỉ sử dụng {n+1} hệ số đầu tiên")
            Hs = Hs[:n+1]
            num_weights = n+1
        
        print(f"\nCông thức: ∫[a,b] f(x)dx ≈ (b-a) * Σᵢ₌₀ⁿ Hᵢ * f(xᵢ)")
        print(f"Với:")
        print(f"  - Số khoảng chia: n = {actual_n}")
        print(f"  - Bước chia: h = (b-a)/n = ({b}-{a})/{actual_n} = {h:.10f}")
        print(f"  - Hệ trọng số: H₀, H₁, ..., H{actual_n} ({num_weights} hệ số)")
        print(f"\n⚠️ LƯU Ý:")
        print(f"  Các hệ trọng số CHỈ phụ thuộc vào n = {actual_n}, KHÔNG phụ thuộc vào hàm f(x)!")
        print(f"  → Với cùng n, các hệ số giống nhau cho mọi hàm f(x).")
        
        print(f"\nSử dụng hệ trọng số đã cung cấp:")
        for i in range(num_weights):
            print(f"  H{i} = {Hs[i]:.10f}")
    else:
        # Tính tự động
        print(f"\nCông thức: ∫[a,b] f(x)dx ≈ (b-a) * Σᵢ₌₀ⁿ Hᵢ * f(xᵢ)")
        print(f"Với:")
        print(f"  - Số khoảng chia: n = {n}")
        print(f"  - Bước chia: h = (b-a)/n = ({b}-{a})/{n} = {h:.10f}")
        print(f"  - Hệ trọng số: H₀, H₁, ..., Hₙ (các hệ số Cotes)")
        print(f"\n⚠️ LƯU Ý:")
        print(f"  Các hệ trọng số CHỈ phụ thuộc vào n = {n}, KHÔNG phụ thuộc vào hàm f(x)!")
        print(f"  → Với cùng n, các hệ số giống nhau cho mọi hàm f(x).")
        
        print(f"\nHệ trọng số Cotes Hᵢ được tính từ công thức:")
        print(f"  Hᵢ = (1/n) * [(-1)ⁿ⁻ⁱ / (i! * (n-i)!)] * ∫₀ⁿ [Πⱼ≠ᵢ(t-j)/(i-j)] dt")
        
        print(f"\nCÁC BƯỚC TÍNH HỆ TRỌNG SỐ TỰ ĐỘNG:")
        print(f"Bước 0: Tính tích D = (t-0)(t-1)...(t-{n})")
        print(f"  (Đã tính sẵn trước đó, lưu trong mảng D)")
        
        print(f"\nTính các hệ số Cotes Hᵢ:")
        for i in range(0, n+1):
            print(f"\n  Tính H{i}:")
            Hs[i] = cotez_coef(i, verbose=(n <= 4))  # Chỉ in chi tiết nếu n <= 4
            print(f"  H{i} = {Hs[i]:.10f}")
    
    # Tính tích phân với hệ trọng số đã có
    num_weights = len(Hs)
    for i in range(num_weights):
        E = E + Hs[i]*A[i]
    
    print(f"\nCác điểm chia và giá trị hàm:")
    for i in range(num_weights):
        xi = a + i*h
        print(f"  x{i} = {xi:.6f}  →  f(x{i}) = {A[i]:.6f}  (hệ số H{i} = {Hs[i]:.10f})")
    
    print(f"\nCác bước tính:")
    print(f"  Σᵢ Hᵢ * f(xᵢ) = ", end="")
    terms = [f"{Hs[i]:.6f} * {A[i]:.6f}" for i in range(num_weights)]
    if num_weights > 10:
        # Nếu quá nhiều, chỉ hiển thị một số
        print(f"{terms[0]} + ... + {terms[-1]} ({num_weights} số hạng)")
    else:
        print(" + ".join(terms))
    print(f"                = ", end="")
    products = [Hs[i]*A[i] for i in range(num_weights)]
    if num_weights > 10:
        print(f"{products[0]:.6f} + ... + {products[-1]:.6f} = {E:.10f}")
    else:
        print(" + ".join([f"{p:.6f}" for p in products]))
        print(f"                = {E:.10f}")
    print(f"\n  Kết quả = (b-a) * Σᵢ Hᵢ * f(xᵢ)")
    print(f"           = ({b}-{a}) * {E:.10f}")
    print(f"           = {b-a:.6f} * {E:.10f}")
    print(f"           = {E*(b-a):.10f}")
    print(f'\nHệ số Cotes ứng với n = {actual_n}                : {[round(h, 10) for h in Hs]}')
    print(f'Tích phân bằng công thức Newton - Cotez  : {E*(b-a):.10f}')

def newton_cotez_error() -> float: 
    """ Sai số của công thức Newton - Cotez"""
    g = Derivative(f, (x, n), evaluate=True)
    print(f"\nĐạo hàm cấp {n}: f⁽{n}⁾(x) = {g}")
    print(f"  Dạng rút gọn: f⁽{n}⁾(x) = {simplify(g)}")
    
    if (n % 2 == 0):
        g2 = Derivative(f, (x, n+2), evaluate=True)
        print(f"\nĐạo hàm cấp {n+2}: f⁽{n+2}⁾(x) = {g2}")
        print(f"  Dạng rút gọn: f⁽{n+2}⁾(x) = {simplify(g2)}")
        
        D1 = D.copy()
        multiply_horner(D1, n+1)
        m2 = max(f, n+2)
        error = abs(float(m2)*poly_integral(D1, 0, n)*(h**(n+3))/math.factorial(n+2))
        print(f"\nCông thức sai số (n chẵn): |E| ≤ Mₙ₊₂ * I(Dₙ₊₁) * hⁿ⁺³ / (n+2)!")
        print(f"  Với Mₙ₊₂ = max|f⁽{n+2}⁾(x)| trên [{a}, {b}] = {m2:.10f}")
        print(f"  I(Dₙ₊₁) = {poly_integral(D1, 0, n):.10f}")
        print(f"  hⁿ⁺³ = {h**(n+3):.10f}")
        print(f"  (n+2)! = {math.factorial(n+2)}")
        print(f"  |E| ≤ {error:.10f}")
        print(f"Sai số công thức Newton - Cotez          : {error:.10f}")
    else:
        g1 = Derivative(f, (x, n+1), evaluate=True)
        print(f"\nĐạo hàm cấp {n+1}: f⁽{n+1}⁾(x) = {g1}")
        print(f"  Dạng rút gọn: f⁽{n+1}⁾(x) = {simplify(g1)}")
        
        m1 = max(f, n+1)
        error = abs(float(m1)*poly_integral(D, 0, n)*(h**(n+2))/math.factorial(n+1))
        print(f"\nCông thức sai số (n lẻ): |E| ≤ Mₙ₊₁ * I(Dₙ) * hⁿ⁺² / (n+1)!")
        print(f"  Với Mₙ₊₁ = max|f⁽{n+1}⁾(x)| trên [{a}, {b}] = {m1:.10f}")
        print(f"  I(Dₙ) = {poly_integral(D, 0, n):.10f}")
        print(f"  hⁿ⁺² = {h**(n+2):.10f}")
        print(f"  (n+1)! = {math.factorial(n+1)}")
        print(f"  |E| ≤ {error:.10f}")
        print(f"Sai số công thức Newton - Cotez          : {error:.10f}")


def max(fx, i):
    """ Tìm maximum của đạo hàm cấp i của hàm f(x)"""
    g   = Derivative(fx,(x, i), evaluate=True)
    m1  = abs(maximum(g, x, Interval(a, b)))
    m2  = abs(minimum(g, x, Interval(a, b)))
    if m1 > m2:
        m = m1
    else:
        m = m2
    return m

def composite_newton_cotes(n_small, Hs_small, auto_mode=True):
    """ Tính tích phân bằng phương pháp Newton-Cotes tổng hợp (Composite)
    
    Args:
        n_small: Số khoảng chia nhỏ cho mỗi panel (1,2,3,4,5,6)
        Hs_small: Hệ trọng số cho công thức nhỏ (n_small+1 hệ số)
        auto_mode: True nếu tự động ghép, False nếu cho phép chọn
    """
    global n, a, b, h, f
    
    print("\n" + "="*60)
    print("PHƯƠNG PHÁP NEWTON-COTES TỔNG HỢP (COMPOSITE)")
    print("="*60)
    
    # Kiểm tra điều kiện: n phải chia hết cho n_small
    if n % n_small != 0:
        print(f"\n⚠️ CẢNH BÁO: n = {n} không chia hết cho n_small = {n_small}")
        print(f"  → Cần n chia hết cho n_small để áp dụng Composite Newton-Cotes")
        print(f"  → Sẽ điều chỉnh n = {n - (n % n_small)} (làm tròn xuống)")
        n = n - (n % n_small)
        h = (b - a) / n
    
    num_panels = n // n_small
    h_small = (b - a) / n  # Bước chia tổng thể
    h_panel = n_small * h_small  # Độ dài mỗi panel
    
    print(f"\nCông thức Composite: ∫[a,b] f(x)dx = Σᵢ ∫[xᵢ, xᵢ₊ₙₛₘₐₗₗ] f(x)dx")
    print(f"Với:")
    print(f"  - Tổng số khoảng chia: n = {n}")
    print(f"  - Bậc công thức nhỏ: n_small = {n_small}")
    print(f"  - Số panel: {num_panels}")
    print(f"  - Độ dài mỗi panel: h_panel = {h_panel:.10f}")
    print(f"  - Bước chia tổng thể: h = {h_small:.10f}")
    
    print(f"\nHệ trọng số cho công thức nhỏ (n={n_small}):")
    for i in range(n_small + 1):
        print(f"  H{i} = {Hs_small[i]:.10f}")
    
    # Tính tích phân bằng cách ghép
    total_result = 0
    print(f"\nTính tích phân cho từng panel:")
    
    for panel_idx in range(num_panels):
        # Điểm bắt đầu và kết thúc của panel
        a_panel = a + panel_idx * h_panel
        b_panel = a + (panel_idx + 1) * h_panel
        
        # Tính các điểm trong panel
        A_panel = [f.subs(x, a_panel + i * h_small) for i in range(n_small + 1)]
        
        # Tính tích phân cho panel này
        E_panel = 0
        for i in range(n_small + 1):
            E_panel += Hs_small[i] * A_panel[i]
        
        result_panel = h_panel * E_panel
        total_result += result_panel
        
        print(f"\n  Panel {panel_idx + 1}/{num_panels}: [{a_panel:.6f}, {b_panel:.6f}]")
        print(f"    Các điểm trong panel:")
        for i in range(n_small + 1):
            xi = a_panel + i * h_small
            print(f"      x{panel_idx*n_small + i} = {xi:.6f}  →  f(x{panel_idx*n_small + i}) = {A_panel[i]:.6f}")
        print(f"    Σᵢ Hᵢ * f(xᵢ) = {E_panel:.10f}")
        print(f"    Kết quả panel = h_panel * Σ = {h_panel:.6f} * {E_panel:.10f} = {result_panel:.10f}")
    
    print(f"\n" + "="*60)
    print(f"KẾT QUẢ TỔNG HỢP:")
    print(f"  Tổng tích phân = Σ (kết quả các panel)")
    print(f"                 = {total_result:.10f}")
    print(f"\nTích phân bằng công thức Newton-Cotes tổng hợp (Composite): {total_result:.10f}")
    
    return total_result

def newton_cotez_exam_mode(Hs_given):
    """ Tính tích phân bằng Newton-Cotes với hệ trọng số cho trước (dùng cho bài thi) """
    global n, a, b, h, A, f
    
    print("\n" + "="*60)
    print("PHƯƠNG PHÁP NEWTON-COTES (VỚI HỆ TRỌNG SỐ CHO TRƯỚC)")
    print("="*60)
    
    num_weights = len(Hs_given)
    actual_n = num_weights - 1
    
    # Kiểm tra nếu n lớn, dùng composite
    if n > 6:
        print(f"\n⚠️ LƯU Ý: n = {n} lớn hơn 6")
        print(f"  → Sẽ sử dụng phương pháp Composite Newton-Cotes")
        print(f"  → Ghép nhiều công thức nhỏ với n_small = {actual_n}")
        return composite_newton_cotes(actual_n, Hs_given, auto_mode=True)
    
    # Tính các điểm chia và giá trị hàm
    h = (b - a) / actual_n
    A = [f.subs(x, a + i*h) for i in range(num_weights)]
    
    print(f"\nCông thức: ∫[a,b] f(x)dx ≈ (b-a) * Σᵢ₌₀ⁿ Hᵢ * f(xᵢ)")
    print(f"Với:")
    print(f"  - Số khoảng chia: n = {actual_n}")
    print(f"  - Bước chia: h = (b-a)/n = ({b}-{a})/{actual_n} = {h:.10f}")
    print(f"  - Hệ trọng số đã cho: H₀, H₁, ..., H{actual_n}")
    
    print(f"\nHệ trọng số đã cung cấp:")
    for i in range(num_weights):
        print(f"  H{i} = {Hs_given[i]:.10f}")
    
    print(f"\nCác điểm chia và giá trị hàm:")
    for i in range(num_weights):
        xi = a + i*h
        print(f"  x{i} = {xi:.6f}  →  f(x{i}) = {A[i]:.6f}  (hệ số H{i} = {Hs_given[i]:.10f})")
    
    # Tính tích phân
    E = 0
    for i in range(num_weights):
        E += Hs_given[i] * A[i]
    
    result = (b - a) * E
    
    print(f"\nCác bước tính:")
    print(f"  Σᵢ Hᵢ * f(xᵢ) = ", end="")
    terms = [f"{Hs_given[i]:.6f} * {A[i]:.6f}" for i in range(num_weights)]
    if num_weights > 10:
        print(f"{terms[0]} + ... + {terms[-1]} ({num_weights} số hạng)")
    else:
        print(" + ".join(terms))
    print(f"                = ", end="")
    products = [Hs_given[i]*A[i] for i in range(num_weights)]
    if num_weights > 10:
        print(f"{products[0]:.6f} + ... + {products[-1]:.6f} = {E:.10f}")
    else:
        print(" + ".join([f"{p:.6f}" for p in products]))
        print(f"                = {E:.10f}")
    print(f"\n  Kết quả = (b-a) * Σᵢ Hᵢ * f(xᵢ)")
    print(f"           = ({b}-{a}) * {E:.10f}")
    print(f"           = {b-a:.6f} * {E:.10f}")
    print(f"           = {result:.10f}")
    print(f'\nTích phân bằng công thức Newton - Cotez  : {result:.10f}')
    
    return result

def _parse_weights_line(weights_input: str):
    """Parse weights like: 7/90 32/90 ... or decimals."""
    weights_str = weights_input.strip().split()
    Hs = []
    for w_str in weights_str:
        if '/' in w_str:
            parts = w_str.split('/')
            if len(parts) != 2:
                raise ValueError(f"Phân số không hợp lệ: {w_str}")
            num = float(parts[0])
            den = float(parts[1])
            if den == 0:
                raise ValueError(f"Mẫu số bằng 0: {w_str}")
            Hs.append(num / den)
        else:
            Hs.append(float(w_str))
    return Hs

def composite_newton_cotes_exam_mode(panel_n: int, Hs_panel: list, N_total: int):
    """Newton-Cotes tổng hợp (Composite): ghép công thức bậc panel_n theo từng panel."""
    global a, b, h, f, A

    if panel_n < 1:
        raise ValueError("panel_n phải >= 1")
    if len(Hs_panel) != panel_n + 1:
        raise ValueError(f"Cần {panel_n+1} hệ số cho Newton-Cotes bậc {panel_n}, nhưng nhận {len(Hs_panel)}")
    if N_total < panel_n:
        raise ValueError(f"N_total phải >= {panel_n} để áp dụng Newton-Cotes bậc {panel_n}")
    if N_total % panel_n != 0:
        raise ValueError(f"N_total = {N_total} phải chia hết cho {panel_n} để ghép Newton-Cotes bậc {panel_n}")

    h = (b - a) / N_total  # bước chia nhỏ
    num_panels = N_total // panel_n
    panel_len = panel_n * h

    print("\n" + "="*60)
    print(f"NEWTON-COTES TỔNG HỢP (COMPOSITE) - BẬC m = {panel_n}")
    print("="*60)

    # (A) Công thức + điều kiện
    print("\nA) CÔNG THỨC & ĐIỀU KIỆN ÁP DỤNG")
    print("  - Chia đều [a,b] thành N khoảng: x_i = a + i*h, i = 0..N")
    print(f"  - N = {N_total},  h = (b-a)/N = ({b}-{a})/{N_total} = {h:.10f}")
    print(f"  - Dùng Newton-Cotes bậc m = {panel_n} (m+1 điểm / 1 panel)")
    print(f"  - Điều kiện ghép: N % m = 0  ⇔  {N_total} % {panel_n} = 0  ✓")
    print(f"  - Số panel: P = N/m = {N_total}/{panel_n} = {num_panels}")
    print(f"  - Độ dài 1 panel: L = m*h = {panel_n}*{h:.10f} = {panel_len:.10f}")
    print("\n  Công thức tổng quát (Composite):")
    print("    I ≈ Σ_{p=0..P-1} [ L * Σ_{j=0..m} H_j * f(x_{p*m + j}) ]")
    print("    (Trong đó H0..Hm là hệ trọng số bậc m do đề cho)")

    # (B) Bước làm
    print("\nB) CÁC BƯỚC LÀM (THEO ĐÚNG TRÌNH TỰ BÀI THI)")
    print("  B1: Xác định N từ đề (vd x0→x40 ⇒ N=40), tính h = (b-a)/N")
    print("  B2: Chọn bậc m theo đề (m=4/5/6), kiểm tra N % m = 0")
    print("  B3: Lập các điểm x_i = a + i*h và tính f(x_i)")
    print("  B4: Chia thành P panel; với mỗi panel p, lấy các điểm x_{p*m},...,x_{p*m+m}")
    print("  B5: Tính I_p = L * Σ_{j=0..m} H_j f(x_{p*m+j}); rồi cộng I = Σ I_p")

    # Hệ số
    print(f"\nHệ trọng số (dùng cho từng panel):")
    for i in range(panel_n + 1):
        print(f"  H{i} = {Hs_panel[i]:.10f}")

    # (C) Minh hoạ panel đầu
    print(f"\nC) MINH HOẠ TÍNH 1 PANEL (panel đầu, p = 0)")
    total_sum = 0.0
    for j in range(panel_n + 1):
        xj = a + j*h
        fj = float(f.subs(x, xj))
        print(f"  x{j} = {xj:.6f}  f(x{j}) = {fj:.10f}  (H{j} = {Hs_panel[j]:.10f})")
        total_sum += Hs_panel[j] * fj
    first_panel_val = panel_len * total_sum
    print(f"  Σ H_j f(x_j) = {total_sum:.10f}")
    print(f"  I_0 = L * Σ H_j f(x_j) = {panel_len:.10f} * {total_sum:.10f} = {first_panel_val:.10f}")
    print("  (Các panel còn lại làm tương tự và cộng lại)")

    # Tính tổng (không in chi tiết tất cả panel để khỏi dài)
    result = 0.0
    for p in range(num_panels):
        base = p * panel_n
        s = 0.0
        for j in range(panel_n + 1):
            xpj = a + (base + j) * h
            s += Hs_panel[j] * float(f.subs(x, xpj))
        result += panel_len * s

    # (D) Kết quả
    print("\nD) KẾT QUẢ")
    print(f"  I (Newton-Cotes tổng hợp, bậc {panel_n}) ≈ {result:.10f}")

    # (E) Sai số (chặn trên)
    print("\nE) ĐÁNH GIÁ SAI SỐ (CHẶN TRÊN)")
    try:
        total_err, per_panel_err, order, M = composite_newton_cotes_error_bound(panel_n, N_total)
        print(f"  - Bậc m = {panel_n} ⇒ dùng đạo hàm bậc k = {order}")
        print(f"  - M = max|f^{order}(x)| trên [{a}, {b}] = {M:.10f}")
        print(f"  - Sai số 1 panel ≤ {per_panel_err:.10f}")
        print(f"  - Số panel P = {num_panels} ⇒ |E| ≤ P * (sai số 1 panel) ≤ {total_err:.10f}")
    except Exception as e:
        print(f"  Không ước lượng được sai số tự động: {e}")

    return result

def composite_newton_cotes_error_bound(panel_n: int, N_total: int):
    """Ước lượng sai số cho Newton-Cotes ghép (chặn trên kiểu tổng các panel)."""
    if N_total % panel_n != 0:
        raise ValueError(f"N_total = {N_total} phải chia hết cho {panel_n}")
    h_step = (b - a) / N_total
    num_panels = N_total // panel_n

    # Tính D = Π_{j=0..panel_n} (t-j)
    D_local = [1]
    for j in range(0, panel_n + 1):
        multiply_horner(D_local, j)

    if panel_n % 2 == 0:
        # n chẵn → dùng đạo hàm bậc n+2
        order = panel_n + 2
        M = max(f, order)
        D1 = D_local.copy()
        multiply_horner(D1, panel_n + 1)
        I = poly_integral(D1.copy(), 0, panel_n)
        per_panel = abs(float(M) * I * (h_step ** (panel_n + 3)) / math.factorial(panel_n + 2))
        total = num_panels * per_panel
        return total, per_panel, order, float(M)
    else:
        # n lẻ → dùng đạo hàm bậc n+1
        order = panel_n + 1
        M = max(f, order)
        I = poly_integral(D_local.copy(), 0, panel_n)
        per_panel = abs(float(M) * I * (h_step ** (panel_n + 2)) / math.factorial(panel_n + 1))
        total = num_panels * per_panel
        return total, per_panel, order, float(M)


def main():
    global n, a, b, f, h, x, D, A, eps
    x           = symbols('x')
    # Hàm toán học	Cách nhập
    # 2**x
    # x² + 1	x**2 + 1
    # sin(x)	sin(x)
    # eˣ	    exp(x)
    # ln(x) 	log(x)
    # log(x, 10)      # logarit cơ số 10
    # √x	    sqrt(x)
    # 1/(x+1)	1/(x+1)
    # x·eˣ	    x*exp(x)
    # sin²(x)	sin(x)**2
    while True:
        try:
            func = input('Nhập hàm f(x): ').strip()
            if not func:
                print("Lỗi: Không được để trống. Vui lòng nhập hàm f(x).")
                print("Ví dụ: x**2 + 1, sin(x), exp(x), log(x), sqrt(x)")
                continue
            
            # Kiểm tra nếu có ký tự đặc biệt không hợp lệ
            if '&' in func or '|' in func or func.startswith('"') or func.startswith("'"):
                print("⚠️ Cảnh báo: Có vẻ như bạn đã nhập nhầm lệnh hoặc đường dẫn.")
                print("Vui lòng nhập hàm toán học, ví dụ:")
                print("  - x**2 + 1")
                print("  - sin(x)")
                print("  - exp(x)")
                print("  - log(x)")
                print("  - sqrt(x)")
                confirm = input("Bạn có muốn nhập lại không? (y/n): ").strip().lower()
                if confirm != 'y':
                    return
                continue
            
            f = sympify(func)
            break
        except SympifyError as e:
            print(f"\n❌ Lỗi: Không thể phân tích hàm '{func}'")
            print(f"Chi tiết: {e}")
            print("\nVui lòng nhập lại hàm f(x) theo đúng cú pháp:")
            print("  - Phép nhân: dùng * (ví dụ: 2*x)")
            print("  - Lũy thừa: dùng ** (ví dụ: x**2)")
            print("  - Hàm lượng giác: sin(x), cos(x), tan(x)")
            print("  - Hàm mũ: exp(x) hoặc 2**x")
            print("  - Hàm logarit: log(x) hoặc log(x, 10)")
            print("  - Căn bậc hai: sqrt(x)")
            print("\nVí dụ hợp lệ:")
            print("  x**2 + 2*x + 1")
            print("  sin(x)*cos(x)")
            print("  exp(-x**2)")
            print("  log(x**2 + 1)")
            retry = input("\nNhập lại? (y/n): ").strip().lower()
            if retry != 'y':
                return
        except Exception as e:
            print(f"\n❌ Lỗi không mong đợi: {e}")
            print("Vui lòng kiểm tra lại hàm f(x) và nhập lại.")
            retry = input("Nhập lại? (y/n): ").strip().lower()
            if retry != 'y':
                return
    init_value  = input('Nhập khoảng lấy tích phân a, b (a < b) cách nhau bởi dấu cách: ')
    a, b        = [float(i) for i in init_value.split()]
    q           = int(input('Chọn bài toán bạn muốn giải quyết (Nhập số theo bài toán)''\n''(1) Tính tích phân (2) Tính số khoảng chia cần thiết (3) Tính hệ trọng số cho n (4) Chế độ bài thi (hệ trọng số cho trước): '))
    if q == 1:
        n       = int(input('Nhập số khoảng chia n: '))
        h       = (b-a)/n
    
        # Tính tích D = (t-0)(t-1)...(t-n) một lần duy nhất
        # D được lưu dưới dạng mảng hệ số đa thức
        D       = [1]  # Khởi tạo đa thức = 1
        for i in range(0, n+1):
            multiply_horner(D, i)                           # Nhân với (t-i) → D = D * (t-i)
        
        A       = [f.subs(x,a+i*h) for i in range(n+1)]     # Tạo mảng lưu giá trị hàm tại các mốc nội suy
        
        print("\n" + "="*60)
        print("THÔNG TIN ĐẦU VÀO")
        print("="*60)
        print(f"Hàm số: f(x) = {func}")
        print(f"Khoảng tích phân: [{a}, {b}]")
        print(f"Số khoảng chia: n = {n}")
        print(f"Bước chia: h = (b-a)/n = ({b}-{a})/{n} = {h:.10f}")
        print(f"Số điểm: {n+1} điểm")

        # Tính giá trị tại điểm giữa cho công thức điểm giữa
        A_mid = [f.subs(x, a + (i + 0.5)*h) for i in range(n)]
        
        if n % 2 == 0:
            midpoint(A_mid)
            midpoint_error()
            print()
            trapezoidal(A)
            trapezoidal_error()
            print()
            simpson(A)
            simpson_error()
            print()
            # Menu lựa chọn cho Newton-Cotes
            print(f"\n" + "="*60)
            print("TÍNH TÍCH PHÂN BẰNG CÔNG THỨC NEWTON-COTES")
            print("="*60)
            
            if n <= 6:
                standard_weights = show_standard_weights()
                print(f"\nBạn muốn:")
                print(f"  (1) Tính hệ trọng số tự động")
                if n in standard_weights:
                    print(f"  (2) Dùng bảng hệ trọng số chuẩn (n={n})")
                print(f"  (3) Nhập hệ trọng số thủ công")
                choice = input("Chọn (1/2/3, mặc định là 1): ").strip()
                
                if choice == '2' and n in standard_weights:
                    Hs_manual = standard_weights[n]
                    newton_cotez(Hs_manual)
                    newton_cotez_error()
                elif choice == '3':
                    Hs_manual = input_manual_weights()
                    newton_cotez(Hs_manual)
                    newton_cotez_error()
                else:
                    newton_cotez()
                    newton_cotez_error()
            else:
                # n > 6: Chỉ dùng Composite, không tính Naive
                print(f"\n⚠️ Số khoảng chia n = {n} > 6")
                print(f"  → Không tính hệ trọng số Naive (không ổn định với n lớn)")
                print(f"  → Chỉ sử dụng phương pháp Composite Newton-Cotes")
                print(f"\nBạn muốn:")
                print(f"  (1) Dùng Composite tự động (khuyến khích)")
                print(f"  (2) Nhập hệ trọng số cho công thức nhỏ và ghép")
                print(f"  (3) Bỏ qua Newton-Cotes")
                answer = input("Chọn (1/2/3): ").strip()
                
                if answer == '3':
                    return
                elif answer == '2':
                    print(f"\nNhập bậc công thức nhỏ (1-6):")
                    n_small = int(input("Nhập n_small (1-6): "))
                    if n_small < 1 or n_small > 6:
                        print("Lỗi: n_small phải từ 1 đến 6")
                        return
                    if n % n_small != 0:
                        print(f"⚠️ Cảnh báo: n = {n} không chia hết cho n_small = {n_small}")
                        print(f"  → Sẽ điều chỉnh n = {n - (n % n_small)}")
                        n = n - (n % n_small)
                        h = (b-a)/n
                        A = [f.subs(x, a+i*h) for i in range(n+1)]
                    
                    weights_input = input(f"Nhập {n_small+1} hệ trọng số cho n={n_small}: ")
                    Hs_small = _parse_weights_line(weights_input)
                    if len(Hs_small) != n_small + 1:
                        print(f"Lỗi: Cần {n_small+1} hệ số")
                        return
                    composite_newton_cotes(n_small, Hs_small, auto_mode=False)
                else:
                    # Tự động Composite
                    newton_cotez(use_composite=True)
        else:
            midpoint(A_mid)
            midpoint_error()
            print()
            trapezoidal(A)
            trapezoidal_error()
            print()
            # Menu lựa chọn cho Newton-Cotes
            print(f"\n" + "="*60)
            print("TÍNH TÍCH PHÂN BẰNG CÔNG THỨC NEWTON-COTES")
            print("="*60)
            
            if n <= 6:
                standard_weights = show_standard_weights()
                print(f"\nBạn muốn:")
                print(f"  (1) Tính hệ trọng số tự động")
                if n in standard_weights:
                    print(f"  (2) Dùng bảng hệ trọng số chuẩn (n={n})")
                print(f"  (3) Nhập hệ trọng số thủ công")
                choice = input("Chọn (1/2/3, mặc định là 1): ").strip()
                
                if choice == '2' and n in standard_weights:
                    Hs_manual = standard_weights[n]
                    newton_cotez(Hs_manual)
                    newton_cotez_error()
                elif choice == '3':
                    Hs_manual = input_manual_weights()
                    newton_cotez(Hs_manual)
                    newton_cotez_error()
                else:
                    newton_cotez()
                    newton_cotez_error()
            else:
                # n > 6: Chỉ dùng Composite, không tính Naive
                print(f"\n⚠️ Số khoảng chia n = {n} > 6")
                print(f"  → Không tính hệ trọng số Naive (không ổn định với n lớn)")
                print(f"  → Chỉ sử dụng phương pháp Composite Newton-Cotes")
                print(f"\nBạn muốn:")
                print(f"  (1) Dùng Composite tự động (khuyến khích)")
                print(f"  (2) Nhập hệ trọng số cho công thức nhỏ và ghép")
                print(f"  (3) Bỏ qua Newton-Cotes")
                answer = input("Chọn (1/2/3): ").strip()
                
                if answer == '3':
                    return
                elif answer == '2':
                    print(f"\nNhập bậc công thức nhỏ (1-6):")
                    n_small = int(input("Nhập n_small (1-6): "))
                    if n_small < 1 or n_small > 6:
                        print("Lỗi: n_small phải từ 1 đến 6")
                        return
                    if n % n_small != 0:
                        print(f"⚠️ Cảnh báo: n = {n} không chia hết cho n_small = {n_small}")
                        print(f"  → Sẽ điều chỉnh n = {n - (n % n_small)}")
                        n = n - (n % n_small)
                        h = (b-a)/n
                        A = [f.subs(x, a+i*h) for i in range(n+1)]
                    
                    weights_input = input(f"Nhập {n_small+1} hệ trọng số cho n={n_small}: ")
                    Hs_small = _parse_weights_line(weights_input)
                    if len(Hs_small) != n_small + 1:
                        print(f"Lỗi: Cần {n_small+1} hệ số")
                        return
                    composite_newton_cotes(n_small, Hs_small, auto_mode=False)
                else:
                    # Tự động Composite
                    newton_cotez(use_composite=True)

    if q == 2:
        eps     = float(input('Nhập epsilon: '))
        trpezoidal_intervals()
        simpson_intervals()
    
    if q == 4:
        # Chế độ bài thi: Newton-Cotes dùng hệ số bậc 4/5/6 (ghép theo N tổng của bài)
        print("\n" + "="*60)
        print("CHẾ ĐỘ BÀI THI: NEWTON-COTES BẬC 4/5/6 (DÙNG HỆ TRỌNG SỐ CHO TRƯỚC)")
        print("="*60)
        print_exam_cheatsheet()
        print("\nÝ nghĩa trong đề thi:")
        print("  - N: tổng số khoảng chia của bài (vd: x0→x40 nghĩa là N=40)")
        print("  - m=4/5/6: bậc Newton-Cotes (mỗi lần áp dụng dùng m+1 điểm)")
        print("  - Điều kiện ghép: N phải chia hết cho m")
        print("\nChương trình sẽ tính trên cùng N bằng:")
        print("  - Công thức điểm giữa (composite)")
        print("  - Công thức hình thang (composite)")
        print("  - Công thức Simpson (composite, nếu N chẵn)")
        print("  - Newton-Cotes ghép bậc m=4/5/6 (dùng hệ trọng số cho trước)")
        
        N_total = int(input('\nNhập tổng số khoảng chia của bài (N): '))
        if N_total <= 0:
            print("Lỗi: N phải là số nguyên dương.")
            return

        n = N_total
        h = (b-a)/n
        
        # Tính các giá trị hàm
        A       = [f.subs(x, a + i*h) for i in range(n+1)]
        A_mid   = [f.subs(x, a + (i + 0.5)*h) for i in range(n)]
        
        print("\n" + "="*60)
        print("THÔNG TIN ĐẦU VÀO")
        print("="*60)
        print(f"Hàm số: f(x) = {func}")
        print(f"Khoảng tích phân: [{a}, {b}]")
        print(f"Số khoảng chia: n = {n}")
        print(f"Bước chia: h = (b-a)/n = ({b}-{a})/{n} = {h:.10f}")
        print(f"Số điểm: {n+1} điểm")
        
        # Tính bằng các phương pháp
        midpoint(A_mid)
        midpoint_error()
        print()
        
        trapezoidal(A)
        trapezoidal_error()
        print()
        
        if n % 2 == 0:
            simpson(A)
            simpson_error()
            print()

        # Newton-Cotes bậc 4/5/6 theo kiểu ghép trên N
        print("\n" + "="*60)
        print("NEWTON-COTES GHÉP (BẬC 4/5/6) - NHẬP HỆ TRỌNG SỐ CHO TRƯỚC")
        print("="*60)
        print("\nBạn muốn tính cho các bậc nào? (mặc định: 4 5 6)")
        m_list_str = input("Nhập danh sách bậc m, cách nhau bởi dấu cách (vd: 4 6): ").strip()
        if not m_list_str:
            m_list = [4, 5, 6]
        else:
            m_list = [int(v) for v in m_list_str.split()]

        standard_weights = show_standard_weights()

        for m in m_list:
            print("\n" + "-"*60)
            print(f"XỬ LÝ NEWTON-COTES BẬC m = {m}")
            print("-"*60)
            if m <= 0:
                print("Bỏ qua: m phải là số nguyên dương.")
                continue
            if N_total % m != 0:
                print(f"Bỏ qua: N = {N_total} không chia hết cho m = {m} nên không ghép được.")
                continue

            print("\nChọn cách nhập hệ trọng số:")
            if m in standard_weights:
                print(f"  (1) Dùng bảng chuẩn (m={m})")
            print("  (2) Nhập hệ số theo đề (khuyến nghị)")
            mode = input("Chọn (mặc định 2): ").strip()
            if mode == '1' and m in standard_weights:
                Hs_panel = standard_weights[m]
            else:
                print(f"\nNhập {m+1} hệ số H0..H{m} (áp dụng cho từng panel bậc {m})")
                print("Ví dụ m=4: 7/90 32/90 12/90 32/90 7/90")
                while True:
                    try:
                        line = input(f"Nhập {m+1} hệ số: ").strip()
                        Hs_panel = _parse_weights_line(line)
                        if len(Hs_panel) != m + 1:
                            print(f"Lỗi: cần đúng {m+1} hệ số, bạn nhập {len(Hs_panel)}.")
                            continue
                        break
                    except Exception as e:
                        print(f"Lỗi: {e}")

            # Tính tích phân Newton-Cotes ghép
            composite_newton_cotes_exam_mode(m, Hs_panel, N_total)

            # Ước lượng sai số (chặn trên)
            try:
                total_err, per_panel_err, order, M = composite_newton_cotes_error_bound(m, N_total)
                print(f"\nƯớc lượng sai số (chặn trên):")
                print(f"  - Dùng đạo hàm bậc {order}, M = max|f^{order}(x)| trên [{a}, {b}] = {M:.10f}")
                print(f"  - Sai số mỗi panel ≤ {per_panel_err:.10f}")
                print(f"  - Tổng {N_total//m} panel ⇒ |E| ≤ {total_err:.10f}")
            except Exception as e:
                print(f"\nKhông ước lượng được sai số (SymPy có thể không tìm được max/min): {e}")
    
    if q == 3:
        print("\n" + "="*60)
        print("TÍNH HỆ TRỌNG SỐ NEWTON-COTES")
        print("="*60)
        print("\nNhập số khoảng chia n để tính các hệ trọng số tương ứng.")
        print("Ví dụ: n=1 (Hình thang), n=2 (Simpson), n=3 (Simpson 3/8), ...")
        print("Lưu ý: Với n lớn (>10), tính toán có thể mất thời gian.")
        
        while True:
            try:
                n_val = int(input("\nNhập n (hoặc nhập 0 để xem bảng chuẩn, -1 để thoát): "))
                
                if n_val == -1:
                    break
                elif n_val == 0:
                    show_standard_weights()
                elif n_val < 1:
                    print("Lỗi: n phải là số nguyên dương.")
                else:
                    if n_val > 15:
                        confirm = input(f"n = {n_val} khá lớn, tính toán có thể lâu. Tiếp tục? (y/n): ").strip().lower()
                        if confirm != 'y':
                            continue
                    
                    Hs = calculate_weights_for_n(n_val)
                    
                    # Hỏi có muốn lưu không
                    save_choice = input("\nBạn có muốn copy các hệ số này để dùng sau? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        print("\nCó thể copy dòng sau để nhập thủ công:")
                        print(f"[{', '.join([f'{h:.15f}' for h in Hs])}]")
                
                continue_choice = input("\nTiếp tục tính hệ trọng số cho n khác? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
                    
            except ValueError:
                print("Lỗi: Vui lòng nhập số nguyên hợp lệ.")
            except Exception as e:
                print(f"Lỗi: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()
