# -*- coding: utf-8 -*-
import numpy as np
import math
import sys

# Cấu hình encoding cho Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def data_input():
    """Đọc dữ liệu từ file input.txt"""
    x = []
    y = []
    with open('dhtp/input.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                x.append(float(parts[0]))
                y.append(float(parts[1]))   
    x = np.array(x)
    y = np.array(y)
    n = len(x) - 1  # số khoảng chia
    return x, y, n

def trapezoidal(x, y, n):
    """Tính tích phân bằng phương pháp hình thang từ dữ liệu rời rạc"""
    print("\n" + "="*60)
    print("PHƯƠNG PHÁP HÌNH THANG (TRAPEZOIDAL RULE)")
    print("="*60)
    print(f"\nCông thức: ∫[a,b] f(x)dx ≈ Σᵢ₌₀ⁿ⁻¹ (hᵢ/2) * [f(xᵢ) + f(xᵢ₊₁)]")
    print(f"Hay công thức I = (h/2) * [f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(xₙ)]")
    print(f"Với hᵢ = xᵢ₊₁ - xᵢ")
    print(f"\nKhoảng tích phân: [{x[0]:.6f}, {x[-1]:.6f}]")
    print(f"Số khoảng chia: n = {n}")
    print(f"Số điểm: {n+1} điểm")
    
    print(f"\nCác điểm và giá trị:")
    for i in range(n+1):
        if (i < 3 or i > n-2):
            print(f"  x{i} = {x[i]:.6f}  →  y{i} = f(x{i}) = {y[i]:.6f}")
    
    # Kiểm tra khoảng cách đều
    h = x[1] - x[0]
    is_uniform = True
    for i in range(1, n):
        if abs((x[i+1] - x[i]) - h) > 1e-10:
            is_uniform = False
            break
    
    result = 0
    
    if is_uniform:
        # Công thức tổng quát khi khoảng cách đều
        print(f"\nCác bước tính (khoảng cách đều h = {h:.10f}):")
        print(f"  Bước 1: Tính tổng đầu cuối")
        print(f"    y₀ + yₙ = {y[0]:.6f} + {y[n]:.6f} = {y[0] + y[n]:.6f}")
        
        print(f"\n  Bước 2: Tính tổng các điểm giữa (nhân với hệ số 2)")
        middle_sum = sum(y[i] for i in range(1, n))
        middle_indices = list(range(1, n))
        print(f"    Các điểm giữa: {', '.join([f'x{i}' for i in middle_indices[:5]])}{'...' if len(middle_indices) > 5 else ''}")
        print(f"    Σyᵢ(giữa) = ", end="")
        if len(middle_indices) <= 10:
            print(" + ".join([f"y{i}" for i in middle_indices]), end="")
        else:
            print(" + ".join([f"y{i}" for i in middle_indices[:3]]) + " + ... + " + " + ".join([f"y{i}" for i in middle_indices[-3:]]), end="")
        print(f" = ", end="")
        if len(middle_indices) <= 10:
            print(" + ".join([f"{y[i]:.6f}" for i in middle_indices]), end="")
        else:
            print(" + ".join([f"{y[i]:.6f}" for i in middle_indices[:3]]) + " + ... + " + " + ".join([f"{y[i]:.6f}" for i in middle_indices[-3:]]), end="")
        print(f" = {middle_sum:.6f}")
        print(f"    2 * Σyᵢ(giữa) = 2 * {middle_sum:.6f} = {2*middle_sum:.6f}")
        
        print(f"\n  Bước 3: Tính tích phân")
        print(f"    I = (h/2) * [y₀ + 2Σyᵢ(giữa) + yₙ]")
        print(f"      = ({h:.10f}/2) * [{y[0]:.6f} + 2*{middle_sum:.6f} + {y[n]:.6f}]")
        print(f"      = {h/2:.10f} * [{y[0]:.6f} + {2*middle_sum:.6f} + {y[n]:.6f}]")
        print(f"      = {h/2:.10f} * [{y[0] + y[n]:.6f} + {2*middle_sum:.6f}]")
        print(f"      = {h/2:.10f} * [{y[0] + y[n] + 2*middle_sum:.6f}]")
        
        result = (h/2) * (y[0] + y[n] + 2*middle_sum)
        print(f"      = {result:.10f}")
    else:
        # Tính từng khoảng khi khoảng cách không đều
        print(f"\nCác bước tính (khoảng cách không đều):")
        print(f"  Tích phân = Σᵢ₌₀ⁿ⁻¹ (hᵢ/2) * (yᵢ + yᵢ₊₁)")
        print(f"\n  Tính từng khoảng:")
        
        areas = []
        for i in range(n):
            h_i = x[i+1] - x[i]
            area_i = (h_i / 2) * (y[i] + y[i+1])
            result += area_i
            areas.append(area_i)
            
            if (i < 3 or i > n-3):
                print(f"    Khoảng [{i}, {i+1}]:")
                print(f"      h{i} = x{i+1} - x{i} = {x[i+1]:.6f} - {x[i]:.6f} = {h_i:.6f}")
                print(f"      Diện tích = (h{i}/2) * (y{i} + y{i+1})")
                print(f"                = ({h_i:.6f}/2) * ({y[i]:.6f} + {y[i+1]:.6f})")
                print(f"                = {h_i/2:.6f} * {y[i] + y[i+1]:.6f}")
                print(f"                = {area_i:.10f}")
        
        print(f"\n  Bước 4: Tổng hợp kết quả")
        print(f"    Tổng = ", end="")
        if n <= 10:
            print(" + ".join([f"{a:.10f}" for a in areas]), end="")
        else:
            print(" + ".join([f"{a:.10f}" for a in areas[:3]]) + " + ... + " + " + ".join([f"{a:.10f}" for a in areas[-3:]]), end="")
        print(f" = {result:.10f}")
    
    print(f"\nTích phân bằng phương pháp hình thang: {result:.10f}")
    return result

def simpson(x, y, n):
    """Tính tích phân bằng phương pháp Simpson từ dữ liệu rời rạc"""
    print("\n" + "="*60)
    print("PHƯƠNG PHÁP SIMPSON")
    print("="*60)
    
    # Kiểm tra số khoảng có chẵn không
    if n % 2 != 0:
        print(f"\n⚠️  CẢNH BÁO: Phương pháp Simpson yêu cầu số khoảng chia n phải là số chẵn!")
        print(f"   Hiện tại n = {n} là số lẻ.")
        print(f"   Sẽ sử dụng phương pháp hình thang cho khoảng cuối cùng.")
        use_composite = True
    else:
        use_composite = False
    
    # Kiểm tra khoảng cách đều
    h = x[1] - x[0]
    is_uniform = True
    for i in range(1, n):
        if abs((x[i+1] - x[i]) - h) > 1e-10:
            is_uniform = False
            break
    
    if not is_uniform:
        print(f"\n⚠️  CẢNH BÁO: Các điểm không cách đều!")
        print(f"   Phương pháp Simpson yêu cầu khoảng cách đều.")
        print(f"   Sẽ sử dụng phương pháp hình thang tổng quát.")
        return trapezoidal(x, y, n)
    
    print(f"\nCông thức: ∫[a,b] f(x)dx ≈ h/3 * [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 4f(xₙ₋₁) + f(xₙ)]")
    print(f"Với h = (b-a)/n = ({x[-1]}-{x[0]})/{n} = {h:.10f}")
    print(f"Khoảng tích phân: [{x[0]:.6f}, {x[-1]:.6f}]")
    print(f"Số khoảng chia: n = {n} {'✓' if n%2==0 else '✗'}")
    
    print(f"\nCác điểm và giá trị:")
    for i in range(n+1):
        coeff = 1 if (i == 0 or i == n) else (4 if i%2 == 1 else 2)
        if (i < 2 or i > n-2):
            print(f"  x{i} = {x[i]:.6f}  →  y{i} = {y[i]:.6f}  (hệ số: {coeff})")
    
    if use_composite:
        # Sử dụng Simpson cho n-1 khoảng đầu và hình thang cho khoảng cuối
        result = 0
        # Simpson cho n-1 khoảng đầu (n-1 phải chẵn)
        m = n - 1
        if m >= 2 and m % 2 == 0:
            simp_odd = sum(y[i] for i in range(1, m, 2))
            simp_even = sum(y[i] for i in range(2, m, 2))
            
            print(f"\nCác bước tính (Simpson cho [{x[0]}, {x[m]}]):")
            print(f"  Tổng đầu cuối: y₀ + yₘ = {y[0]:.6f} + {y[m]:.6f} = {y[0] + y[m]:.6f}")
            
            odd_indices = list(range(1, m, 2))
            even_indices = list(range(2, m, 2))
            
            if odd_indices:
                print(f"  Các điểm lẻ: {', '.join([f'x{i}' for i in odd_indices[:5]])}{'...' if len(odd_indices) > 5 else ''}")
                print(f"  Tổng điểm lẻ: Σyᵢ(lẻ) = ", end="")
                if len(odd_indices) <= 10:
                    print(" + ".join([f"y{i}" for i in odd_indices]), end="")
                else:
                    print(" + ".join([f"y{i}" for i in odd_indices[:3]]) + " + ... + " + " + ".join([f"y{i}" for i in odd_indices[-3:]]), end="")
                print(f" = {simp_odd:.6f}")
                print(f"  4 * Σyᵢ(lẻ) = 4 * {simp_odd:.6f} = {4*simp_odd:.6f}")
            
            if even_indices:
                print(f"  Các điểm chẵn: {', '.join([f'x{i}' for i in even_indices[:5]])}{'...' if len(even_indices) > 5 else ''}")
                print(f"  Tổng điểm chẵn: Σyᵢ(chẵn) = ", end="")
                if len(even_indices) <= 10:
                    print(" + ".join([f"y{i}" for i in even_indices]), end="")
                else:
                    print(" + ".join([f"y{i}" for i in even_indices[:3]]) + " + ... + " + " + ".join([f"y{i}" for i in even_indices[-3:]]), end="")
                print(f" = {simp_even:.6f}")
                print(f"  2 * Σyᵢ(chẵn) = 2 * {simp_even:.6f} = {2*simp_even:.6f}")
            
            result_simp = (h/3) * (y[0] + y[m] + 4*simp_odd + 2*simp_even)
            print(f"\n  Kết quả Simpson = h/3 * [y₀ + 4Σyᵢ(lẻ) + 2Σyᵢ(chẵn) + yₘ]")
            print(f"                  = {h}/3 * [{y[0]:.6f} + 4*{simp_odd:.6f} + 2*{simp_even:.6f} + {y[m]:.6f}]")
            print(f"                  = {h/3:.10f} * [{y[0] + y[m]:.6f} + {4*simp_odd:.6f} + {2*simp_even:.6f}]")
            print(f"                  = {h/3:.10f} * [{y[0] + y[m] + 4*simp_odd + 2*simp_even:.6f}]")
            print(f"                  = {result_simp:.10f}")
            result += result_simp
        
        # Hình thang cho khoảng cuối
        h_last = x[n] - x[n-1]
        area_last = (h_last / 2) * (y[n-1] + y[n])
        print(f"\n  Hình thang cho khoảng cuối [{x[n-1]}, {x[n]}]:")
        print(f"    h = {x[n]:.6f} - {x[n-1]:.6f} = {h_last:.6f}")
        print(f"    Diện tích = (h/2) * (y{n-1} + y{n})")
        print(f"              = ({h_last:.6f}/2) * ({y[n-1]:.6f} + {y[n]:.6f})")
        print(f"              = {h_last/2:.6f} * {y[n-1] + y[n]:.6f}")
        print(f"              = {area_last:.10f}")
        result += area_last
        
        print(f"\n  Tổng kết quả = Simpson + Hình thang")
        print(f"                = {result - area_last:.10f} + {area_last:.10f}")
        print(f"                = {result:.10f}")
        print(f"\nTích phân bằng phương pháp Simpson (hỗn hợp): {result:.10f}")
    else:
        # Simpson chuẩn
        simp_odd = sum(y[i] for i in range(1, n, 2))
        simp_even = sum(y[i] for i in range(2, n, 2))
        
        print(f"\nCác bước tính:")
        print(f"  Bước 1: Tính tổng đầu cuối")
        print(f"    y₀ + yₙ = {y[0]:.6f} + {y[n]:.6f} = {y[0] + y[n]:.6f}")
        
        if n >= 2:
            odd_indices = list(range(1, n, 2))
            even_indices = list(range(2, n, 2))
            
            print(f"\n  Bước 2: Tính tổng các điểm lẻ (nhân với hệ số 4)")
            print(f"    Các điểm lẻ: {', '.join([f'x{i}' for i in odd_indices[:5]])}{'...' if len(odd_indices) > 5 else ''}")
            print(f"    Σyᵢ(lẻ) = ", end="")
            if len(odd_indices) <= 10:
                print(" + ".join([f"y{i}" for i in odd_indices]), end="")
            else:
                print(" + ".join([f"y{i}" for i in odd_indices[:3]]) + " + ... + " + " + ".join([f"y{i}" for i in odd_indices[-3:]]), end="")
            print(f" = ", end="")
            if len(odd_indices) <= 10:
                print(" + ".join([f"{y[i]:.6f}" for i in odd_indices]), end="")
            else:
                print(" + ".join([f"{y[i]:.6f}" for i in odd_indices[:3]]) + " + ... + " + " + ".join([f"{y[i]:.6f}" for i in odd_indices[-3:]]), end="")
            print(f" = {simp_odd:.6f}")
            print(f"    4 * Σyᵢ(lẻ) = 4 * {simp_odd:.6f} = {4*simp_odd:.6f}")
            
            print(f"\n  Bước 3: Tính tổng các điểm chẵn (nhân với hệ số 2)")
            print(f"    Các điểm chẵn: {', '.join([f'x{i}' for i in even_indices[:5]])}{'...' if len(even_indices) > 5 else ''}")
            print(f"    Σyᵢ(chẵn) = ", end="")
            if len(even_indices) <= 10:
                print(" + ".join([f"y{i}" for i in even_indices]), end="")
            else:
                print(" + ".join([f"y{i}" for i in even_indices[:3]]) + " + ... + " + " + ".join([f"y{i}" for i in even_indices[-3:]]), end="")
            print(f" = ", end="")
            if len(even_indices) <= 10:
                print(" + ".join([f"{y[i]:.6f}" for i in even_indices]), end="")
            else:
                print(" + ".join([f"{y[i]:.6f}" for i in even_indices[:3]]) + " + ... + " + " + ".join([f"{y[i]:.6f}" for i in even_indices[-3:]]), end="")
            print(f" = {simp_even:.6f}")
            print(f"    2 * Σyᵢ(chẵn) = 2 * {simp_even:.6f} = {2*simp_even:.6f}")
        
        print(f"\n  Bước 4: Tính tích phân")
        print(f"    I = h/3 * [y₀ + 4Σyᵢ(lẻ) + 2Σyᵢ(chẵn) + yₙ]")
        print(f"      = {h}/3 * [{y[0]:.6f} + 4*{simp_odd:.6f} + 2*{simp_even:.6f} + {y[n]:.6f}]")
        print(f"      = {h/3:.10f} * [{y[0]:.6f} + {4*simp_odd:.6f} + {2*simp_even:.6f} + {y[n]:.6f}]")
        print(f"      = {h/3:.10f} * [{y[0] + y[n]:.6f} + {4*simp_odd:.6f} + {2*simp_even:.6f}]")
        print(f"      = {h/3:.10f} * [{y[0] + y[n] + 4*simp_odd + 2*simp_even:.6f}]")
        
        result = (h/3) * (y[0] + y[n] + 4*simp_odd + 2*simp_even)
        print(f"      = {result:.10f}")
        print(f"\nTích phân bằng phương pháp Simpson: {result:.10f}")
    
    return result

def newton_cotes(x, y, n, m_choice=0):
    """Tính tích phân bằng phương pháp Newton-Cotes tổng hợp (Composite) từ dữ liệu rời rạc.
    
    - Nếu n <= 6: tương đương Newton-Cotes "native" (1 panel).
    - Nếu n > 6: tự động ghép nhiều panel với bậc m <= 6 (mặc định chọn lớn nhất có thể).
      Điều kiện ghép: n % m == 0 (n là tổng số khoảng, m là bậc công thức nhỏ).
    """
    print("\n" + "="*60)
    print("PHƯƠNG PHÁP NEWTON-COTES")
    print("="*60)
    
    # Kiểm tra khoảng cách đều
    h = x[1] - x[0]
    is_uniform = True
    for i in range(1, n):
        if abs((x[i+1] - x[i]) - h) > 1e-10:
            is_uniform = False
            break
    
    if not is_uniform:
        print(f"\n⚠️  CẢNH BÁO: Các điểm không cách đều!")
        print(f"   Phương pháp Newton-Cotes yêu cầu khoảng cách đều.")
        print(f"   Không thể áp dụng phương pháp này.")
        return None
    
    # In công thức và thông tin cơ bản
    a_total = x[0]
    b_total = x[-1]
    print(f"\nCông thức Newton-Cotes tổng hợp (Composite):")
    print(f"  ∫[a,b] f(x)dx ≈ Σ_panel [L_panel * Σ_j₌₀ᵐ H_j * f(x_j)]")
    print(f"  Với mỗi panel: I_panel ≈ L * [H₀*f(x₀) + H₁*f(x₁) + ... + Hₘ*f(xₘ)]")
    print(f"  Trong đó: L = độ dài panel = m*h, h = (b-a)/n")
    print(f"\nKhoảng tích phân: [{a_total:.6f}, {b_total:.6f}]")
    print(f"Số khoảng chia: n = {n}")
    print(f"Số điểm: {n+1} điểm")
    print(f"Bước đều: h = (b-a)/n = ({b_total}-{a_total})/{n} = {h:.10f}")
    
    # In một số điểm và giá trị hàm
    print(f"\nCác điểm và giá trị (một số điểm đầu/cuối):")
    for i in range(min(4, n+1)):
        print(f"  x{i} = {x[i]:.6f}  →  y{i} = f(x{i}) = {y[i]:.6f}")
    if n+1 > 8:
        print(f"  ...")
    for i in range(max(4, n-3), n+1):
        print(f"  x{i} = {x[i]:.6f}  →  y{i} = f(x{i}) = {y[i]:.6f}")

    def _multiply_horner(A, i):
        """Nhân đa thức với (t-i)."""
        A.append(0)
        for j in range(len(A) - 1, 0, -1):
            A[j] = A[j] - A[j - 1] * i
        return A

    def _divide_horner(A, i):
        """Chia đa thức cho (t-i)."""
        X = A.copy()
        X.pop()
        for j in range(1, len(X)):
            X[j] = i * X[j - 1] + X[j]
        return X

    def _poly_integral(A, a_val, b_val):
        """Tính tích phân xác định của đa thức từ a_val đến b_val."""
        I = 0.0
        for j in range(len(A)):
            if A[j] == 0:
                continue
            power = len(A) - j - 1
            coeff = A[j] / (power + 1)
            I += coeff * (b_val ** (power + 1) - a_val ** (power + 1))
        return I

    def _cotes_weights(m):
        """Tính hệ trọng số Newton-Cotes H0..Hm cho bậc m (chuẩn hóa theo công thức panel)."""
        # Tạo đa thức D(t) = (t-0)(t-1)...(t-m)
        D = [1]
        for i in range(m + 1):
            _multiply_horner(D, i)

        Hs = []
        for i in range(m + 1):
            X = _divide_horner(D.copy(), i)
            integral_val = _poly_integral(X, 0, m)
            sign = (-1) ** (m - i)
            H_i = (1 / m) * (sign / (math.factorial(i) * math.factorial(m - i))) * integral_val
            Hs.append(H_i)
        return Hs

    # Xác định chiến lược chọn bậc
    if m_choice in (None, 0):
        print(f"\nChọn bậc: tự động (m ≤ 6, ưu tiên m lớn nhất có thể)")
        auto_mode = True
    else:
        if not (1 <= int(m_choice) <= 6):
            print(f"\n⚠️  CẢNH BÁO: m = {m_choice} không hợp lệ (chỉ nhận 1..6 hoặc 0=tự động).")
            print("   → Chuyển sang chế độ tự động.")
            m_choice = 0
            auto_mode = True
        else:
            print(f"\nChọn bậc: m = {m_choice} (cố định cho các panel đủ điều kiện)")
            auto_mode = False

    total_result = 0.0
    weights_cache = {}
    panel_results = []
    printed_weights = set()  # Track các bậc m đã in hệ số

    print(f"\n" + "="*60)
    print("CÁC BƯỚC TÍNH TOÁN CHI TIẾT")
    print("="*60)
    
    p = 0
    start = 0
    while start < n:
        remaining = n - start
        if auto_mode:
            m = min(6, remaining)  # tự động: ưu tiên bậc cao nhất có thể trên phần còn lại
        else:
            m = min(int(m_choice), remaining)
        end = start + m

        if m not in weights_cache:
            weights_cache[m] = _cotes_weights(m)
        Hs = weights_cache[m]

        a_p = x[start]
        b_p = x[end]
        L = b_p - a_p

        # In hệ số Cotes nếu chưa in cho bậc này
        if m not in printed_weights:
            print(f"\nHệ số Cotes cho bậc m = {m}:")
            for j in range(m + 1):
                print(f"  H{j} = {Hs[j]:.10f}")
            print(f"  (Tổng kiểm tra: ΣH_j = {sum(Hs):.6f})")
            printed_weights.add(m)

        # Tính tích phân cho panel này
        if p < 3 or start >= n - 2*m:
            print(f"\nPanel {p+1}: bậc m = {m}, khoảng [{a_p:.6f}, {b_p:.6f}]")
            print(f"  Độ dài panel: L = {b_p:.6f} - {a_p:.6f} = {L:.10f}")
            print(f"  Các điểm trong panel:")
        
        panel_sum = 0.0
        terms_str = []
        values_str = []
        
        for j in range(m + 1):
            idx = start + j
            term_val = Hs[j] * y[idx]
            panel_sum += term_val
            if j < 3 or j >= m - 1:
                if p < 3 or start >= n - 2*m:
                    print(f"    x{idx} = {x[idx]:.6f}  →  y{idx} = {y[idx]:.6f}  (H{j} = {Hs[j]:.10f})")
            if j == 0:
                terms_str.append(f"H{j}*y{idx}")
                values_str.append(f"{Hs[j]:.10f}*{y[idx]:.6f}")
            elif j < m:
                terms_str.append(f"H{j}*y{idx}")
                values_str.append(f"{Hs[j]:.10f}*{y[idx]:.6f}")
            else:
                terms_str.append(f"H{j}*y{idx}")
                values_str.append(f"{Hs[j]:.10f}*{y[idx]:.6f}")
        if p < 3 or start >= n - 2*m:
            print(f"\n  Bước tính cho panel {p+1}:")
            print(f"    Σ H_j * y_j = ", end="")
            if m + 1 <= 7:
                print(" + ".join(terms_str))
            else:
                print(" + ".join(terms_str[:3]) + " + ... + " + " + ".join(terms_str[-2:]))
            
            print(f"                 = ", end="")
            if m + 1 <= 7:
                print(" + ".join(values_str))
            else:
                print(" + ".join(values_str[:3]) + " + ... + " + " + ".join(values_str[-2:]))
            
            print(f"                 = {panel_sum:.10f}")
        
        I_p = L * panel_sum
        total_result += I_p
        panel_results.append((m, I_p, a_p, b_p))
        if p < 3 or start >= n - 2*m:
            print(f"\n    I_panel_{p+1} = L * Σ H_j * y_j")
            print(f"                 = {L:.10f} * {panel_sum:.10f}")
            print(f"                 = {I_p:.10f}")

        p += 1
        start = end

    # Tổng hợp kết quả
    print(f"\n" + "="*60)
    print("TỔNG HỢP KẾT QUẢ")
    print("="*60)
    print(f"Tổng số panel: {p}")
    print(f"\nTích phân = Σ (I_panel)")
    print(f"          = ", end="")
    if p <= 5:
        print(" + ".join([f"I_panel_{i+1}" for i in range(p)]))
        print(f"          = ", end="")
        print(" + ".join([f"{panel_results[i][1]:.10f}" for i in range(p)]))
    else:
        print("I_panel_1 + I_panel_2 + ... + I_panel_" + str(p-1) + " + I_panel_" + str(p))
        print(f"          = ", end="")
        print(f"{panel_results[0][1]:.10f} + {panel_results[1][1]:.10f} + ... + {panel_results[-2][1]:.10f} + {panel_results[-1][1]:.10f}")
    print(f"          = {total_result:.10f}")
    
    print(f"\nTích phân bằng phương pháp Newton-Cotes tổng hợp: {total_result:.10f}")
    return total_result

def main():
    """Hàm main để chạy chương trình"""
    print("="*60)
    print("TÍNH TÍCH PHÂN TỪ DỮ LIỆU RỜI RẠC")
    print("="*60)
    
    # Đọc dữ liệu
    x, y, n = data_input()
    
    print(f"\nĐã đọc {n+1} điểm dữ liệu từ file input.txt")
    print(f"Khoảng tích phân: [{x[0]:.6f}, {x[-1]:.6f}]")
    
    # Tính tích phân bằng các phương pháp
    result_trap = trapezoidal(x, y, n)
    
    result_simp = simpson(x, y, n)
    
    print("\n" + "="*60)
    print("CHỌN BẬC CHO NEWTON-COTES (COMPOSITE)")
    print("="*60)
    print("Nhập bậc m (1..6) để ghép Newton-Cotes bậc m.")
    print("Nhập 0 để tự động chọn bậc theo từng panel.")
    try:
        m_choice = int(input("Chọn m (0..6): ").strip())
    except Exception:
        m_choice = 0
    result_nc = newton_cotes(x, y, n, m_choice=m_choice)
    
    # Tóm tắt kết quả
    print("\n" + "="*60)
    print("TÓM TẮT KẾT QUẢ")
    print("="*60)
    print(f"Phương pháp hình thang      : {result_trap:.10f}")
    if result_simp is not None:
        print(f"Phương pháp Simpson        : {result_simp:.10f}")
    if result_nc is not None:
        print(f"Phương pháp Newton-Cotes    : {result_nc:.10f}")

if __name__ == '__main__':
    main()