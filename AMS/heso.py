import sympy as sp

def calc_adams_bashforth_coeffs(s_steps):
    """
    Tính hệ số Adams-Bashforth dựa trên công thức chính xác trong ảnh image_148761.png
    """
    u = sp.symbols('u')
    betas = []
    
    # Lưu ý: Trong ảnh, i chạy từ 1 đến s
    for i in range(1, s_steps + 1):
        
        # 1. Xây dựng biểu thức trong dấu tích phân
        # Product_{j=1, j!=i}^s (u + j - 1)
        product_term = 1
        for j in range(1, s_steps + 1):
            if j == i:
                continue
            product_term *= (u + j - 1)
            
        # 2. Tính tích phân từ 0 đến 1
        integral_val = sp.integrate(product_term, (u, 0, 1))
        
        # 3. Tính hệ số giai thừa bên ngoài
        # Công thức: (-1)^(i-1) / [ (i-1)! * (s-i)! ]
        numerator_factor = (-1)**(i - 1)
        denominator_factor = sp.factorial(i - 1) * sp.factorial(s_steps - i)
        
        factor = numerator_factor / denominator_factor
        
        # 4. Tính Beta_i
        beta_i = factor * integral_val
        betas.append(beta_i)
        
    return betas

def calc_adams_moulton_coeffs(s_steps):
    """
    Tính các hệ số Beta_i cho phương pháp Adams-Moulton s bước
    dựa trên công thức tích phân.
    """
    u = sp.symbols('u')
    betas = []
    
    # Công thức: Beta_i = [(-1)^i / (i! * (s-i)!)] * Integral_0^1 [ Product_{j=0, j!=i}^s (u + j - 1) du ]
    
    # Duyệt qua i từ 0 đến s
    for i in range(s_steps + 1):
        # 1. Xây dựng biểu thức bên trong dấu tích phân (Product term)
        product_term = 1
        for j in range(s_steps + 1):
            if j == i:
                continue # Bỏ qua phần tử j = i theo công thức
            product_term *= (u + j - 1)
            
        # 2. Tính tích phân từ 0 đến 1
        integral_val = sp.integrate(product_term, (u, 0, 1))
        
        # 3. Tính phần hệ số giai thừa bên ngoài
        # (-1)^i
        sign = (-1)**i
        # i! * (s-i)!
        denom = sp.factorial(i) * sp.factorial(s_steps - i)
        factor = sign / denom
        
        # 4. Kết quả Beta_i
        beta_i = factor * integral_val
        betas.append(beta_i)
        
    return betas

# --- CHẠY THỬ NGHIỆM ---

# Thử với s = 2 (Adams-Moulton 2 bước - Thường gọi là bậc 3)
# Công thức thường thấy: y_{n+1} = y_n + h/12 * (5*f_{n+1} + 8*f_n - 1*f_{n-1})
# Tức là Beta_0 = 5/12, Beta_1 = 8/12, Beta_2 = -1/12
s = input("Nhập số bước: ")
coeffs = calc_adams_bashforth_coeffs(int(s))

print(f"--- Các hệ số Adams-Bashforth cho s = {s} ---")
for idx, val in enumerate(coeffs):
    print(f"Beta_{idx+1} = {val}")

coeffs = calc_adams_moulton_coeffs(int(s))
print(f"--- Các hệ số Adams-Moulton cho s = {s} ---")
for idx, val in enumerate(coeffs):
    print(f"Beta_{idx} = {val}")