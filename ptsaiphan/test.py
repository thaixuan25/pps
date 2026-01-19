import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

def solve_eigenvalue_program(L, N, p_func, q_func, r_func):
    """
    Giải bài toán giá trị riêng: -[p(x)u']' - q(x)u = lambda * r(x) * u
    Sử dụng công thức sai phân chính xác từ hình ảnh cung cấp.
    
    Inputs:
        L: Độ dài đoạn [0, L]
        N: Số khoảng chia
        p_func, q_func, r_func: Các hàm số p(x), q(x), r(x)
    Outputs:
        eigenvalues: Các giá trị riêng tìm được (sắp xếp tăng dần)
        eigenvectors: Các vector riêng tương ứng
        x_grid: Lưới điểm x
    """
    # 1. Thiết lập lưới
    h = L / N
    x_grid = np.linspace(0, L, N + 1)
    
    # Ma trận K sẽ có kích thước (N-1) x (N-1) 
    # (Do điều kiện biên u0 = uN = 0, ta chỉ tìm u1...u_{N-1})
    size = N - 1
    K = np.zeros((size, size))
    
    # 2. Xây dựng Ma trận K dựa trên công thức a_i, b_i, c_i
    for i in range(size):
        # Chỉ số thực tế của nút lưới là i+1 (vì i chạy từ 0..N-2 tương ứng u1..u_{N-1})
        idx_real = i + 1 
        xi = x_grid[idx_real]
        
        # Tính toán các giá trị hàm tại điểm xi và điểm bán nguyên
        p_minus = p_func(xi - h/2)  # p(xi - h/2)
        p_plus  = p_func(xi + h/2)  # p(xi + h/2)
        r_val   = r_func(xi)        # r(xi)
        q_val   = q_func(xi)        # q(xi)
        
        # --- ÁP DỤNG CÔNG THỨC TRONG ẢNH ---
        # Hệ số a_i (đi với u_{i-1})
        ai = -p_minus / ((h**2) * r_val)
        
        # Hệ số b_i (đi với u_i) - Lưu ý dấu trừ của q/r
        bi = (p_plus + p_minus) / ((h**2) * r_val) - (q_val / r_val)
        
        # Hệ số c_i (đi với u_{i+1})
        ci = -p_plus / ((h**2) * r_val)
        
        # Điền vào ma trận K (Phương trình: a*u_{i-1} + (b-lambda)u_i + c*u_{i+1} = 0 => K*u = lambda*u)
        # Đường chéo chính
        K[i, i] = bi
        
        # Đường chéo dưới (liên quan u_{i-1})
        if i > 0:
            K[i, i-1] = ai
            
        # Đường chéo trên (liên quan u_{i+1})
        if i < size - 1:
            K[i, i+1] = ci
            
    # 3. Giải bài toán giá trị riêng của ma trận K
    # Dùng hàm eig để tìm toàn bộ
    print("Kích thước ma trận K: ", K.shape)
    print(K)
    vals, vecs = eig(K)
    matrixK = K[1:-1, 1:-1]
    print("Kích thước ma trận matrixK: ", matrixK.shape)
    
    char_poly_coeffs = np.poly(matrixK)
    print("Đa thức đặc trưng: ", char_poly_coeffs)
    eigenvl = np.roots(char_poly_coeffs)
    # Loại bỏ phần ảo
    eigenvl = eigenvl.real
    # Loại bỏ trùng lặp
    eigenvl = np.unique(eigenvl)
    print("Giá trị riêng: ")
    for i in eigenvl:
        print(f"{i:>6.4f}")
    # Lọc kết quả: Chỉ lấy phần thực (về lý thuyết Sturm-Liouville lambda là số thực)
    vals = np.real(vals)
    
    # Sắp xếp từ bé đến lớn
    idx_sorted = np.argsort(vals)
    vals = vals[idx_sorted]
    vecs = vecs[:, idx_sorted]
    
    return vals, vecs, x_grid

# =======================================================
# VÍ DỤ KIỂM CHỨNG (Bài toán dây rung cơ bản)
# Phương trình: -u'' = lambda * u  trên [0, 1]
# Nghiệm lý thuyết: lambda_k = (k * pi)^2  => 9.86, 39.47, 88.82...
# =======================================================

# Định nghĩa các hàm p, q, r cho phương trình -u'' = lambda u
# So sánh với dạng tổng quát -[p u']' - q u = lambda r u
# Ta có: p=1, q=0, r=1
def p(x): return 1.0
def q(x): return 0.0
def r(x): return 1.0

# Chạy chương trình
L_test = 1.0
N_test = 50 # Số điểm chia
lambdas, modes, x = solve_eigenvalue_program(L_test, N_test, p, q, r)

# In kết quả
print(f"--- KẾT QUẢ TÌM GIÁ TRỊ RIÊNG (N={N_test}) ---")
print(f"{'Mode k':<10} | {'Lambda Tính toán':<20} | {'Lambda Lý thuyết':<20} | {'Sai số':<10}")
print("-" * 70)

for k in range(5): # In 5 giá trị đầu tiên
    val_calc = lambdas[k]
    val_true = ((k + 1) * np.pi) ** 2
    error = abs(val_calc - val_true)
    print(f"{k+1:<10} | {val_calc:<20.4f} | {val_true:<20.4f} | {error:<10.4f}")

# Vẽ đồ thị hàm riêng đầu tiên (u1)
# Thêm điểm biên 0 vào đầu và cuối để vẽ cho đẹp
u_plot = np.concatenate(([0], np.real(modes[:, 0]), [0]))
plt.figure(figsize=(8, 5))
plt.plot(x, u_plot, 'b-o', label=f'Hàm riêng k=1 ($\lambda \\approx {lambdas[0]:.2f}$)')
plt.title("Dáng điệu hàm riêng đầu tiên $u_1(x)$")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.legend()
plt.show()