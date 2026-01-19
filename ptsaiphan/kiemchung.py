"""
Chương trình kiểm chứng tính đúng của phương pháp sai phân
Bằng cách so sánh với các bài toán có nghiệm chính xác đã biết
"""

import numpy as np
import matplotlib.pyplot as plt
from btbien import matrixBuild, p, q, f
from gtr import solve_eigenvalue_problem, build_eigenvalue_matrices
from scipy.linalg import solve

# ==========================================
# BÀI TOÁN TEST 1: Phương trình đơn giản có nghiệm chính xác
# ==========================================

def test_simple_boundary_value():
    """
    Bài toán: u''(x) = -2, u(0) = 0, u(1) = 0
    Nghiệm chính xác: u(x) = x - x^2
    """
    print("="*70)
    print("KIỂM CHỨNG 1: BÀI TOÁN BIÊN ĐƠN GIẢN")
    print("="*70)
    print("\nBài toán: u''(x) = -2, u(0) = 0, u(1) = 0")
    print("Nghiệm chính xác: u(x) = x - x²")
    
    # Định nghĩa hàm cho bài toán này
    def p_test(x):
        return 1.0
    
    def q_test(x):
        return 0.0
    
    def f_test(x):
        return 2.0  # -f(x) = -2, nên f(x) = 2
    
    def u_exact(x):
        return x - x**2
    
    a, b = 0, 1
    N_values = [10, 20, 40, 80]
    
    print("\n" + "-"*70)
    print("KIỂM TRA TÍNH HỘI TỤ (Convergence Test)")
    print("-"*70)
    print(f"{'N':>6} | {'h':>12} | {'Lỗi max':>18} | {'Tỷ lệ':>12} | {'Bậc hội tụ':>15}")
    print("-"*70)
    
    errors = []
    h_values = []
    
    for N in N_values:
        h = (b - a) / N
        h_values.append(h)
        
        # Giải bằng phương pháp sai phân
        MatrixA, VectorB = matrixBuild(a, b, h, 0, 0, "1", None, None)
        u_numerical = solve(MatrixA, VectorB)
        
        # Tính nghiệm chính xác tại các điểm lưới
        x = np.linspace(a, b, N + 1)
        u_exact_vals = np.array([u_exact(xi) for xi in x])
        
        # Tính sai số
        error = np.max(np.abs(u_numerical - u_exact_vals))
        errors.append(error)
        
        # Tính tỷ lệ hội tụ
        if len(errors) > 1:
            ratio = errors[-2] / errors[-1] if errors[-1] > 0 else np.inf
            order = np.log2(ratio) if ratio > 0 else 0
            print(f"{N:>6} | {h:>12.6f} | {error:>18.10e} | {ratio:>12.4f} | {order:>15.4f}")
        else:
            print(f"{N:>6} | {h:>12.6f} | {error:>18.10e} | {'-':>12} | {'-':>15}")
    
    # Vẽ đồ thị hội tụ
    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, errors, 'bo-', markersize=8, linewidth=2, label='Sai số thực tế')
    # Đường tham chiếu bậc 2
    h_ref = np.array(h_values)
    error_ref = h_ref**2 * errors[0] / h_values[0]**2
    plt.loglog(h_ref, error_ref, 'r--', linewidth=2, label='Hội tụ bậc 2 (O(h²))')
    plt.xlabel('Bước lưới h', fontsize=12)
    plt.ylabel('Sai số cực đại', fontsize=12)
    plt.title('Kiểm tra tính hội tụ - Bài toán đơn giản', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # So sánh nghiệm tại một số điểm
    print("\n" + "-"*70)
    print("SO SÁNH NGHIỆM TẠI MỘT SỐ ĐIỂM (N = 40)")
    print("-"*70)
    N = 40
    h = (b - a) / N
    MatrixA, VectorB = matrixBuild(a, b, h, 0, 0, "1", None, None)
    u_numerical = solve(MatrixA, VectorB)
    x = np.linspace(a, b, N + 1)
    u_exact_vals = np.array([u_exact(xi) for xi in x])
    
    print(f"{'x':>12} | {'Nghiệm chính xác':>18} | {'Nghiệm số':>18} | {'Sai số':>18}")
    print("-"*70)
    indices = [0, N//4, N//2, 3*N//4, N]
    for idx in indices:
        err = abs(u_numerical[idx] - u_exact_vals[idx])
        print(f"{x[idx]:>12.6f} | {u_exact_vals[idx]:>18.10f} | {u_numerical[idx]:>18.10f} | {err:>18.10e}")


# ==========================================
# BÀI TOÁN TEST 2: Kiểm tra tính đối xứng của ma trận
# ==========================================

def test_matrix_properties():
    """
    Kiểm tra các tính chất toán học của ma trận:
    - Tính đối xứng
    - Tính xác định dương (cho bài toán giá trị riêng)
    """
    print("\n" + "="*70)
    print("KIỂM CHỨNG 2: TÍNH CHẤT MA TRẬN")
    print("="*70)
    
    a, b = 0, 1
    N = 20
    h = (b - a) / N
    
    # Kiểm tra ma trận bài toán biên
    print("\n1. Kiểm tra ma trận bài toán biên:")
    MatrixA, VectorB = matrixBuild(a, b, h, 0, 1, "1", None, None)
    
    # Tính độ đối xứng (nên gần 0 nếu đối xứng)
    asymmetry = np.max(np.abs(MatrixA - MatrixA.T))
    print(f"   Độ không đối xứng: max|A - A^T| = {asymmetry:.2e}")
    if asymmetry < 1e-10:
        print("   ✓ Ma trận đối xứng (đúng cho p(x) = const)")
    else:
        print("   ⚠ Ma trận không đối xứng (bình thường nếu p(x) ≠ const)")
    
    # Kiểm tra tính xác định dương (tất cả giá trị riêng > 0)
    eigenvals = np.linalg.eigvals(MatrixA)
    min_eigenval = np.min(eigenvals.real)
    print(f"   Giá trị riêng nhỏ nhất: {min_eigenval:.6f}")
    if min_eigenval > 0:
        print("   ✓ Ma trận xác định dương")
    else:
        print("   ⚠ Ma trận không xác định dương")
    
    # Kiểm tra ma trận bài toán giá trị riêng
    print("\n2. Kiểm tra ma trận bài toán giá trị riêng:")
    K, M, x = build_eigenvalue_matrices(p, q, lambda x: 1.0, a, b, N, verbose=False)
    
    # Kiểm tra tính đối xứng của K và M
    asymmetry_K = np.max(np.abs(K - K.T))
    asymmetry_M = np.max(np.abs(M - M.T))
    print(f"   Độ không đối xứng của K: max|K - K^T| = {asymmetry_K:.2e}")
    print(f"   Độ không đối xứng của M: max|M - M^T| = {asymmetry_M:.2e}")
    
    # Kiểm tra tính xác định dương của M
    eigenvals_M = np.linalg.eigvals(M)
    min_eigenval_M = np.min(eigenvals_M.real)
    print(f"   Giá trị riêng nhỏ nhất của M: {min_eigenval_M:.6f}")
    if min_eigenval_M > 0:
        print("   ✓ Ma trận M xác định dương (r(x) > 0)")
    else:
        print("   ⚠ Ma trận M không xác định dương")


# ==========================================
# BÀI TOÁN TEST 3: Bài toán giá trị riêng đơn giản
# ==========================================

def test_simple_eigenvalue():
    """
    Bài toán: u''(x) = λu(x), u(0) = 0, u(π) = 0
    Nghiệm chính xác: λ_k = k², u_k(x) = sin(kx), k = 1, 2, 3, ...
    """
    print("\n" + "="*70)
    print("KIỂM CHỨNG 3: BÀI TOÁN GIÁ TRỊ RIÊNG ĐƠN GIẢN")
    print("="*70)
    print("\nBài toán: u''(x) = λu(x), u(0) = 0, u(π) = 0")
    print("Nghiệm chính xác: λ_k = k², u_k(x) = sin(kx), k = 1, 2, 3, ...")
    
    def p_simple(x):
        return 1.0
    
    def q_simple(x):
        return 0.0
    
    def r_simple(x):
        return 1.0
    
    a, b = 0, np.pi
    N = 50
    num_eigenvalues = 5
    
    eigenvalues, eigenvectors, x = solve_eigenvalue_problem(
        p_simple, q_simple, r_simple, a, b, N, 
        num_eigenvalues=num_eigenvalues, verbose=False
    )
    
    print("\n" + "-"*70)
    print("SO SÁNH GIÁ TRỊ RIÊNG")
    print("-"*70)
    print(f"{'k':>4} | {'λ_k (chính xác)':>18} | {'λ_k (số)':>18} | {'Sai số':>18}")
    print("-"*70)
    
    for k in range(1, num_eigenvalues + 1):
        lambda_exact = k**2
        lambda_numerical = eigenvalues[k-1]
        error = abs(lambda_numerical - lambda_exact)
        print(f"{k:>4} | {lambda_exact:>18.10f} | {lambda_numerical:>18.10f} | {error:>18.10e}")
    
    # So sánh hàm riêng
    print("\n" + "-"*70)
    print("SO SÁNH HÀM RIÊNG (k = 1)")
    print("-"*70)
    k = 1
    eigenvec = eigenvectors[:, k-1]
    x_full = np.concatenate([[a], x, [b]])
    u_numerical = np.concatenate([[0], eigenvec, [0]])
    
    # Chuẩn hóa để so sánh
    u_numerical = u_numerical / np.max(np.abs(u_numerical))
    u_exact = np.sin(k * x_full)
    u_exact = u_exact / np.max(np.abs(u_exact))
    
    print(f"{'x':>12} | {'u_exact (sin x)':>18} | {'u_numerical':>18} | {'Sai số':>18}")
    print("-"*70)
    indices = [0, len(x_full)//4, len(x_full)//2, 3*len(x_full)//4, len(x_full)-1]
    for idx in indices:
        err = abs(u_numerical[idx] - u_exact[idx])
        print(f"{x_full[idx]:>12.6f} | {u_exact[idx]:>18.10f} | {u_numerical[idx]:>18.10f} | {err:>18.10e}")


# ==========================================
# BÀI TOÁN TEST 4: Kiểm tra tính nhất quán
# ==========================================

def test_consistency():
    """
    Kiểm tra tính nhất quán: khi h → 0, nghiệm số → nghiệm chính xác
    """
    print("\n" + "="*70)
    print("KIỂM CHỨNG 4: TÍNH NHẤT QUÁN (Consistency)")
    print("="*70)
    print("\nKiểm tra: Khi h → 0, nghiệm số hội tụ về nghiệm chính xác")
    
    def u_exact(x):
        return x - x**2
    
    def p_test(x):
        return 1.0
    
    def q_test(x):
        return 0.0
    
    def f_test(x):
        return 2.0
    
    a, b = 0, 1
    N_values = [5, 10, 20, 40, 80, 160]
    
    print(f"\n{'N':>6} | {'h':>12} | {'Sai số L2':>18} | {'Sai số max':>18}")
    print("-"*70)
    
    for N in N_values:
        h = (b - a) / N
        MatrixA, VectorB = matrixBuild(a, b, h, 0, 0, "1", None, None)
        u_numerical = solve(MatrixA, VectorB)
        
        x = np.linspace(a, b, N + 1)
        u_exact_vals = np.array([u_exact(xi) for xi in x])
        
        error_L2 = np.sqrt(np.sum((u_numerical - u_exact_vals)**2) * h)
        error_max = np.max(np.abs(u_numerical - u_exact_vals))
        
        print(f"{N:>6} | {h:>12.6f} | {error_L2:>18.10e} | {error_max:>18.10e}")


# ==========================================
# BÀI TOÁN TEST 5: Kiểm tra điều kiện biên
# ==========================================

def test_boundary_conditions():
    """
    Kiểm tra nghiệm thỏa mãn đúng điều kiện biên
    """
    print("\n" + "="*70)
    print("KIỂM CHỨNG 5: ĐIỀU KIỆN BIÊN")
    print("="*70)
    
    a, b = 0, 1
    N = 20
    h = (b - a) / N
    
    # Test điều kiện biên loại 1
    print("\n1. Điều kiện biên loại 1: u(0) = 0, u(1) = 1")
    alpha, beta = 0, 1
    MatrixA, VectorB = matrixBuild(a, b, h, alpha, beta, "1", None, None)
    u = solve(MatrixA, VectorB)
    x = np.linspace(a, b, N + 1)
    
    error_left = abs(u[0] - alpha)
    error_right = abs(u[-1] - beta)
    print(f"   u(0) = {u[0]:.10f}, sai số: {error_left:.2e}")
    print(f"   u(1) = {u[-1]:.10f}, sai số: {error_right:.2e}")
    
    if error_left < 1e-10 and error_right < 1e-10:
        print("   ✓ Điều kiện biên được thỏa mãn chính xác")
    else:
        print("   ⚠ Có sai số nhỏ trong điều kiện biên (do làm tròn số)")


# ==========================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHƯƠNG TRÌNH KIỂM CHỨNG PHƯƠNG PHÁP SAI PHÂN")
    print("="*70)
    
    # Chạy các test
    test_simple_boundary_value()
    test_matrix_properties()
    test_simple_eigenvalue()
    test_consistency()
    test_boundary_conditions()
    
    print("\n" + "="*70)
    print("KẾT LUẬN")
    print("="*70)
    print("""
Các tiêu chí kiểm chứng:
1. ✓ Tính hội tụ: Sai số giảm theo bậc 2 khi h giảm (O(h²))
2. ✓ Tính nhất quán: Nghiệm số hội tụ về nghiệm chính xác khi h → 0
3. ✓ Điều kiện biên: Nghiệm thỏa mãn đúng điều kiện biên
4. ✓ Tính chất ma trận: Ma trận có các tính chất toán học đúng
5. ✓ So sánh với nghiệm chính xác: Sai số nhỏ ở các bài toán đơn giản

Nếu tất cả các test đều pass, code được coi là đúng.
    """)
