import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded, eigh
from scipy.sparse.linalg import eigsh
from scipy.optimize import fsolve
from numpy.polynomial import Polynomial

# ==========================================
# ĐỊNH NGHĨA CÁC HÀM p(x), q(x), f(x)
# ==========================================
def p(x):
    """Hàm p(x) trong phương trình [p(x)u'(x)]' - q(x)u(x) = λ*r(x)*u(x)"""
    return 1

def q(x):
    """Hàm q(x) trong phương trình [p(x)u'(x)]' - q(x)u(x) = λ*r(x)*u(x)"""
    return 0

def r(x):
    """Hàm r(x) trong phương trình [p(x)u'(x)]' - q(x)u(x) = λ*r(x)*u(x)"""
    return 1 

#==========================================
# a: điểm bắt đầu
# b: điểm kết thúc
# h: bước
# alpha: điều kiện biên trái u(a) / u'(a) / u'(a) - φ(a)*u(a)
# beta: điều kiện biên phải u(b) / u'(b) / u'(b) - φ(b)*u(b)
#==========================================
def matrixBuild(a, b, h, alpha, beta, loai, phi1, phi2):
    N = int((b-a)/h)
    x = np.linspace(a, b, N+1)
    print("Đặt A[i] = p(x[i-1/2])")
    print("Đặt B[i] = p(x[i-1/2]) + p(x[i+1/2]) + h^2*q(x[i])")
    print("Đặt C[i] = p(x[i+1/2])")
    
    def A(x):
        return p(x-h/2)/(h**2*r(x))
    def B(x):
        return p(x-h/2) + p(x+h/2)/(h**2*r(x)) + q(x)/r(x)
    def C(x):
        return p(x+h/2)/(h**2*r(x))
    print("Phương trình trở thành A[i]*u[i-1] - B[i]*u[i] + C[i]*u[i+1] = λ*r(x[i])*u(x[i])")
    MatrixA = np.zeros((N+1, N+1))
    VectorB = np.zeros(N+1)
    i = 0
    if loai == "1":
        MatrixA[i][i] = 1
        VectorB[i] = alpha
    # Xử lí phần giữa
    for i in range(1, N):
        MatrixA[i][i] = -B(x[i])
        MatrixA[i][i-1] = A(x[i])
        MatrixA[i][i+1] = C(x[i])
        VectorB[i] = r(x[i])
    i = N
    if loai == "1":
        MatrixA[i][i] = 1
        VectorB[i] = beta
    return MatrixA, VectorB

# ==========================================
# HÀM XÂY DỰNG MA TRẬN CHO BÀI TOÁN GIÁ TRỊ RIÊNG
# ==========================================
def build_eigenvalue_matrices(p_func, q_func, r_func, a, b, N, verbose=True):
    """
    Xây dựng ma trận K (stiffness) và M (mass) cho bài toán giá trị riêng:
    [p(x)u'(x)]' - q(x)u(x) = λ*r(x)*u(x)
    
    Với điều kiện biên loại 1: u(a) = 0, u(b) = 0
    
    Args:
        p_func: hàm p(x)
        q_func: hàm q(x)
        r_func: hàm r(x)
        a: điểm bắt đầu
        b: điểm kết thúc
        N: số điểm lưới (không tính biên)
        verbose: có in thông tin chi tiết không
    
    Returns:
        K: ma trận stiffness (N-1) x (N-1)
        M: ma trận mass (N-1) x (N-1)
        x: mảng các điểm lưới bên trong (không tính biên)
    """
    h = (b - a) / N
    x_full = np.linspace(a, b, N+1)
    x = x_full[1:-1]  # Chỉ lấy các điểm bên trong (loại bỏ biên)
    n = len(x)  # Số điểm bên trong
    
    K = np.zeros((n, n))
    M = np.zeros((n, n))
    
    if verbose:
        print(f"\nXây dựng ma trận cho bài toán giá trị riêng:")
        print(f"- Số điểm lưới: {N+1} (bao gồm biên)")
        print(f"- Số điểm bên trong: {n}")
        print(f"- Bước h = {h:.6f}")
    
    # Xây dựng ma trận K và M
    for i in range(n):
        xi = x[i]
        
        # Ma trận K (stiffness matrix)
        # Phần tử đường chéo: -[p(x-h/2) + p(x+h/2)]/(h^2) - q(x)
        if i > 0:
            K[i, i-1] = p_func(xi - h/2) / (h**2)
        K[i, i] = -(p_func(xi - h/2) + p_func(xi + h/2)) / (h**2) - q_func(xi)
        if i < n-1:
            K[i, i+1] = p_func(xi + h/2) / (h**2)
        
        # Ma trận M (mass matrix) - ma trận đường chéo từ r(x)
        M[i, i] = r_func(xi)
    
    if verbose:
        print(f"\nMa trận K ({K.shape[0]}x{K.shape[1]}):")
        print(K)
        print(f"\nMa trận M ({M.shape[0]}x{M.shape[1]}):")
        print(M)
    
    return K, M, x

# ==========================================
# HÀM GIẢI BÀI TOÁN GIÁ TRỊ RIÊNG
# ==========================================
def solve_eigenvalue_problem(p_func, q_func, r_func, a, b, N, num_eigenvalues=5, verbose=True):
    """
    Giải bài toán giá trị riêng: [p(x)u'(x)]' - q(x)u(x) = λ*r(x)*u(x)
    Với điều kiện biên loại 1: u(a) = 0, u(b) = 0
    
    Args:
        p_func: hàm p(x)
        q_func: hàm q(x)
        r_func: hàm r(x)
        a: điểm bắt đầu
        b: điểm kết thúc
        N: số điểm lưới
        num_eigenvalues: số giá trị riêng cần tìm
        verbose: có in thông tin chi tiết không
    
    Returns:
        eigenvalues: mảng các giá trị riêng (sắp xếp tăng dần)
        eigenvectors: ma trận các vector riêng (mỗi cột là một vector riêng)
        x: mảng các điểm lưới bên trong
    """
    # Xây dựng ma trận K và M
    K, M, x = build_eigenvalue_matrices(p_func, q_func, r_func, a, b, N, verbose=False)
    
    if verbose:
        print("\n" + "="*70)
        print("GIẢI BÀI TOÁN GIÁ TRỊ RIÊNG: K*u = λ*M*u")
        print("="*70)
    
    # Giải bài toán giá trị riêng tổng quát
    # Sử dụng scipy.linalg.eigh cho ma trận đối xứng
    eigenvalues, eigenvectors = eigh(K, M)
    
    # Sắp xếp theo giá trị riêng tăng dần
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Lấy num_eigenvalues giá trị riêng nhỏ nhất
    eigenvalues = eigenvalues[:num_eigenvalues]
    eigenvectors = eigenvectors[:, :num_eigenvalues]
    
    if verbose:
        print(f"\nTìm được {len(eigenvalues)} giá trị riêng nhỏ nhất:")
        print(f"{'STT':>4} | {'Giá trị riêng λ':>18}")
        print("-" * 30)
        for i, lam in enumerate(eigenvalues):
            print(f"{i+1:>4} | {lam:>18.10f}")
    
    return eigenvalues, eigenvectors, x

# ==========================================
# HÀM TÌM GIÁ TRỊ RIÊNG BẰNG PHƯƠNG TRÌNH ĐẶC TRƯNG
# det(A - λI) = 0
# ==========================================
def find_eigenvalues_by_determinant(MatrixA, a=None, b=None, h=None, r_func=None, num_eigenvalues=5, verbose=True):
    """
    Tìm giá trị riêng bằng cách giải phương trình đặc trưng: det(A - λ*M) = 0
    
    Phương pháp:
    1. Xây dựng ma trận M (ma trận đường chéo chứa r(x[i]))
    2. Xây dựng đa thức đặc trưng từ ma trận A và M: det(A - λ*M) = 0
    3. Tìm nghiệm của đa thức đặc trưng (đó chính là giá trị riêng)
    
    Args:
        MatrixA: ma trận đã được xây dựng từ matrixBuild
        a: điểm bắt đầu (để tính lại ma trận M)
        b: điểm kết thúc (để tính lại ma trận M)
        h: bước (để tính lại ma trận M)
        r_func: hàm r(x) (để tính lại ma trận M)
        num_eigenvalues: số giá trị riêng cần tìm
        verbose: có in thông tin chi tiết không
    
    Returns:
        eigenvalues: mảng các giá trị riêng (sắp xếp tăng dần)
        eigenvectors: ma trận các vector riêng tương ứng
    """
    if verbose:
        print("\n" + "="*70)
        print("TÌM GIÁ TRỊ RIÊNG BẰNG PHƯƠNG TRÌNH ĐẶC TRƯNG")
        print("="*70)
        print(f"Kích thước ma trận ban đầu: {MatrixA.shape}")
    
    # ==========================================
    # QUAN TRỌNG: Loại bỏ hàng đầu và hàng cuối
    # ==========================================
    # Ma trận từ matrixBuild có cấu trúc:
    # - Hàng 0 (i=0): Điều kiện biên trái (u(a) = alpha) → MatrixA[0][0] = 1
    # - Hàng 1 đến N-1: Phương trình sai phân thực sự (chứa giá trị riêng)
    # - Hàng N (i=N): Điều kiện biên phải (u(b) = beta) → MatrixA[N][N] = 1
    #
    # Để tìm giá trị riêng, ta chỉ cần ma trận con từ hàng 1 đến N-1,
    # cột 1 đến N-1 (loại bỏ biên)
    # ==========================================
    
    # Kiểm tra xem có phải điều kiện biên loại 1 không
    # (hàng đầu và hàng cuối có dạng [1, 0, 0, ...] và [..., 0, 0, 1])
    is_boundary_type1 = (MatrixA.shape[0] >= 3 and 
                         MatrixA[0, 0] == 1.0 and 
                         np.allclose(MatrixA[0, 1:], 0) and
                         MatrixA[-1, -1] == 1.0 and 
                         np.allclose(MatrixA[-1, :-1], 0))
    
    if is_boundary_type1:
        # Loại bỏ hàng đầu và hàng cuối, cột đầu và cột cuối
        MatrixA_inner = MatrixA[1:-1, 1:-1]
        if verbose:
            print(f"Phát hiện điều kiện biên loại 1")
            print(f"Loại bỏ hàng đầu và hàng cuối (điều kiện biên)")
            print(f"Ma trận con (chỉ phần giữa): {MatrixA_inner.shape}")
            print(f"  → Hàng 0 và hàng {MatrixA.shape[0]-1} đã được loại bỏ")
            print(f"  → Cột 0 và cột {MatrixA.shape[1]-1} đã được loại bỏ")
    else:
        # Nếu không phải điều kiện biên loại 1, dùng toàn bộ ma trận
        MatrixA_inner = MatrixA
        if verbose:
            print("Không phải điều kiện biên loại 1, sử dụng toàn bộ ma trận")
    
    # ==========================================
    # XÂY DỰNG MA TRẬN M (ma trận khối lượng chứa r(x[i]))
    # ==========================================
    n = MatrixA_inner.shape[0]
    
    if a is not None and b is not None and h is not None and r_func is not None:
        # Tính lại ma trận M từ các điểm lưới bên trong
        N_full = MatrixA.shape[0] - 1
        x_full = np.linspace(a, b, N_full + 1)
        if is_boundary_type1:
            # Chỉ lấy các điểm bên trong (loại bỏ biên)
            x_inner = x_full[1:-1]
        else:
            x_inner = x_full
        
        # Ma trận M là ma trận đường chéo chứa r(x[i])
        MatrixM = np.diag([r_func(xi) for xi in x_inner])
        
        if verbose:
            print(f"\nMa trận dùng để tìm giá trị riêng:")
            print(f"  - Ma trận A_inner: {MatrixA_inner.shape}")
            print(f"  - Ma trận M (đường chéo chứa r(x[i])): {MatrixM.shape}")
            print("\nPhương pháp: Giải phương trình det(A_inner - λ*M) = 0")
            print("Bước 1: Tính đa thức đặc trưng từ ma trận A_inner và M")
    else:
        # Nếu không có thông tin về r(x), dùng ma trận đơn vị (phương pháp cũ)
        MatrixM = np.eye(n)
        if verbose:
            print(f"\nMa trận dùng để tìm giá trị riêng: {MatrixA_inner.shape}")
            print("⚠ Cảnh báo: Không có thông tin về r(x), sử dụng ma trận đơn vị I")
            print("  → Giải phương trình det(A_inner - λ*I) = 0")
            print("  → Để giải đúng, cần truyền a, b, h, r_func vào hàm")
    
    # ==========================================
    # TÍNH ĐA THỨC ĐẶC TRƯNG: det(A - λ*M) = 0
    # ==========================================
    # Chuyển về dạng chuẩn: det(M^(-1)*A - λ*I) = 0
    # Hoặc giải trực tiếp bằng scipy.linalg.eig(A, M)
    # Nhưng để dùng np.poly, ta cần chuyển về dạng det(A*M^(-1) - λ*I) = 0
    try:
        # Tính M^(-1)
        M_inv = np.linalg.inv(MatrixM)
        # Chuyển về dạng chuẩn: A*M^(-1) - λ*I
        A_normalized = MatrixA_inner @ M_inv
        # Tính đa thức đặc trưng
        char_poly_coeffs = np.poly(A_normalized)
    except:
        # Nếu M không khả nghịch, dùng phương pháp khác
        if verbose:
            print("⚠ Ma trận M không khả nghịch, sử dụng phương pháp trực tiếp")
        # Giải trực tiếp bằng scipy.linalg.eig
        from scipy.linalg import eig
        eigenvalues, eigenvectors_direct = eig(MatrixA_inner, MatrixM)
        # Sắp xếp và lấy num_eigenvalues giá trị riêng nhỏ nhất
        idx = np.argsort(eigenvalues.real)
        eigenvalues = eigenvalues[idx][:num_eigenvalues]
        eigenvectors_inner = eigenvectors_direct[:, idx][:, :num_eigenvalues]
        
        # Mở rộng vector riêng về kích thước ban đầu
        if is_boundary_type1:
            eigenvectors = np.zeros((MatrixA.shape[0], len(eigenvalues)), dtype=complex)
            eigenvectors[1:-1, :] = eigenvectors_inner
            eigenvectors[0, :] = 0
            eigenvectors[-1, :] = 0
        else:
            eigenvectors = eigenvectors_inner
        
        if verbose:
            print(f"\nTìm được {len(eigenvalues)} giá trị riêng nhỏ nhất:")
            print(f"{'STT':>4} | {'Giá trị riêng λ (thực)':>25} | {'Giá trị riêng λ (ảo)':>25}")
            print("-" * 60)
            for i, lam in enumerate(eigenvalues):
                print(f"{i+1:>4} | {lam.real:>25.10f} | {lam.imag:>25.10f}")
        
        return eigenvalues, eigenvectors
    
    if verbose:
        print(f"\nĐa thức đặc trưng bậc {n}:")
        print("P(λ) = ", end="")
        for i, coeff in enumerate(char_poly_coeffs):
            if i == 0:
                print(f"{coeff:.6f}*λ^{n-i}", end="")
            elif i < len(char_poly_coeffs) - 1:
                if coeff >= 0:
                    print(f" + {coeff:.6f}*λ^{n-i}", end="")
                else:
                    print(f" - {abs(coeff):.6f}*λ^{n-i}", end="")
            else:
                if coeff >= 0:
                    print(f" + {coeff:.6f}", end="")
                else:
                    print(f" - {abs(coeff):.6f}", end="")
        print(" = 0")
    
    # Tìm nghiệm của đa thức đặc trưng (đó chính là giá trị riêng)
    if verbose:
        print("\nBước 2: Tìm nghiệm của đa thức đặc trưng")
    
    eigenvalues = np.roots(char_poly_coeffs)
    
    # Sắp xếp theo giá trị riêng tăng dần (theo phần thực)
    idx = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[idx]
    
    # Lấy num_eigenvalues giá trị riêng nhỏ nhất
    eigenvalues = eigenvalues[:num_eigenvalues]
    
    # Tính vector riêng tương ứng (chỉ cho phần ma trận bên trong)
    # Giải hệ (A_inner - λ*M)*v = 0 để tìm vector riêng
    eigenvectors_inner = np.zeros((n, len(eigenvalues)), dtype=complex)
    for i, lam in enumerate(eigenvalues):
        try:
            # Tạo ma trận (A_inner - λ*M)
            A_minus_lambdaM = MatrixA_inner - lam * MatrixM
            # Tìm vector riêng bằng cách giải hệ thuần nhất
            # Sử dụng SVD để tìm vector trong không gian null
            U, s, Vh = np.linalg.svd(A_minus_lambdaM)
            # Vector riêng là cột cuối cùng của Vh (tương ứng với giá trị kỳ dị nhỏ nhất)
            eigenvec = Vh[-1, :]
            # Chuẩn hóa theo chuẩn M: v^T * M * v = 1
            norm_M = np.sqrt(np.abs(eigenvec.conj() @ MatrixM @ eigenvec))
            if norm_M > 1e-10:
                eigenvec = eigenvec / norm_M
            else:
                eigenvec = eigenvec / np.linalg.norm(eigenvec)
            eigenvectors_inner[:, i] = eigenvec
        except:
            # Nếu không tìm được, dùng phương pháp trực tiếp
            try:
                from scipy.linalg import eig
                eigenvalues_direct, eigenvectors_direct = eig(MatrixA_inner, MatrixM)
                idx_closest = np.argmin(np.abs(eigenvalues_direct - lam))
                eigenvec = eigenvectors_direct[:, idx_closest]
                # Chuẩn hóa
                norm_M = np.sqrt(np.abs(eigenvec.conj() @ MatrixM @ eigenvec))
                if norm_M > 1e-10:
                    eigenvec = eigenvec / norm_M
                else:
                    eigenvec = eigenvec / np.linalg.norm(eigenvec)
                eigenvectors_inner[:, i] = eigenvec
            except:
                # Phương pháp cuối cùng: dùng eigvals
                eigenvalues_direct, eigenvectors_direct = np.linalg.eig(MatrixA_inner)
                idx_closest = np.argmin(np.abs(eigenvalues_direct - lam))
                eigenvectors_inner[:, i] = eigenvectors_direct[:, idx_closest]
    
    # Mở rộng vector riêng về kích thước ban đầu (thêm 0 ở đầu và cuối cho biên)
    if is_boundary_type1:
        eigenvectors = np.zeros((MatrixA.shape[0], len(eigenvalues)), dtype=complex)
        eigenvectors[1:-1, :] = eigenvectors_inner
        # Phần biên = 0 (điều kiện biên đồng nhất)
        eigenvectors[0, :] = 0
        eigenvectors[-1, :] = 0
    else:
        eigenvectors = eigenvectors_inner
    
    if verbose:
        print(f"\nTìm được {len(eigenvalues)} giá trị riêng nhỏ nhất:")
        print(f"{'STT':>4} | {'Giá trị riêng λ (thực)':>25} | {'Giá trị riêng λ (ảo)':>25}")
        print("-" * 60)
        for i, lam in enumerate(eigenvalues):
            print(f"{i+1:>4} | {lam.real:>25.10f} | {lam.imag:>25.10f}")
        
        print("\nBước 3: Kiểm tra lại bằng cách tính det(A_inner - λ*M):")
        print(f"{'STT':>4} | {'λ':>25} | {'det(A_inner - λ*M)':>25}")
        print("-" * 60)
        for i, lam in enumerate(eigenvalues):
            det_value = np.linalg.det(MatrixA_inner - lam * MatrixM)
            print(f"{i+1:>4} | {lam.real:>25.10f} | {det_value:>25.10e}")
    
    return eigenvalues, eigenvectors

# ==========================================
# HÀM TÌM GIÁ TRỊ RIÊNG TỪ MATRIXA ĐÃ CÓ (PHƯƠNG PHÁP TRỰC TIẾP)
# ==========================================
def find_eigenvalues_from_matrix(MatrixA, num_eigenvalues=5, verbose=True):
    """
    Tìm giá trị riêng từ ma trận MatrixA đã được xây dựng.
    Sử dụng phương pháp trực tiếp: np.linalg.eig()
    
    Args:
        MatrixA: ma trận đã được xây dựng
        num_eigenvalues: số giá trị riêng cần tìm
        verbose: có in thông tin chi tiết không
    
    Returns:
        eigenvalues: mảng các giá trị riêng
        eigenvectors: ma trận các vector riêng
    """
    if verbose:
        print("\n" + "="*70)
        print("TÌM GIÁ TRỊ RIÊNG TỪ MA TRẬN A (PHƯƠNG PHÁP TRỰC TIẾP)")
        print("="*70)
        print(f"Kích thước ma trận: {MatrixA.shape}")
    
    # Giải bài toán giá trị riêng: A*u = λ*u
    eigenvalues, eigenvectors = np.linalg.eig(MatrixA)
    
    # Sắp xếp theo giá trị riêng tăng dần (theo phần thực)
    idx = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Lấy num_eigenvalues giá trị riêng nhỏ nhất
    eigenvalues = eigenvalues[:num_eigenvalues]
    eigenvectors = eigenvectors[:, :num_eigenvalues]
    
    if verbose:
        print(f"\nTìm được {len(eigenvalues)} giá trị riêng nhỏ nhất:")
        print(f"{'STT':>4} | {'Giá trị riêng λ (thực)':>25} | {'Giá trị riêng λ (ảo)':>25}")
        print("-" * 60)
        for i, lam in enumerate(eigenvalues):
            print(f"{i+1:>4} | {lam.real:>25.10f} | {lam.imag:>25.10f}")
    
    return eigenvalues, eigenvectors

if __name__ == "__main__":
    print("="*70)
    print("CHƯƠNG TRÌNH GIẢI BÀI TOÁN GIÁ TRỊ RIÊNG BẰNG PHƯƠNG PHÁP SAI PHÂN")
    print("="*70)
    
    a = float(input("Nhập điểm bắt đầu a: "))
    b = float(input("Nhập điểm kết thúc b: "))
    h = float(input("Nhập bước h: "))
    
        alpha = float(input("Nhập α (u(a)): "))
        beta = float(input("Nhập β (u(b)): "))
        phi1, phi2 = None, None
    MatrixA, VectorB = matrixBuild(a, b, h, alpha, beta, "1", phi1, phi2)
    print(MatrixA)
    # Hỏi người dùng có muốn tìm giá trị riêng không
    tim_gia_tri_rieng = 'y'
    
    if tim_gia_tri_rieng == 'y':
        num_eigenvalues = MatrixA.shape[0]
        print("\nChọn phương pháp tìm giá trị riêng:")
        print("1. Bằng phương trình đặc trưng det(A - λI) = 0 (từ ma trận A đã xây dựng)")
        print("2. Phương pháp trực tiếp np.linalg.eig()")
        print("3. Phương pháp Sturm-Liouville (K*u = λ*M*u)")
        phuong_phap = input("Chọn phương pháp (1/2/3, mặc định 1): ") or "1"
        
        if phuong_phap == "1":
            # Sử dụng phương trình đặc trưng: det(A - λ*M) = 0
            # Truyền thêm a, b, h, r để tính ma trận M chứa r(x[i])
            eigenvalues, eigenvectors = find_eigenvalues_by_determinant(
                MatrixA, a=a, b=b, h=h, r_func=r, 
                num_eigenvalues=num_eigenvalues, verbose=True
            )
        elif phuong_phap == "2":
            # Phương pháp trực tiếp
            eigenvalues, eigenvectors = find_eigenvalues_from_matrix(MatrixA, num_eigenvalues=num_eigenvalues, verbose=True)
    else:
            # Phương pháp Sturm-Liouville
            eigenvalues, eigenvectors, x_eig = solve_eigenvalue_problem(p, q, r, a, b, int((b-a)/h), num_eigenvalues=num_eigenvalues, verbose=True)
    
        # Vẽ đồ thị các vector riêng
    print("\n" + "="*70)
        print("VẼ ĐỒ THỊ CÁC VECTOR RIÊNG")
    print("="*70)
    N = int((b-a)/h)
    x = np.linspace(a, b, N+1)
    
        num_plot = min(3, len(eigenvalues))  # Vẽ tối đa 3 vector riêng đầu tiên
        plt.figure(figsize=(12, 8))
        for i in range(num_plot):
            plt.subplot(num_plot, 1, i+1)
            eigenvec = eigenvectors[:, i]
            # Chuẩn hóa vector riêng
            eigenvec = eigenvec / np.max(np.abs(eigenvec))
            plt.plot(x, eigenvec, 'b-', linewidth=2, label=f'Vector riêng {i+1} (λ = {eigenvalues[i]:.6f})')
            plt.plot(x, eigenvec, 'ro', markersize=3)
            plt.xlabel('x', fontsize=10)
            plt.ylabel(f'u_{i+1}(x)', fontsize=10)
            plt.title(f'Vector riêng thứ {i+1} - λ_{i+1} = {eigenvalues[i]:.10f}', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    