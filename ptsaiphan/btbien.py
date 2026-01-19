import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# ==========================================
# ĐỊNH NGHĨA CÁC HÀM p(x), q(x), f(x)
# ==========================================
def p(x):
    """Hàm p(x) trong phương trình [p(x)u'(x)]' - q(x)u(x) = -f(x)"""
    return x**2 + 1

def q(x):
    """Hàm q(x) trong phương trình [p(x)u'(x)]' - q(x)u(x) = -f(x)"""
    return 1

def f(x):
    """Hàm f(x) trong phương trình [p(x)u'(x)]' - q(x)u(x) = -f(x)"""
    return 5*x**2 +2*x + 1
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
    print("Đặt D[i] = -h^2*f(x[i])")
    
    def A(x):
        return p(x-h/2)
    def B(x):
        return p(x-h/2) + p(x+h/2) + h**2*q(x)
    def C(x):
        return p(x+h/2)
    def D(x):
        return -h**2*f(x)
    print("Phương trình trở thành A[i]*u[i-1] - B[i]*u[i] + C[i]*u[i+1] = D[i]")
    MatrixA = np.zeros((N+1, N+1))
    VectorB = np.zeros(N+1)
    print("Xử lí biên trái")    
    i = 0
    if loai == "1":
        MatrixA[i][i] = 1
        VectorB[i] = alpha
    elif loai == "2":
        print(f"Xét trường hợp tại biên trái x0 = {a} có u'({a}) = {alpha}")
        print(f"Nhân cả 2 vế với p({a})")
        print(f"Ta được p({a}) * u'({a}) = p({a}) * {alpha} = {p(a)*alpha}")
        print(f"Ta có μ1 = - p({a}) * u'({a}) = {-p(a)*alpha}")
        mu1 = -p(a)*alpha
        print(f"Ta xác định được μ1 = {mu1}")
        print(f"Sử dụng phương trình p(0.5)*u1 - [p(0.5) + h^2*q(0)/2]*u0 = -h^2*f(0)/2 -h*μ1")
        MatrixA[i][i] = -(p(a+h/2) + h**2*q(a)/2)
        MatrixA[i][i+1] = p(a+h/2)
        VectorB[i] = -h**2*f(a)/2 -h*mu1
    elif loai == "3":
        print(f"Xét trường hợp tại biên trái x0 = {a} có u'({a}) = {alpha}")
        print(f"Ta có u'({a}) - {phi1}*u({a}) = {alpha}")
        print(f"Nhân cả 2 vế với p({a})")
        print(f"p({a})*u'({a}) - p({a})*{phi1}*u({a}) = {alpha*p(a)} = -μ1")
        print(f"So sánh với dạng chuẩn p({a})*u'({a}) - φ({a})*u({a}) = {alpha*p(a)} = -μ1")
        print(f"Ta có φ({a}) = p({a})*{phi1}")
        phi = p(a)*phi1
        mu1 = -alpha * p(a)
        print(f"Ta xác định được φ({a}) = {phi}")
        print(f"Ta xác định được μ1 = {mu1}")
        print(f"Sử dụng phương trình p(0.5)*u1 - [p(0.5) + h^2*q(0)/2 + φ1]*u0 = -h^2*f(0)/2 -h*μ1")
        MatrixA[i][i] = -(p(a+h/2) + h**2*q(a)/2 + phi)
        MatrixA[i][i+1] = p(a+h/2)
        VectorB[i] = -h**2*f(a)/2 -h*mu1
    # Xử lí phần giữa
    for i in range(1, N):
        MatrixA[i][i] = -B(x[i])
        MatrixA[i][i-1] = A(x[i])
        MatrixA[i][i+1] = C(x[i])
        VectorB[i] = D(x[i])
    print("Xử lí biên phải")
    i = N
    if loai == "1":
        MatrixA[i][i] = 1
        VectorB[i] = beta
    elif loai == "2":
        print(f"Xét trường hợp tại biên phải xn = {b} có u'({b}) = {beta}")
        print(f"Nhân cả 2 vế với p({b})")
        print(f"Ta được p({b}) * u'({b}) = p({b}) * {beta} = {p(b)*beta}")
        print(f"Ta có μ2 = - p({b}) * u'({b}) = {-p(b)*beta}")
        mu2 = -p(b)*beta
        print(f"Ta xác định được μ2 = {mu2}")
        print(f"Sử dụng phương trình -p(n-0.5)*un-1 + [p(n-0.5) + h^2*q(n)/2]*un = h^2*f(n)/2 -h*μ2")
        MatrixA[i][i] = p(b-h/2) + h**2*q(b)/2
        MatrixA[i][i-1] = - p(b-h/2)
        VectorB[i] = h**2*f(b)/2 -h*mu2
    elif loai == "3":
        print(f"Xét trường hợp tại biên phải xn = {b} có u'({b}) = {beta}")
        print(f"Ta có u'({b}) - {phi2}*u({b}) = {beta}")
        print(f"Nhân cả 2 vế với p({b})")
        print(f"p({b})*u'({b}) - p({b})*{phi2}*u({b}) = {p(b)*beta} = -μ2")
        print(f"So sánh với dạng chuẩn p({b})*u'({b}) - φ({b})*u({b}) = {p(b)*beta} = -μ2")
        print(f"Ta có φ({b}) = p({b})*{phi2}")
        phi2 = p(b)*phi2
        mu2 = -beta * p(b)
        print(f"Ta xác định được φ({b}) = {phi2}")
        print(f"Ta xác định được μ2 = {mu2}")
        print(f"Sử dụng phương trình -p(n-0.5)*un-1 + [p(n-0.5) + h^2*q(n)/2 - φ2]*un = h^2*f(n)/2 -h*μ2")
        MatrixA[i][i] = p(b-h/2) + h**2*q(b)/2 - phi2
        MatrixA[i][i-1] = - p(b-h/2)
        VectorB[i] = h**2*f(b)/2 -h*mu2
    return MatrixA, VectorB

if __name__ == "__main__":
    print("="*70)
    print("CHƯƠNG TRÌNH GIẢI BÀI TOÁN BIÊN BẰNG PHƯƠNG PHÁP SAI PHÂN")
    print("="*70)
    
    a = float(input("Nhập điểm bắt đầu a: "))
    b = float(input("Nhập điểm kết thúc b: "))
    h = float(input("Nhập bước h: "))
    
    loai = input("Nhập loại điều kiện biên (1/2/3): ")
    
    if loai == "1":
        print("\nĐiều kiện biên loại 1: u(a) = α, u(b) = β")
        alpha = float(input("Nhập α (u(a)): "))
        beta = float(input("Nhập β (u(b)): "))
        phi1, phi2 = None, None
    elif loai == "2":
        print("\nĐiều kiện biên loại 2: p(a)u'(a) = -μ₁, p(b)u'(b) = -μ₂")
        print("Nhập u'(a) và u'(b):")
        alpha = float(input("Nhập u'(a): "))
        beta = float(input("Nhập u'(b): "))
        phi1, phi2 = None, None
    elif loai == "3":
        print("\nĐiều kiện biên loại 3: p(a)u'(a) - σ₁u(a) = -μ₁, p(b)u'(b) - σ₂u(b) = -μ₂")
        print("Nhập u'(a) - φ₁u(a) và u'(b) - φ₂u(b):")
        alpha = float(input("Nhập u'(a) - φ₁u(a): "))
        beta = float(input("Nhập u'(b) - φ₂u(b): "))
        phi1 = float(input("Nhập φ₁: "))
        phi2 = float(input("Nhập φ₂: "))
    else:
        print("Loại điều kiện biên không hợp lệ!")
        exit(1)
    
    MatrixA, VectorB = matrixBuild(a, b, h, alpha, beta, loai, phi1, phi2)
    
    print("\n" + "="*70)
    print("GIẢI HỆ PHƯƠNG TRÌNH")
    print("="*70)
    print(f"\nMa trận hệ số A ({MatrixA.shape[0]}x{MatrixA.shape[1]}):")
    print(MatrixA)
    print(f"\nVector vế phải B:")
    print(VectorB)
    
    # Giải hệ phương trình
    u = np.linalg.solve(MatrixA, VectorB)
    N = int((b-a)/h)
    x = np.linspace(a, b, N+1)
    
    print("\n" + "="*70)
    print("KẾT QUẢ NGHIỆM")
    print("="*70)
    print(f"\n{'i':>4} | {'x_i':>12} | {'u_i':>18}")
    print("-" * 40)
    for i in range(len(x)):
        if i < 5 or i >= len(x) - 5 or i % max(1, (N+1)//10) == 0:
            print(f"{i:>4} | {x[i]:>12.6f} | {u[i]:>18.10f}")
    
    # Tìm giá trị lớn nhất và nhỏ nhất
    u_max = np.max(u)
    u_min = np.min(u)
    idx_max = np.argmax(u)
    idx_min = np.argmin(u)
    
    print("\n" + "="*70)
    print("PHÂN TÍCH KẾT QUẢ")
    print("="*70)
    print(f"Giá trị lớn nhất: {u_max:.10f} tại x = {x[idx_max]:.6f} (i = {idx_max})")
    print(f"Giá trị nhỏ nhất: {u_min:.10f} tại x = {x[idx_min]:.6f} (i = {idx_min})")
    
    # Vẽ đồ thị
    print("\n" + "="*70)
    print("VẼ ĐỒ THỊ NGHIỆM")
    print("="*70)
    plt.figure(figsize=(10, 6))
    plt.plot(x, u, 'b-', linewidth=2, label='Nghiệm gần đúng u(x)')
    plt.plot(x, u, 'ro', markersize=4, label='Điểm lưới')
    plt.plot(x[idx_max], u_max, 'g*', markersize=15, label=f'Max: u({x[idx_max]:.4f}) = {u_max:.4f}')
    plt.plot(x[idx_min], u_min, 'm*', markersize=15, label=f'Min: u({x[idx_min]:.4f}) = {u_min:.4f}')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.title(f'Nghiệm bài toán biên - Loại {loai}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()