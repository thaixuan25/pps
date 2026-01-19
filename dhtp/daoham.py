import math
import sys
import io

# Cấu hình encoding UTF-8 cho output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def print_formulas():
    """ In các công thức quan trọng """
    print("="*60)
    print("CÁC CÔNG THỨC SỬ DỤNG:")
    print("="*60)
    print("1. Bước nhảy h:")
    print("   h = (x[n] - x[0]) / n")
    print()
    print("2. Biến đổi t:")
    print("   t = (a - x[0]) / h")
    print()
    print("3. Đa thức nội suy Newton:")
    print("   P(t) = Σ [y[i] * (-1)^(n-i) / (i! * (n-i)!) * A/(t-i)]")
    print("   với A = (t-0)(t-1)...(t-n)")
    print()
    print("4. Đạo hàm cấp 1:")
    print("   f'(a) ≈ (1/h) * P'(t)")
    print("   P'(t) = Σ [Pt[i] * (n-i) * t^(n-i-1)]")
    print("="*60)
    print()

def Input():
    """ Nhập bảng số """
    global x, y, n, a, h, t
    x = []
    y = []
    with open('dhtp/input.txt','r+') as f:                   
        for line in f:
            x.append(float(line.split()[0]))
            y.append(float(line.split()[1]))
    n = len(x)-1
    h = (x[n]-x[0])/n
    a = float(input("Nhập điểm cần tính đạo hàm: "))
    t = (a - x[0])/h
    
    print("="*60)
    print("CÁC BƯỚC TÍNH TOÁN:")
    print("="*60)
    print(f"Bước 1: Đọc dữ liệu từ file")
    print(f"   - Số điểm n = {n}")
    print(f"   - Điểm cần tính đạo hàm: a = {a}")
    print(f"   - Bước nhảy h = (x[{n}] - x[0]) / {n} = ({x[n]} - {x[0]}) / {n} = {h:.6f}")
    print(f"   - Biến đổi t = (a - x[0]) / h = ({a} - {x[0]}) / {h:.6f} = {t:.6f}")
    print()

def multiply_horner(A, i) -> list:
    """ Nhân một đa thức với (t-i) """ 
    A.append(0)
    for j in range(len(A)-1,0,-1):
        A[j] = A[j] - A[j - 1] * i
    return A

def devide_horner(A, i) -> list:
    """ Chia một đa thức với (t-i) """
    for j in range(1, len(X)):
        X[j] = i*X[j-1] + A[j]
    return X

def P_t() -> list:
    """ Tính P(t) """
    print(f"Bước 2: Tính đa thức A = (t-0)(t-1)...(t-{n})")
    print(f"   A = {[round(a_val, 6) for a_val in A]}")
    print()
    print(f"Bước 3: Tính đa thức nội suy P(t)")
    Pt = [0]*(n+1)
    for i in range(n+1):
        D = devide_horner(A, i)
        print(f"   - Với i = {i}: A/(t-{i}) = {[round(d_val, 6) for d_val in D]}")
        for j in range(n+1):
            Pt[j] = Pt[j] + D[j]*((-1)**(n-i))/(math.factorial(i)*math.factorial(n-i))*y[i]
    print(f"   P(t) = {[round(p, 6) for p in Pt]}")
    print()
    return Pt

def deri_approx(Pt):
    """ Tính gần đúng đạo hàm cấp 1 """
    print(f"Bước 4: Tính đạo hàm P'(t)")
    print("   P'(t) = Σ [Pt[i] * (n-i) * t^(n-i-1)]")
    P_prime_terms = []
    ans = 0
    for i in range(n):
        term = Pt[i]*(n-i)*(t**(n-i-1))
        P_prime_terms.append(term)
        ans = ans + (1/h)*term
        print(f"   - Pt[{i}] * {n-i} * {t:.6f}^({n-i-1}) = {Pt[i]:.6f} * {n-i} * {t**(n-i-1):.6f} = {term:.6f}")
    print(f"   P'(t) = {sum(P_prime_terms):.6f}")
    print()
    print(f"Bước 5: Tính đạo hàm tại điểm a")
    print(f"   f'({a}) = (1/h) * P'(t) = (1/{h:.6f}) * {sum(P_prime_terms):.6f}")
    print("="*60)
    print(f"KẾT QUẢ: f'({a}) ≈ {ans:.6f}")
    print("="*60)
    
# In các công thức
print_formulas()

Input()

A = [1]
for i in range(0, n+1):
    A = multiply_horner(A, i)   # Mảng chứa tích các (t-i)

X = [1]*(n+1)   # Tạo mảng lưu giá trị phép chia A cho (t-i)

Pt = P_t()
deri_approx(Pt) 