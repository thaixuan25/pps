import numpy as np
import matplotlib.pyplot as plt

def data():
    """
    Hàm đọc dữ liệu từ file sinx.txt
    File có định dạng: mỗi dòng chứa 2 số cách nhau bởi dấu cách (x y)
    Trả về: mảng x (hoành độ), mảng y (tung độ), n (số đoạn = số điểm - 1)
    """
    global x, y, n
    x = []
    y = []
    # Đọc từng dòng trong file
    with open('spline/input.txt', 'r+') as f:
        for line in f.readlines():
            # Tách dòng thành 2 phần: x và y
            x.append(float(line.split()[0]))
            y.append(float(line.split()[1]))
    # Chuyển sang numpy array để tính toán dễ dàng hơn
    x = np.asarray(x)
    y = np.asarray(y)
    # Sắp xếp lại cặp (x, y) theo x
    x, y = zip(*sorted(zip(x, y)))
    x = np.asarray(x)
    y = np.asarray(y)
    print("x và y sau khi sắp xếp:")
    for i in range(x.size):
        print(f"x[{i}] = {x[i]}, y[{i}] = {y[i]}")
    n = len(x)
    return x, y, n

def spline3(x, y, n, d0, dn, baccuadieukien):
    print("\n" + "="*60)
    print("PHƯƠNG PHÁP SPLINE CUBIC (HÀM GHÉP TRƠN CẤP 3)")
    print("="*60)
    
    print(f"\nCông thức Spline Cubic:")
    print(f"  Trên mỗi đoạn [xᵢ, xᵢ₊₁], hàm spline có dạng:")
    print(f"  Sᵢ(x) = aᵢ(x-xᵢ)³ + bᵢ(x-xᵢ)² + cᵢ(x-xᵢ) + dᵢ")
    print(f"  Hoặc khai triển: Sᵢ(x) = Aᵢx³ + Bᵢx² + Cᵢx + Dᵢ")
    
    print(f"\nĐiều kiện:")
    print(f"  1. Sᵢ(xᵢ) = yᵢ, Sᵢ(xᵢ₊₁) = yᵢ₊₁  (nội suy)")
    print(f"  2. S'ᵢ(xᵢ₊₁) = S'ᵢ₊₁(xᵢ₊₁)  (liên tục đạo hàm cấp 1)")
    print(f"  3. S''ᵢ(xᵢ₊₁) = S''ᵢ₊₁(xᵢ₊₁)  (liên tục đạo hàm cấp 2)")
    print(f"  4. Điều kiện biên tại x₀ và xₙ")
    
    #Tính khoảng cách giữa các điểm x
    h = np.diff(x)
    print(f"\nBước 1: Tính khoảng cách giữa các điểm")
    print(f"  hᵢ = xᵢ₊₁ - xᵢ")
    for i in range(len(h)):
        print(f"  h{i} = x{i+1} - x{i} = {x[i+1]:.6f} - {x[i]:.6f} = {h[i]:.6f}")
    
    B = np.zeros(n)
    A = np.zeros((n, n))
    ## Xây dựng ma trận A
    print(f"\nBước 2: Xây dựng hệ phương trình A*m = B")
    print(f"  Với mᵢ = S''(xᵢ) là đạo hàm cấp 2 tại điểm xᵢ")
    print(f"  Hệ phương trình có dạng:")
    print(f"    hᵢ₋₁/6 * mᵢ₋₁ + (hᵢ₋₁+hᵢ)/3 * mᵢ + hᵢ/6 * mᵢ₊₁ = (yᵢ₊₁-yᵢ)/hᵢ - (yᵢ-yᵢ₋₁)/hᵢ₋₁")
    
    i=0
    if (baccuadieukien == "2"):
        print(f"\n  Điều kiện biên cấp 2 tại điểm đầu x₀:")
        print(f"    m₀ = {d0:.6f}")
        A[i][i] = 1
        B[i] = d0
    elif (baccuadieukien == "1"):
        print(f"\n  Điều kiện biên cấp 1 tại điểm đầu x₀:")
        print(f"    S'(x₀) = {d0:.6f}")
        print(f"    h₀/3 * m₀ + h₀/6 * m₁ = (y₁-y₀)/h₀ - {d0:.6f}")
        print(f"    {h[i]/3:.6f} * m₀ + {h[i]/6:.6f} * m₁ = {(y[i+1]-y[i])/h[i]:.6f} - {d0:.6f} = {(y[i+1]-y[i])/h[i]-d0:.6f}")
        A[i][i] = h[i]/3
        A[i][i+1] = h[i]/6
        B[i] = (y[i+1]-y[i])/h[i]-d0
    else:
        print("Bậc điều kiện biên không hợp lệ")
        return
    
    print(f"\n  Các phương trình tại các điểm trong (i = 1 đến {n-2}):")
    for i in range(1, n-1):
        A[i][i-1] = h[i-1]/6
        A[i][i] = (h[i-1]+h[i])/3
        A[i][i+1] = h[i]/6
        B[i] = (y[i+1]-y[i])/h[i]- (y[i]-y[i-1])/h[i-1]
        if i <= 2 or i >= n-3:
            print(f"    i = {i}:")
            print(f"      {h[i-1]/6:.6f} * m{i-1} + {(h[i-1]+h[i])/3:.6f} * m{i} + {h[i]/6:.6f} * m{i+1} = {(y[i+1]-y[i])/h[i]:.6f} - {(y[i]-y[i-1])/h[i-1]:.6f} = {B[i]:.6f}")
    
    i = n-1
    if (baccuadieukien == "2"):
        print(f"\n  Điều kiện biên cấp 2 tại điểm cuối xₙ:")
        print(f"    mₙ = {dn:.6f}")
        A[i][i] = 1
        B[i] = dn
    elif (baccuadieukien == "1"):
        print(f"\n  Điều kiện biên cấp 1 tại điểm cuối xₙ:")
        print(f"    S'(xₙ) = {dn:.6f}")
        print(f"    hₙ₋₁/6 * mₙ₋₁ + hₙ₋₁/3 * mₙ = {dn:.6f} - (yₙ-yₙ₋₁)/hₙ₋₁")
        print(f"    {h[i-1]/6:.6f} * m{i-1} + {h[i-1]/3:.6f} * m{i} = {dn:.6f} - {(y[i]-y[i-1])/h[i-1]:.6f} = {dn- (y[i]-y[i-1])/h[i-1]:.6f}")
        A[i][i] = h[i-1]/3
        A[i][i-1] = h[i-1]/6
        B[i] = dn- (y[i]-y[i-1])/h[i-1]
    else:
        print("Bậc điều kiện biên không hợp lệ")
        return
    
    print(f"\n  Ma trận A ({n}x{n}):")
    print(A)
    print(f"\n  Vector B:")
    print(B)
    ## Giải hệ phương trình A*m = B
    print(f"\nBước 3: Giải hệ phương trình A*m = B")
    print(f"  Sử dụng phương pháp giải hệ phương trình tuyến tính")
    m = np.linalg.solve(A, B)
    print(f"  Các hệ số mᵢ (đạo hàm cấp 2 tại các điểm):")
    for i in range(len(m)):
        print(f"    m{i} = {m[i]:.10f}")
    
    #Ghép đoạn spline
    print(f"\nBước 4: Tính các hệ số của hàm spline trên mỗi đoạn")
    print(f"  Trên đoạn [xᵢ, xᵢ₊₁], hàm spline có dạng:")
    print(f"    Sᵢ(x) = aᵢ(x-xᵢ)³ + bᵢ(x-xᵢ)² + cᵢ(x-xᵢ) + dᵢ")
    print(f"  Với các hệ số:")
    print(f"    aᵢ = (mᵢ₊₁ - mᵢ) / (6 * hᵢ)")
    print(f"    bᵢ = mᵢ / 2")
    print(f"    cᵢ = (yᵢ₊₁ - yᵢ) / hᵢ - hᵢ * (2*mᵢ + mᵢ₊₁) / 6")
    print(f"    dᵢ = yᵢ")
    
    a = np.zeros(m.size-1)
    b = np.zeros(m.size-1)
    c = np.zeros(m.size-1)
    d = np.zeros(m.size-1)
    a_coeff = np.zeros(m.size-1)  # Hệ số trong dạng chuẩn
    b_coeff = np.zeros(m.size-1)
    c_coeff = np.zeros(m.size-1)
    d_coeff = np.zeros(m.size-1)
    
    for i in range(m.size-1):
        # Hệ số trong dạng chuẩn (theo t = x - x[i])
        a_i = (m[i+1] - m[i]) / (6 * h[i])
        b_i = m[i] / 2
        c_i = (y[i+1] - y[i]) / h[i] - h[i] * (2*m[i] + m[i+1]) / 6
        d_i = y[i]
        
        if i < 3 or i >= m.size-4:
            print(f"\n  Đoạn [{i}]: [x{i} = {x[i]:.6f}, x{i+1} = {x[i+1]:.6f}]")
            print(f"    a{i} = (m{i+1} - m{i}) / (6 * h{i}) = ({m[i+1]:.10f} - {m[i]:.10f}) / (6 * {h[i]:.6f}) = {a_i:.10f}")
            print(f"    b{i} = m{i} / 2 = {m[i]:.10f} / 2 = {b_i:.10f}")
            print(f"    c{i} = (y{i+1} - y{i}) / h{i} - h{i} * (2*m{i} + m{i+1}) / 6")
            print(f"         = ({y[i+1]:.6f} - {y[i]:.6f}) / {h[i]:.6f} - {h[i]:.6f} * (2*{m[i]:.10f} + {m[i+1]:.10f}) / 6")
            print(f"         = {(y[i+1] - y[i]) / h[i]:.6f} - {h[i] * (2*m[i] + m[i+1]) / 6:.10f} = {c_i:.10f}")
            print(f"    d{i} = y{i} = {d_i:.6f}")
        
        # Khai triển về dạng S(x) = a*x^3 + b*x^2 + c*x + d
        # S(x) = a_i*(x-x[i])^3 + b_i*(x-x[i])^2 + c_i*(x-x[i]) + d_i
        # Khai triển: (x-x[i])^3 = x^3 - 3*x[i]*x^2 + 3*x[i]^2*x - x[i]^3
        #            (x-x[i])^2 = x^2 - 2*x[i]*x + x[i]^2
        #            (x-x[i]) = x - x[i]
        a_coeff[i] = a_i
        b_coeff[i] = -3*a_i*x[i] + b_i
        c_coeff[i] = 3*a_i*x[i]**2 - 2*b_i*x[i] + c_i
        d_coeff[i] = -a_i*x[i]**3 + b_i*x[i]**2 - c_i*x[i] + d_i
        
        # Lưu lại để dùng sau
        a[i] = a_i
        b[i] = b_i
        c[i] = c_i
        d[i] = d_i
    
    print(f"\nBước 5: Khai triển về dạng đa thức chuẩn")
    print(f"  Sᵢ(x) = Aᵢx³ + Bᵢx² + Cᵢx + Dᵢ")
    print(f"  Với:")
    print(f"    Aᵢ = aᵢ")
    print(f"    Bᵢ = -3*aᵢ*xᵢ + bᵢ")
    print(f"    Cᵢ = 3*aᵢ*xᵢ² - 2*bᵢ*xᵢ + cᵢ")
    print(f"    Dᵢ = -aᵢ*xᵢ³ + bᵢ*xᵢ² - cᵢ*xᵢ + dᵢ")
    
    print(f"\n  Các hàm spline trên từng đoạn:")
    for i in range(m.size-1):
        if i < 3 or i >= m.size-4:
            print(f"    Đoạn [{i}]: x ∈ [{x[i]:.6f}, {x[i+1]:.6f}]")
            print(f"      S{i}(x) = {a_coeff[i]:.10f}x³ + {b_coeff[i]:.10f}x² + {c_coeff[i]:.10f}x + {d_coeff[i]:.10f}")
        else:
            print(f"    Đoạn [{i}]: x ∈ [{x[i]:.6f}, {x[i+1]:.6f}]")
            print(f"      S{i}(x) = {a_coeff[i]:.10e}x³ + {b_coeff[i]:.10e}x² + {c_coeff[i]:.10e}x + {d_coeff[i]:.10e}")
    xf = input("\nNhập giá trị x cần tìm: ")
    xf = float(xf)
    print(f"\nBước 6: Tính giá trị S(x) tại x = {xf}")
    found = False
    for i in range(m.size-1):
        if xf >= x[i] and xf <= x[i+1]:
            found = True
            print(f"  x = {xf} thuộc đoạn [{i}]: [x{i} = {x[i]:.6f}, x{i+1} = {x[i+1]:.6f}]")
            print(f"  Sử dụng hàm: S{i}(x) = {a_coeff[i]:.10f}x³ + {b_coeff[i]:.10f}x² + {c_coeff[i]:.10f}x + {d_coeff[i]:.10f}")
            
            # Tính bằng cả hai cách để kiểm tra
            value_x1 = a_coeff[i]*xf**3 + b_coeff[i]*xf**2 + c_coeff[i]*xf + d_coeff[i]
            t = xf - x[i]
            value_x2 = a[i]*t**3 + b[i]*t**2 + c[i]*t + d[i]
            
            print(f"\n  Cách 1: Dùng dạng khai triển")
            print(f"    S({xf}) = {a_coeff[i]:.10f}*{xf}³ + {b_coeff[i]:.10f}*{xf}² + {c_coeff[i]:.10f}*{xf} + {d_coeff[i]:.10f}")
            print(f"           = {a_coeff[i]:.10f}*{xf**3:.6f} + {b_coeff[i]:.10f}*{xf**2:.6f} + {c_coeff[i]:.10f}*{xf:.6f} + {d_coeff[i]:.10f}")
            print(f"           = {a_coeff[i]*xf**3:.10f} + {b_coeff[i]*xf**2:.10f} + {c_coeff[i]*xf:.10f} + {d_coeff[i]:.10f}")
            print(f"           = {value_x1:.10f}")
            
            print(f"\n  Cách 2: Dùng dạng chuẩn với t = x - x{i}")
            print(f"    t = x - x{i} = {xf:.6f} - {x[i]:.6f} = {t:.6f}")
            print(f"    S({xf}) = {a[i]:.10f}*{t:.6f}³ + {b[i]:.10f}*{t:.6f}² + {c[i]:.10f}*{t:.6f} + {d[i]:.10f}")
            print(f"           = {a[i]:.10f}*{t**3:.10f} + {b[i]:.10f}*{t**2:.10f} + {c[i]:.10f}*{t:.6f} + {d[i]:.10f}")
            print(f"           = {a[i]*t**3:.10f} + {b[i]*t**2:.10f} + {c[i]*t:.10f} + {d[i]:.10f}")
            print(f"           = {value_x2:.10f}")
            
            print(f"\n  Kết quả: S({xf}) = {value_x1:.10f}")
            break
    
    if not found:
        print(f"  x = {xf} không thuộc đoạn nào trong khoảng [{x[0]:.6f}, {x[-1]:.6f}]")

if __name__ == "__main__":
    x, y, n = data()
    baccuadieukien = input("Nhập bậc điều kiện biên (1/2): ")
    d0 = float(input("Nhập đạo hàm tại điểm đầu: "))
    dn = float(input("Nhập đạo hàm tại điểm cuối: "))
    if d0 is None:
        d0 = 0
    if dn is None:
        dn = 0
    spline3(x, y, n, d0, dn, baccuadieukien)