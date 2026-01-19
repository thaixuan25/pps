import numpy as np
import matplotlib.pyplot as plt

def hoocneNhan(A,xk):
    A.append(1)
    for i in range(len(A)-2,0,-1):
        A[i] = A[i - 1] - A[i] * xk
    A[0] = - A[0] * xk
    return A

def hoocneChia(A,xk):
    B = np.ones(len(A) - 1)
    for i in range(len(B) - 2,-1,-1):
        B[i] = A[i + 1] + B[i + 1] * xk
    return B

def init():
    global x,y,n
    x = []
    y = []
    with open('Lagrange/input.txt','r+') as f:
            for line in f.readlines():
                xt = float(line.split(' ')[0])
                yt = float(line.split(' ')[1])
                check = True
                for x_check in x:
                    if x_check == xt:
                        check = False
                        print(f"x[{xt}] da ton tai")
                        break
                if check:
                    x.append(xt)
                    y.append(yt)
                plt.scatter(xt,yt)
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

def PolyCoefficients(xt, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    yt = 0
    for i in range(o):
        yt += coeffs[i] * xt ** i
    return yt


def main():
    init()
    print("Nhap x can tinh: ", end="")
    try:
        xt = float(input())
    except:
        xt = None
    print("Bảng nội suy:")
    width = 10  # độ rộng cột 
    # In header của table
    print('-' * ((n+1)*width))
    print(f"{'':<{width}}", end="")  # cột trống đầu tiên
    for i in range(n):
        print(f"{f'x[{i}]':^{width}}", end="")
    print()
    print('-' * ((n+1)*width))
    # In dòng giá trị x
    print(f"{'x':<{width}}", end="")
    for i in range(n):
        print(f"{x[i]:^{width}.4f}", end="")
    print()
    print('-' * ((n+1)*width))
    # In các dòng giá trị hiệu (x_j - x_i), cột đầu tên x[i]
    for i in range(n):
        print(f"{x[i]:<{width}.4f}", end="")
        for j in range(n):
            if i != j:
                print(f"{(x[j] - x[i]):^{width}.4f}", end="")
            else:
                print(f"{1:^{width}.4f}", end="")
        print()
    print('-' * ((n+1)*width))

    # tinh D
    D = []
    for i in range(n):
        D.append(1)
        for j in range(n):
            if(i != j):
                D[i] *= (x[i] - x[j])
    D = np.asarray(D)
    for i in range(n):
        print(f"y[{i}]/D[{i}] = {y[i]/D[i]:^{width}.4f}")
    print('-' * ((n+1)*width))
    
    # tinh w
    print("BẢNG TÍNH TÍCH:")
    w = [1]
    table_w = []
    table_w.append([0] * (n+1-len(w)) + w[::-1])
    for i in range(n):
        w = hoocneNhan(w, x[i])
        table_w.append([0] * (n+1-len(w)) + w[::-1])
    w = np.asarray(w)
    
    # In bảng hệ số w(x) dạng table đẹp
    col_width = 12
    total_cols = n + 2  # 1 for "Bước", rest for x^i
    sep = '+' + '+'.join(['-' * col_width for _ in range(total_cols)]) + '+'

    # Header
    deg_labels = [f"x^{deg}" for deg in reversed(range(n+1))]
    header_cells = ["Bước"] + deg_labels
    header = "|" + "|".join(f"{s:^{col_width}}" for s in header_cells) + "|"

    print(sep)
    print(header)
    print(sep)

    for idx, row_coef in enumerate(table_w):
        row = f"|{idx:^{col_width}}"
        for coef in row_coef:
            row += f"|{coef:^{col_width}.4f}"
        row += "|"
        print(row)
        print(sep)
    # tinh wi
    print("\nBảng các hệ số của đa thức wi(x):")
    wi = []
    for i in range(n):
        wi.append(hoocneChia(w,x[i]))
    wi = np.asarray(wi)
    # Header
    header = f" {'':<7} |"
    for i in reversed(range(n)):
        header += f"   x^{i:<5} |"
    print(header)
    print("-" * len(header))

    # Rows
    for i in range(n):
        row_header = f"w_{i}(x)"
        row = f" {row_header:<7} |"
        for j in reversed(range(n)):
            row += f" {wi[i,j]:>9.4f} |"
        print(row)
    
    # tinh px
    px = np.zeros(n)
    for i in range(n):
        for j in range(n):
            px[i] += wi[j,i] * y[j] / D[j]

    print("Đa thức nội suy Lagrange:")
    for i in range(n):
        print(f"x^{i} = {px[i]}")
        
    if (xt != None):
        print(f"P({xt}) = {PolyCoefficients(xt,px)}")
    plt.scatter(x,y)
    
    xt = np.linspace(x[0]-0.5,x[n - 1] + 0.5,1000)
    plt.plot(xt,PolyCoefficients(xt,px))
    plt.savefig("Lagrange/mygraph.png")

    #plt.show()

if __name__=='__main__':
    main()