
#import library
import numpy as np
import matplotlib.pyplot as plt
import math

#load input
x=[]
y=[]
check=True
with open ('binhphuongtoithieu/input.txt','r') as f:
  for line in f: 
    #print(line)
    try:
      (_x,_y)=line.split()
      x.append(float(_x))
      y.append(float(_y))
    except ValueError:
      check=False
      print("Lỗi input")
#check input
if check:
  if len(x)==len(y):
    print("Input thỏa mãn")
  else:
    print("Kiểm tra Input")

#plot input (x,y)
plt.plot(x,y,'*')
plt.scatter(x,y)

#define function
def phi0(x):
  return 1
def phi1(x):
  return x**2
def phi2(x):
  return 1/x
def phi3(x):
  return x**3
def phi4(x):
  return x**4
def phi5(x):
  return math.sin(x)
def phi6(x):
  return math.cos(x)
def phi7(x):
  return math.sin(2*x)
def phi8(x):
  return math.cos(2*x)
def phi9(x):
  return math.exp(x)
def phi10(x):
  return math.exp(-x)

#tra ve ma tran sau khi thay cac x vao hàm số phi  
def pack1(phi,x):
    result=[]
    for i in range(len(phi)):
        temp=list(map(phi[i],x))
        result.append(temp)
    return np.array(result).T

def pack4(x) :
    change=[]
    for i in range(len(x)):
        temp1=math.log10(x[i])
        change.append(temp1)
    return np.array(change)

#nhân ma trận nghich dao voi ma tran theta 
def pack2(theta):
    return theta.T@theta

#Dùng viền quanh để tính nghịch đảo của M    
def vienquanh_inverse(A):
    n,_=A.shape
    if n==1:
        return 1/A
    elif n>1:
        start=1/A[0,0]
        for i in range(n-1):
            alpha11=start
            alpha12=A[:(i+1),i+1].reshape(i+1,1)
            alpha21=A[i+1,:i+1]
            alpha22=A[i+1,i+1]
            if i==0:
                X=alpha11*alpha12
            else :
                X=alpha11@alpha12
            if i==0:
                Y=alpha21*alpha11
            else :
                Y=alpha21@alpha11
                Y=Y.reshape(1,-1)
            if i==0:
                theta=alpha22-Y*alpha12
            else :
                theta=alpha22-Y@alpha12           
            if i==0:
                beta11=alpha11+(1/theta)*(X*Y)
            else :
                beta11=alpha11+(1/theta)*(X@Y)            
            beta12=-(1/theta)*X
            beta21=-(1/theta)*Y
            beta22=1/theta
            tempt_result=np.vstack((np.hstack((beta11,beta12)),np.hstack((beta21,beta22))))
            start=tempt_result 
        return tempt_result

#Tính giá trị của ma trận hệ số
def pack3(theta,M,y):
    return vienquanh_inverse(M)@theta.T@y

#Nhập dạng phương trình muốn xuất
phi=[phi0,phi1,phi2]

print("\n" + "="*60)
print("PHƯƠNG PHÁP BÌNH PHƯƠNG TỐI THIỂU (LEAST SQUARES)")
print("="*60)

print(f"\nMục tiêu: Tìm hàm xấp xỉ f(x) = Σᵢ₌₀ᵐ aᵢφᵢ(x) sao cho")
print(f"  min Σⱼ₌₀ⁿ (yⱼ - f(xⱼ))²")
print(f"  với n = {len(x)} điểm dữ liệu, m = {len(phi)} hàm cơ sở")

print(f"\nCác hàm cơ sở được chọn:")
for i in range(len(phi)):
    func_name = phi[i].__name__
    if func_name == 'phi0':
        print(f"  φ{i}(x) = 1")
    elif func_name == 'phi1':
        print(f"  φ{i}(x) = x²")
    elif func_name == 'phi2':
        print(f"  φ{i}(x) = 1/x")
    elif func_name == 'phi3':
        print(f"  φ{i}(x) = x³")
    elif func_name == 'phi4':
        print(f"  φ{i}(x) = x⁴")
    elif func_name == 'phi5':
        print(f"  φ{i}(x) = sin(x)")
    elif func_name == 'phi6':
        print(f"  φ{i}(x) = cos(x)")
    elif func_name == 'phi7':
        print(f"  φ{i}(x) = sin(2x)")
    elif func_name == 'phi8':
        print(f"  φ{i}(x) = cos(2x)")
    elif func_name == 'phi9':
        print(f"  φ{i}(x) = eˣ")
    elif func_name == 'phi10':
        print(f"  φ{i}(x) = e⁻ˣ")

print(f"\nBước 1: Xây dựng ma trận theta (ma trận các hàm cơ sở)")
print(f"  theta[i,j] = φⱼ(xᵢ)")
print(f"  Kích thước: {len(x)} x {len(phi)}")

theta=pack1(phi,x)


print(f"\nBước 2: Tính ma trận M = theta^T * theta")
print(f"  M là ma trận vuông {len(phi)} x {len(phi)}")
print(f"  M[i,j] = Σₖ₌₀ⁿ⁻¹ theta[k,i] * theta[k,j] = Σₖ₌₀ⁿ⁻¹ φᵢ(xₖ) * φⱼ(xₖ)")

M=pack2(theta)
print(f"\n  Ma trận M:")
print(M)

print(f"\nBước 3: Tính nghịch đảo của ma trận M")
print(f"  Sử dụng phương pháp viền quanh (bordering method)")
print(f"  Phương pháp này tính M^(-1) từng bước bằng cách thêm từng hàng/cột")
print(f"  (M^(-1) sẽ được tính trong bước tiếp theo)")

# Tính theta^T * y trước để hiển thị
theta_T_y = theta.T @ np.array(y)
print(f"\n  Tính theta^T * y (vector {len(phi)} x 1):")
print(f"    theta^T * y = {theta_T_y}")

print(f"\nBước 4: Tính vector hệ số a = M^(-1) * theta^T * y")
print(f"  a = M^(-1) * theta^T * y")
print(f"  Trong đó:")
print(f"    - M^(-1) được tính bằng phương pháp viền quanh (trong hàm pack3)")
print(f"    - theta^T * y = {theta_T_y}")
print(f"    - a là vector hệ số {len(phi)} x 1")

a=pack3(theta,M,y)
print(f"\n  Vector hệ số a:")
for i in range(len(a)):
    print(f"    a{i} = {a[i]:.10f}")

print(f"\nBước 5: Viết hàm xấp xỉ")
print(f"  f(x) = ", end="")
terms = []
for i in range(len(phi)):
    func_name = phi[i].__name__
    if func_name == 'phi0':
        term = f"{a[i]:.10f}"
    elif func_name == 'phi1':
        term = f"{a[i]:.10f} * x²"
    elif func_name == 'phi2':
        term = f"{a[i]:.10f} / x"
    elif func_name == 'phi3':
        term = f"{a[i]:.10f} * x³"
    elif func_name == 'phi4':
        term = f"{a[i]:.10f} * x⁴"
    elif func_name == 'phi5':
        term = f"{a[i]:.10f} * sin(x)"
    elif func_name == 'phi6':
        term = f"{a[i]:.10f} * cos(x)"
    elif func_name == 'phi7':
        term = f"{a[i]:.10f} * sin(2x)"
    elif func_name == 'phi8':
        term = f"{a[i]:.10f} * cos(2x)"
    elif func_name == 'phi9':
        term = f"{a[i]:.10f} * eˣ"
    elif func_name == 'phi10':
        term = f"{a[i]:.10f} * e⁻ˣ"
    else:
        term = f"{a[i]:.10f} * φ{i}(x)"
    terms.append(term)
print(" + ".join(terms))

def find_y(x,u,a):
  y=0
  for i in range (0,len(u)):
    y=y+a[i]*phi[i](x)
  return y

# Tính toán các loại sai số
def tinh_sai_so(x_thuc, y_thuc, phi, a):
    """
    Tính các loại sai số giữa giá trị thực tế và giá trị dự đoán
    """
    n = len(x_thuc)
    y_du_doan = []
    
    # Tính giá trị dự đoán tại các điểm x
    for i in range(n):
        y_pred = find_y(x_thuc[i], phi, a)
        y_du_doan.append(y_pred)
    
    y_thuc = np.array(y_thuc)
    y_du_doan = np.array(y_du_doan)
    
    # SSE - Tổng bình phương sai số (Sum of Squared Errors)
    sse = np.sum((y_thuc - y_du_doan)**2)
    
    # MSE - Sai số bình phương trung bình (Mean Squared Error)
    mse = sse / n
    
    # RMSE - Căn bậc hai của MSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    
    # MAE - Sai số tuyệt đối trung bình (Mean Absolute Error)
    mae = np.mean(np.abs(y_thuc - y_du_doan))
    
    # R² - Hệ số xác định (R-squared)
    y_trung_binh = np.mean(y_thuc)
    ss_tot = np.sum((y_thuc - y_trung_binh)**2)  # Tổng bình phương tổng thể
    r_squared = 1 - (sse / ss_tot) if ss_tot != 0 else 0
    
    return {
        'SSE': sse,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r_squared,
        'y_thuc': y_thuc,
        'y_du_doan': y_du_doan
    }

# Tính và in ra sai số
print(f"\nBước 6: Tính các loại sai số")
print(f"  Đánh giá độ chính xác của hàm xấp xỉ")

sai_so = tinh_sai_so(x, y, phi, a)

print(f"\n  Công thức các sai số:")
print(f"    SSE = Σⱼ₌₀ⁿ⁻¹ (yⱼ - f(xⱼ))²")
print(f"    MSE = SSE / n")
print(f"    RMSE = √MSE")
print(f"    MAE = (1/n) * Σⱼ₌₀ⁿ⁻¹ |yⱼ - f(xⱼ)|")
print(f"    R² = 1 - SSE / SS_tot")
print(f"        với SS_tot = Σⱼ₌₀ⁿ⁻¹ (yⱼ - ȳ)², ȳ = (1/n) * Σⱼ₌₀ⁿ⁻¹ yⱼ")

print('\n' + '='*60)
print('KẾT QUẢ ĐÁNH GIÁ SAI SỐ:')
print('='*60)

# Tính chi tiết SSE
y_du_doan = sai_so['y_du_doan']
y_thuc = sai_so['y_thuc']
sse_chi_tiet = np.sum((y_thuc - y_du_doan)**2)
print(f"\n  Tính SSE:")
print(f"    SSE = Σⱼ₌₀ⁿ⁻¹ (yⱼ - f(xⱼ))²")
print(f"         = ", end="")
if len(x) <= 10:
    terms_sse = [f"({y_thuc[i]:.6f} - {y_du_doan[i]:.6f})²" for i in range(len(x))]
    print(" + ".join(terms_sse))
else:
    terms_sse = [f"({y_thuc[i]:.6f} - {y_du_doan[i]:.6f})²" for i in range(3)]
    print(" + ".join(terms_sse) + " + ... + ", end="")
    terms_sse_end = [f"({y_thuc[i]:.6f} - {y_du_doan[i]:.6f})²" for i in range(len(x)-3, len(x))]
    print(" + ".join(terms_sse_end))
print(f"         = {sai_so['SSE']:.10f}")

y_trung_binh = np.mean(y_thuc)
print(f"\n  Tính R²:")
print(f"    ȳ = (1/n) * Σⱼ₌₀ⁿ⁻¹ yⱼ = {y_trung_binh:.6f}")
ss_tot = np.sum((y_thuc - y_trung_binh)**2)
print(f"    SS_tot = Σⱼ₌₀ⁿ⁻¹ (yⱼ - ȳ)² = {ss_tot:.10f}")
print(f"    R² = 1 - SSE / SS_tot = 1 - {sai_so['SSE']:.10f} / {ss_tot:.10f} = {sai_so['R²']:.10f}")

print(f'\n  SSE (Tổng bình phương sai số): {sai_so["SSE"]:.10f}')
print(f'  MSE (Sai số bình phương trung bình): {sai_so["MSE"]:.10f}')
print(f'  RMSE (Căn bậc hai của MSE): {sai_so["RMSE"]:.10f}')
print(f'  MAE (Sai số tuyệt đối trung bình): {sai_so["MAE"]:.10f}')
print(f'  R² (Hệ số xác định): {sai_so["R²"]:.10f}')
print('='*60)

# In ra bảng so sánh giá trị thực và dự đoán
print('\nBước 7: Bảng so sánh giá trị thực tế và dự đoán')
print('-'*70)
print(f"{'x':>10} | {'y thực tế':>15} | {'y dự đoán':>15} | {'Sai số':>15} | {'(y-y_pred)²':>15}")
print('-'*70)
for i in range(len(x)):
    sai_so_tung_diem = abs(sai_so['y_thuc'][i] - sai_so['y_du_doan'][i])
    sai_so_binh_phuong = (sai_so['y_thuc'][i] - sai_so['y_du_doan'][i])**2
    if i < 5 or i >= len(x) - 5:
        print(f'{x[i]:>10.4f} | {sai_so["y_thuc"][i]:>15.6f} | {sai_so["y_du_doan"][i]:>15.6f} | {sai_so_tung_diem:>15.6f} | {sai_so_binh_phuong:>15.10f}')
    elif i == 5:
        print(f"{'...':>10} | {'...':>15} | {'...':>15} | {'...':>15} | {'...':>15}")
print('-'*70)

def graph(x,y,phi,a):
  x_test=np.linspace(min(x),max(x),100000)
  y_test=[]
  for i in range (0,len(x_test)):
    y_test.append(find_y(x_test[i],phi,a))
  plt.scatter(x,y,s=30,cmap='palete')
  plt.plot(x_test,y_test,'r')


graph(x,y,phi,a)
plt.show()



