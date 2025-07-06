import matplotlib.backends
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
# from mpl_toolkits.mplot3d import Axes3D

def llg_equation(M, H_eff, gamma, alpha, Ms):
    """
    計算 Landau-Lifshitz-Gilbert 方程式的右側 dM/dt。
    M: 當前磁化向量 (numpy array [Mx, My, Mz])
    H_eff: 有效磁場向量 (numpy array [Hx, Hy, Hz])
    gamma: 旋磁比
    alpha: 阻尼係數
    Ms: 飽和磁化強度
    """
    
    # 將 M 正規化為單位向量 m
    m = M / np.linalg.norm(M)
    
    # LLG 方程式的顯式形式
    # 這裡假設 H_eff 已經是處理好的有效磁場
    
    # 第一項：進動項 -gamma * M x H_eff
    precession_term = -gamma * np.cross(M, H_eff)
    
    # 第二項：阻尼項 (alpha / Ms) * M x (M x H_eff)
    # 實際上，通常使用下面的顯式形式會更穩定
    # dM/dt = (-gamma / (1 + alpha**2)) * [M x H_eff + (alpha/Ms) * M x (M x H_eff)]
    
    # 為了計算方便，我們直接使用標準形式並推導出顯式表達式
    # 讓 dM_dt = A + B * dM_dt
    # 其中 A = -gamma * M x H_eff
    # B = (alpha / Ms) * M x
    # 則 dM_dt (I - B) = A
    # dM_dt = (I - B)^-1 * A
    # 但更常見和穩定的做法是直接用整理後的顯式形式
    
    dM_dt = (-gamma / (1 + alpha**2)) * (np.cross(M, H_eff) + (alpha / Ms) * np.cross(m, np.cross(M, H_eff)))
    
    return dM_dt

def runge_kutta_4th_order(func, M0, H_eff_func, gamma, alpha, Ms, t_span, dt):
    """
    使用四階 Runge-Kutta 方法求解 LLG 方程式。
    func: 描述 dM/dt 的函數 (llg_equation)
    M0: 初始磁化向量
    H_eff_func: 計算有效磁場的函數 (可隨時間或磁化變化)
    gamma: 旋磁比
    alpha: 阻尼係數
    Ms: 飽和磁化強度
    t_span: 時間範圍 [t_start, t_end]
    dt: 時間步長
    """
    t_start, t_end = t_span
    num_steps = int((t_end - t_start) / dt)
    
    t_values = np.linspace(t_start, t_end, num_steps + 1)
    M_values = np.zeros((num_steps + 1, 3))
    M_values[0] = M0
    
    for i in range(num_steps):
        t_n = t_values[i]
        M_n = M_values[i]
        
        # 計算當前的有效磁場
        H_eff_n = H_eff_func(t_n, M_n)
        
        k1 = dt * func(M_n, H_eff_n, gamma, alpha, Ms)
        
        # 為了計算 k2, k3, k4，需要更新時間和 M
        # 注意：H_eff_func 可能需要 M_n_plus_half 來計算 H_eff
        H_eff_k2 = H_eff_func(t_n + dt/2, M_n + k1/2)
        k2 = dt * func(M_n + k1/2, H_eff_k2, gamma, alpha, Ms)
        
        H_eff_k3 = H_eff_func(t_n + dt/2, M_n + k2/2)
        k3 = dt * func(M_n + k2/2, H_eff_k3, gamma, alpha, Ms)
        
        H_eff_k4 = H_eff_func(t_n + dt, M_n + k3)
        k4 = dt * func(M_n + k3, H_eff_k4, gamma, alpha, Ms)
        
        M_next = M_n + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # 確保磁化向量的長度保持不變 (這對於 LLG 非常重要)
        M_values[i+1] = M_next / np.linalg.norm(M_next) * Ms
        
    return t_values, M_values

# --- 模擬參數設定 ---
Ms = 1.0  # 飽和磁化強度 (單位化，例如 1.0 T)
gamma_0 = 2.2128e5  # 旋磁比 (rad/(A*m*s) 或 (rad/(s*Oe) if using CGS))
gamma = gamma_0  # 通常會直接使用 gamma，因為 LLG 方程式中的 gamma 已經包含了 mu_0
alpha = 0.01  # 阻尼係數

# 有效磁場函數 (這裡我們只用一個常數外部磁場作為範例)
def effective_field(t, M):
    # 這個函數可以變得非常複雜，包含交換場、各向異性場、退磁場等
    # H_ext = np.array([0.0, 0.0, 1000.0]) # 外部磁場 (例如 Oe 或 A/m)
    H_ext = np.array([0.0, 0.0, 1e5]) # 外部磁場 (A/m)
    return H_ext

# 初始磁化向量 (非零且不是沿著磁場方向，這樣才能看到進動)
M0 = np.array([0.0, Ms, 0.0]) 

# 時間參數
t_span = [0, 10e-9]  # 模擬時間從 0 到 10 納秒
dt = 1e-12           # 時間步長 1 皮秒

# 運行模擬
t_results, M_results = runge_kutta_4th_order(llg_equation, M0, effective_field, gamma, alpha, Ms, t_span, dt)

# --- 結果可視化 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(M_results[:, 0], M_results[:, 1], M_results[:, 2], label='Magnetization Trajectory')
ax.set_xlabel('Mx')
ax.set_ylabel('My')
ax.set_zlabel('Mz')
ax.set_title('Landau-Lifshitz-Gilbert Equation Simulation (RK4)')
ax.legend()
ax.grid(True)

# 設置軸的比例，確保球形軌跡可見
max_val = np.max(np.abs(M_results)) * 1.1
ax.set_xlim([-max_val, max_val])
ax.set_ylim([-max_val, max_val])
ax.set_zlim([-max_val, max_val])

plt.show()

# 繪製磁化分量隨時間的變化
plt.figure(figsize=(12, 6))
plt.plot(t_results * 1e9, M_results[:, 0], label='Mx')
plt.plot(t_results * 1e9, M_results[:, 1], label='My')
plt.plot(t_results * 1e9, M_results[:, 2], label='Mz')
plt.xlabel('Time (ns)')
plt.ylabel('Magnetization Component')
plt.title('Magnetization Components over Time')
plt.legend()
plt.grid(True)
plt.show()
