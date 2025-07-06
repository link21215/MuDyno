import numpy as np

def runge_kutta_4th_order(llg_equation, m0, H_eff_func, gamma, alpha, Ms, t_span, dt):
    """
    使用四階 Runge-Kutta 方法求解 LLG 方程式。
    llg_equation: 描述 dM/dt 的函數 (llg_equation)
    m0: 初始磁化向量
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
    m_values = np.zeros((num_steps + 1, 3))
    m_values[0] = m0
    
    for i in range(num_steps):
        t_n = t_values[i]
        m_n = m_values[i]
        
        # 計算當前的有效磁場
        H_eff_n = H_eff_func(t_n, m_n)
        
        k1 = dt * llg_equation(m_n, H_eff_n, gamma, alpha)
        
        # 為了計算 k2, k3, k4，需要更新時間和 M
        # 注意：H_eff_func 可能需要 M_n_plus_half 來計算 H_eff
        H_eff_k2 = H_eff_func(t_n + dt/2, m_n + k1/2)
        k2 = dt * llg_equation(m_n + k1/2, H_eff_k2, gamma, alpha)
        
        H_eff_k3 = H_eff_func(t_n + dt/2, m_n + k2/2)
        k3 = dt * llg_equation(m_n + k2/2, H_eff_k3, gamma, alpha)
        
        H_eff_k4 = H_eff_func(t_n + dt, m_n + k3)
        k4 = dt * llg_equation(m_n + k3, H_eff_k4, gamma, alpha)
        
        m_next = m_n + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # 確保磁化向量的長度保持不變 (這對於 LLG 非常重要)
        m_values[i+1] = m_next / np.linalg.norm(m_next)
        
    return t_values, m_values * Ms

def runge_kutta_fehlberg_45(llg_equation, m0, H_eff_func, gamma, alpha, Ms, t_span, dt_initial, atol=1e-6, rtol=1e-6):
    """
    使用自適應步長的 Runge-Kutta-Fehlberg (RKF45) 方法求解 LLG 方程式。
    llg_equation: 描述 dM/dt 的函數 (llg_equation)
    M0: 初始磁化向量
    H_eff_func: 計算有效磁場的函數 (可隨時間或磁化變化)
    gamma: 旋磁比
    alpha: 阻尼係數
    Ms: 飽和磁化強度
    t_span: 時間範圍 [t_start, t_end]
    dt_initial: 初始時間步長
    atol: 絕對誤差容忍度
    rtol: 相對誤差容忍度
    """
    if m0.dtype == np.int_:
        m0 = m0.astype(np.float64)
    t_start, t_end = t_span
    
    t_values = [t_start]
    m_values = [m0]
    
    current_t = t_start
    current_m = m0
    dt = dt_initial # 當前時間步長
    
    # Butcher Tableau 係數 for RKF45 (Cash-Karp)
    # c 係數
    c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    # a 係數 (嵌套列表，方便索引)
    a = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/4, 0, 0, 0, 0, 0],
        [3/32, 9/32, 0, 0, 0, 0],
        [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
        [439/216, -8, 3680/513, -845/4104, 0, 0],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]
    ], dtype = np.float64)
    # b 係數 for 4th order solution (y_n+1)
    b4 = np.array([25/216, 0, 1408/2565, 2197/4100, -1/5, 0])
    # b* 係數 for 5th order solution (y*_n+1)
    b5 = np.array([16/135, 0, 6656/12825, 28561/56430, -11/40, 2/55])

    while current_t < t_end:
        # 確保不會跨過 t_end
        if current_t + dt > t_end:
            dt = t_end - current_t
        
        # 計算 k_i
        k = np.zeros((6, len(current_m))) # 6個 k_i 向量，每個向量3個分量 (Mx, My, Mz)
        
        # k1
        H_eff_k1 = H_eff_func(current_t, M = current_m)
        k[0] = dt * llg_equation(current_m, H_eff_k1, gamma, alpha)
        
        # k2 到 k6
        for i in range(1, 6):
            m_intermediate = current_m.copy()
            for j in range(i):
                m_intermediate += a[i, j] * k[j]
            
            H_eff_ki = H_eff_func(current_t + c[i] * dt, M = m_intermediate)
            k[i] = dt * llg_equation(m_intermediate, H_eff_ki, gamma, alpha)
            
        # 計算四階 (M_next_4) 和五階 (M_next_5) 估計
        m_next_4 = current_m + np.dot(b4, k)
        m_next_5 = current_m + np.dot(b5, k)
        
        # 誤差估計
        error_vector = m_next_5 - m_next_4
        # 為了計算誤差，我們需要參考標度。對於向量，通常用向量的範數。
        # scale = atol + rtol * np.linalg.norm(current_M) # 這裡使用當前 M 的範數作為參考
        # 更穩健的誤差估計，考慮各分量
        scale = atol + rtol * np.linalg.norm(m_next_4) # 或 M_next_5
        
        error_magnitude = np.linalg.norm(error_vector)
        
        # 新步長控制
        # 建議的步長調整因子。Safety factor 0.9 通常用於防止過度自信的步長增加。
        # power 1/5 for 5th order method. For RKF45, we use 1/4 or 1/5 depending on context,
        # but 1/5 is more common for the fifth-order estimate.
        # However, the classic formula uses power 1/4 based on the error of the lower order method.
        # Let's use 1/5 for the fifth-order method's error control.
        
        # Ensure error_magnitude is not zero to avoid division by zero or log(0) issues
        if error_magnitude == 0:
            dt_new = dt * 1.5 # If error is effectively zero, just increase step size safely
        else:
            # 調整步長因子，限制在一定範圍內避免過大或過小
            factor = 0.9 * (scale / error_magnitude)**(0.2) # (1/5 power)
            factor = np.clip(factor, 0.1, 5.0) # 限制因子在 [0.1, 5.0] 之間，防止步長變化過劇烈
            dt_new = dt * factor

        # 檢查誤差是否在容忍範圍內
        if error_magnitude <= scale:
            # 誤差在容忍範圍內，接受當前步的結果
            current_t += dt
            
            # **重要：重新正規化磁化向量**
            # LLG 方程式的固有性質是磁化向量模長不變。
            # 數值誤差會導致模長漂移，因此需要強制正規化。
            m_accepted = m_next_5 # 接受更高階的估計
            m_accepted_normalized = m_accepted / np.linalg.norm(m_accepted)
            
            current_m = m_accepted_normalized
            
            t_values.append(current_t)
            m_values.append(current_m)
            
            # 如果可以，增大步長以提高效率
            dt = dt_new
        else:
            # 誤差太大，拒絕當前步的結果，減小步長並重試
            dt = dt_new
            # 不要更新 current_t 和 current_M，因為這一步被拒絕了

        # 確保 dt 不會小到無法進步
        if dt < 1e-19: # 設置一個最小步長閾值
            print(f"Warning: Step size became too small at t={current_t}. Exiting.")
            break
        
    return np.array(t_values), np.array(m_values) * Ms

if __name__ == "__main__":
    pass
