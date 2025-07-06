import numpy as np

def llg_equation(m, H_eff, gamma, alpha):
    """
    計算 Landau-Lifshitz-Gilbert 方程式的右側 dM/dt。
    M: 當前磁化向量 (numpy array [Mx, My, Mz])
    H_eff: 有效磁場向量 (numpy array [Hx, Hy, Hz])
    gamma: 旋磁比
    alpha: 阻尼係數
    Ms: 飽和磁化強度
    """
    
    # 將 M 正規化為單位向量 m
    # m = M / np.linalg.norm(M)
    
    # LLG 方程式的顯式形式
    # 這裡假設 H_eff 已經是處理好的有效磁場
    
    # 第一項：進動項 -gamma * M x H_eff
    # precession_term = -gamma * np.cross(M, H_eff)
    
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
    mXH = np.cross(m, H_eff)
    mXmXH = np.cross(m, mXH)
    
    dm_dt = (-gamma / (1 + alpha**2)) * (mXH + alpha * mXmXH)
    
    return dm_dt