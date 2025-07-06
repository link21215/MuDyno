from engine import field, solver, llg
import numpy as np
from matplotlib import pyplot as plt

def main():
    # --- 模擬參數設定 ---
    Ms = 1.0  # 飽和磁化強度 (單位化，例如 1.0 T)
    gamma_0 = 2.2128e5  # 旋磁比 (rad/(A*m*s) 或 (rad/(s*Oe) if using CGS))
    gamma = gamma_0  # 通常會直接使用 gamma，因為 LLG 方程式中的 gamma 已經包含了 mu_0
    alpha = 0.01  # 阻尼係數
    my_field = field.H_field_helper(EXT_H = [0, 0, 1e5])
    t_results, M_results = solver.runge_kutta_fehlberg_45(
        llg_equation = llg.llg_equation,
        m0 = np.array([0, 1, 0]),
        H_eff_func = my_field.external_field,
        gamma = gamma,
        alpha = alpha,
        Ms = Ms,
        t_span = (0, 10e-9),
        dt_initial = 1e-12,
        atol = 1e-3,
        rtol = 1e-3
    )

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

    # 將三維軌跡圖存檔
    plt.savefig("magnetization_trajectory.png")
    plt.close(fig)

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
    # 儲存磁化分量隨時間變化的圖
    plt.savefig("magnetization_components.png")
    plt.close()


if __name__ == "__main__":
    
    main()
