import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

# 系統參數
NT = 32   # 發射端天線數
NR = 8    # 接收端天線數

Ncl = 2  # 信道聚類數量
Nray = 10  # 每個聚類的信道徑數量
max_iter = 5  # 限制最大迭代次數
SNR = np.array([-10, 0, 10])  # 信噪比 (dB)
# 功耗參數
Pmax = 16     # 最大發射功率 (W)
beta = 1 / 0.4  # 放大器效率 (1/0.4)
PCP = 10       # 電路功耗 (W)
PRF = 0.1      # 每個 RF 鏈功耗 (100 mW)
PPS = 0.01     # 每個相移器功耗 (10 mW)
PT = 0.1       # 每個天線發射功耗 (100 mW)
PR = 0.1       # 每個天線接收功耗 (100 mW)

# 隨機生成毫米波信道
def generate_mmwave_channel(NT, NR, Ncl=2, Nray=10, azimuth_mean=[60, -120], elevation_mean=[80, -100], spread=7.5):
    H = np.zeros((NR, NT), dtype=complex)
    for cluster_idx in range(Ncl):
        azimuth = np.random.uniform(azimuth_mean[cluster_idx] - spread, azimuth_mean[cluster_idx] + spread, Nray)
        elevation = np.random.uniform(elevation_mean[cluster_idx] - spread, elevation_mean[cluster_idx] + spread, Nray)
        for ray_idx in range(Nray):
            alpha = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)  # 每條徑的複增益
            a_t = np.exp(1j * np.pi * np.arange(NT) * np.sin(np.deg2rad(azimuth[ray_idx])))
            a_r = np.exp(1j * np.pi * np.arange(NR) * np.sin(np.deg2rad(elevation[ray_idx])))
            H += alpha * np.outer(a_r, a_t.conj())
    return H / np.sqrt(Ncl * Nray)
# 計算頻譜效率
def calculate_se(H, P, sigma2, LT):
    U, S, Vh = svd(H)
    singular_values = S[:LT]  # 使用前 LT 個奇異值
    term = 1 + singular_values**2 * P / sigma2
    term = np.maximum(term, 1e-10)  # 防止 log2 無效值
    se = np.sum(np.log2(term))
    return se

# 計算總功耗
# 更新功耗計算函數
def calculate_power(P, LT, PRF=0.1, PPS=0.01, PCP=10, PT=0.1, PR=0.1):
    # 總功耗 = 放大器功耗 + RF 鏈功耗 + 相移器功耗 + 天線功耗 + 固定電路功耗
    power = beta * P + LT * (PRF + PPS) + NT * PT + NR * PR + PCP
    return power

# Dinkelbach 方法
# 修正 Dinkelbach 方法，限制迭代記錄
def dinkelbach_method(H, Pmax, max_L, sigma2, epsilon=1e-4, max_iter=5, R_min=1):
    best_ee = 0
    best_L = 1
    ee_iterations = []  # 記錄每次迭代的 EE
    for LT in range(1, max_L + 1):
        P = Pmax / LT
        nu = 0
        for _ in range(max_iter):
            se = calculate_se(H, P, sigma2, LT)
            if se < R_min:  # 檢查是否滿足最低頻譜效率約束
                break
            # 使用更新的功耗計算
            power = calculate_power(P, LT, PRF=0.1, PPS=0.01, PCP=10, PT=0.1, PR=0.1)
            if power <= 0 or se <= 0:
                break
            ee = se / power
            ee_iterations.append(ee)
            if abs(ee - nu) < epsilon:
                break
            nu = ee
            P = max((Pmax - nu * power) / se, 1e-6)
        if nu > best_ee:
            best_ee = nu
            best_L = LT
    return best_ee, best_L, ee_iterations[:max_iter]
# 初始化結果存儲
EE_vs_iterations = {snr: [] for snr in SNR}

# 模擬不同 SNR 下的 EE
for snr_db in SNR:
    sigma2 = max(10 ** (-snr_db / 10), 1e-6)
    H = generate_mmwave_channel(NT, NR)  # 固定參數生成信道
    _, _, ee_iterations = dinkelbach_method(H, Pmax, NT, sigma2, max_iter=max_iter)
    EE_vs_iterations[snr_db] = ee_iterations

# 繪製圖表：EE 隨迭代次數變化
plt.figure()
for snr_db in SNR:
    plt.plot(range(1, max_iter + 1), EE_vs_iterations[snr_db],
             label=f"SNR = {snr_db} dB", marker="o")
plt.xlabel("Number of Iterations")
plt.ylabel("Energy Efficiency (bits/Hz/J)")
plt.legend()
plt.title(f"EE versus Number of Iterations at NT = {NT}, NR = {NR}, Pmax = {Pmax} W")
plt.grid(True)
plt.show()

