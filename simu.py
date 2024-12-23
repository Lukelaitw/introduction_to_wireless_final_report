import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

# 系統參數
NT = 32   # 發射端天線數
NR = 8    # 接收端天線數
Ncl = 2  # 信道聚類數量
Nray = 10  # 每個聚類的信道徑數量
max_iter = 5  # 限制最大迭代次數
SNR = [0,5,10,15,20,25,30]  # 信噪比範圍 (dB)
NT_values = np.arange(8, 65, 8)
NT_values_1 = np.arange(20, 140, 20)  # 發射端天線數測試範圍

# 功耗參數
Pmax = 1     # 最大發射功率 (W)
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
            alpha = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)  # 平均功率設置為 1
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
def calculate_power(P, LT, PRF=0.1, PPS=0.01, PCP=10, PT=0.1, PR=0.1):
    power = beta * P + LT * (PRF + PPS) + NT * PT + NR * PR + PCP
    return power

# Dinkelbach 方法
def dinkelbach_method(H, Pmax, max_L, sigma2, epsilon=1e-4, max_iter=5, R_min=1):
    best_ee = 0
    best_L = 1
    ee_iterations = []  # 記錄每次迭代的 EE
    for LT in range(1, max_L + 1):
        P = Pmax / LT
        nu = 0
        for _ in range(max_iter):
            se = calculate_se(H, P, sigma2, LT)
            if se < R_min:  # 如果 SE 小於最小要求值則跳出
                break
            power = calculate_power(P, LT)
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
EE_vs_SNR_DM, SE_vs_SNR_DM = [], []
EE_vs_SNR_BF, SE_vs_SNR_BF = [], []
EE_vs_NT_DM, SE_vs_NT_DM = [], []
EE_vs_NT_BF, SE_vs_NT_BF = [], []

# 信噪比模擬
for snr_db in SNR:
    sigma2 = 10**(-snr_db / 10)
    H = generate_mmwave_channel(NT, NR, Ncl, Nray)
    
    # Dinkelbach 方法
    ee_dm, _, _ = dinkelbach_method(H, Pmax, NT, sigma2)
    se_dm = calculate_se(H, Pmax / NT, sigma2, NT)
    EE_vs_SNR_DM.append(ee_dm)
    SE_vs_SNR_DM.append(se_dm)
    
    # Brute Force 方法
    ee_bf, _ = max((calculate_se(H, Pmax / LT, sigma2, LT) / calculate_power(Pmax / LT, LT), LT)
                   for LT in range(1, NT + 1))
    se_bf = calculate_se(H, Pmax / NT, sigma2, NT)
    EE_vs_SNR_BF.append(ee_bf)
    SE_vs_SNR_BF.append(se_bf)

# 繪製 EE 和 SE 隨 SNR 的變化
plt.figure()
plt.plot(SNR, EE_vs_SNR_DM, label="EE (DM)", linestyle="-", marker="o")
plt.plot(SNR, EE_vs_SNR_BF, label="EE (BF)", linestyle="--", marker="x")
plt.xlabel("SNR (dB)")
plt.ylabel("Energy Efficiency")
plt.legend()
plt.title("EE versus SNR")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(SNR, SE_vs_SNR_DM, label="SE (DM)", linestyle="-", marker="o")
plt.plot(SNR, SE_vs_SNR_BF, label="SE (BF)", linestyle="--", marker="x")
plt.xlabel("SNR (dB)")
plt.ylabel("Efficiency")
plt.legend()
plt.title("SE versus SNR")
plt.grid(True)
plt.show()

"""
# 更新天線數範圍
NT_values = np.arange(40, 121, 20)  # 天線數從 40 到 120

# 模擬 EE 和 SE 隨 NT 的變化
EE_vs_NT_DM, SE_vs_NT_DM = [], []
EE_vs_NT_BF, SE_vs_NT_BF = [], []

for NT2 in NT_values:
    sigma2 = 10**(-10 / 10)  # 固定 SNR = 10 dB
    H = generate_mmwave_channel(NT2, NR, Ncl, Nray)
    
    # Dinkelbach 方法
    ee_dm,_, _ = dinkelbach_method(H, Pmax, NT2, sigma2)
    se_dm = calculate_se(H, Pmax / NT2, sigma2, NT)
    EE_vs_NT_DM.append(ee_dm)
    SE_vs_NT_DM.append(se_dm)
    
    # Brute Force 方法
    ee_bf, _ = max((calculate_se(H, Pmax / LT, sigma2, LT) / calculate_power(Pmax / LT, LT), LT)
                   for LT in range(1, NT2 + 1))
    se_bf = calculate_se(H, Pmax / NT2, sigma2, NT2)
    EE_vs_NT_BF.append(ee_bf)
    SE_vs_NT_BF.append(se_bf)

# 繪製 EE 隨 NT 的變化
plt.figure()
plt.plot(NT_values, EE_vs_NT_DM, label="Dinkelbach Method (DM)", linestyle="-", marker="o")
plt.plot(NT_values, EE_vs_NT_BF, label="Brute Force (BF)", linestyle="--", marker="x")
plt.xlabel("Number of TX antennas NT")
plt.ylabel("Energy Efficiency (bits/Hz/J)")
plt.legend()
plt.title("EE versus NT")
plt.grid(True)
plt.show()

# 繪製 SE 隨 NT 的變化
plt.figure()
plt.plot(NT_values, SE_vs_NT_DM, label="Dinkelbach Method (DM)", linestyle="-", marker="o")
plt.plot(NT_values, SE_vs_NT_BF, label="Brute Force (BF)", linestyle="--", marker="x")
plt.xlabel("Number of TX antennas NT")
plt.ylabel("Spectral Efficiency (bits/s/Hz)")
plt.legend()
plt.title("SE versus NT")
plt.grid(True)
plt.show()
"""