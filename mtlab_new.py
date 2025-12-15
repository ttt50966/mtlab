import numpy as np
import matplotlib.pyplot as plt
import lmfit
import pandas as pd

# 定義物理常數 (可全域使用，避免重複定義)
# U_B: Bohr magneton, H_CONST: Planck constant
U_B = 9.27401E-24  # J/T
H_CONST = 6.62607E-34  # J*s
# 若需要配合原本 code 中的單位 (例如 1E-21 或 1E-13 的修正)，維持原本數值，但建議統一管理

def plot(line, dpiValue, title, xlabel, ylabel, saveFig):
    """
    This is a template for plotting.
    Input arguments remained unchanged.
    """
    plt.figure(dpi=dpiValue)
    ax = plt.axes()
    
    # 風格設定
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 使用 get 避免 Key Error，並設定預設值
    for data in line:
        label = data.get("label", "")
        if label:
            plt.plot(data["x"], data["y"], label=label)
        else:
            plt.plot(data["x"], data["y"])

    # 圖表裝飾
    plt.xlabel(xlabel, fontweight="bold", fontsize="20")
    plt.ylabel(ylabel, fontweight="bold", fontsize="20")
    plt.title(title, fontweight="bold", fontsize="20")
    
    # 只有在有 label 的時候才顯示圖例，避免警告
    if any(data.get("label") for data in line):
        plt.legend(prop=dict(weight='bold'))

    # 美化刻度
    plt.tick_params(direction="in")
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    
    # 設定邊框粗細
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    plt.tight_layout(pad=3, w_pad=4.8, h_pad=3.6)

    if saveFig:
        plt.savefig(title + ".png", dpi=dpiValue)

def moving_average(interval, windowsize):
    """
    Smooth function
    """
    # 移除重複的 import numpy as np
    window = np.ones(int(windowsize)) / float(windowsize)
    return np.convolve(interval, window, 'same')

# --- Diff Fitting ---

def diff_fit_data(params, x):
    A = params['A'].value
    B = params['B'].value
    T = params['T'].value
    H = params['H'].value
    
    # 提取公共項以提升可讀性與效能
    denom = T**2 + (-H + x)**2
    term1 = (2*H - 2*x) / denom**2
    term2 = (-H + x) * term1
    term3 = 1 / denom
    
    model = A*T**2 * term1 - 2*B*T * term2 - 2*B*T * term3
    return model

def diff_residual(params, x, y):
    model = diff_fit_data(params, x)
    # 重要修改: 回傳原始殘差 (y - model)，讓演算法判斷方向
    return y - model

def diff_fitting(params, x, y):
    # Utilize lmfit.minimize
    return lmfit.minimize(diff_residual, params, args=(x, y))

# --- Lorentzen / ST-FMR Fitting ---

def fit_data(params, x):
    S = params['S'].value
    A = params['A'].value
    T = params['T'].value
    Hfmr = params['Hfmr'].value
    c = params['c'].value
    
    denom = (x - Hfmr)**2 + T**2
    model = S * T**2 / denom + A * T * (x - Hfmr) / denom + c
    return model

def residual(params, x, y):
    model = fit_data(params, x)
    return y - model

# --- Kittel Fitting ---

def fit_data_kittel(params, x):
    g = params['g'].value
    M = params['M'].value
    # 統一使用全域常數，或維持原本數值
    u = 9.27401E-24
    h = 6.62607E-34
    
    model = g * u * np.sqrt(x * (x + M)) * (1E-13) / h
    return model

def residual_kittel(params, x, y):
    model = fit_data_kittel(params, x)
    
    # 保留原本特殊的 residual 計算邏輯 (看起來是考慮了 X 軸誤差或是某種反向求解)
    # 但建議檢查這裡的物理意義是否正確
    g = params['g'].value
    M = params['M'].value
    u = 9.27401E-24
    h = 6.62607E-34
    
    # 這段邏輯較複雜，維持原樣，但優化語法
    # 計算 model2 (從 y 反推 x?)
    model2_list = []
    factor = h / (g * u * 1E-13)
    for i in y:
        # roots([1, M, -C]) => x^2 + Mx - C = 0
        C = (i * factor) ** 2
        # Use np.array for roots calculation as it's typically faster and handles complex roots
        # For real roots, np.roots will return real numbers or complex numbers with zero imaginary parts
        poly = [1, M, -C]
        roots = np.roots(poly)
        # Assuming the second root (index 1) is the desired one based on original code
        model2_list.append(roots[1]) 
    
    model2 = np.array(model2_list)
    
    # 這裡原本是回傳 sqrt(dx^2 + dy^2)，這是一種 Total Least Squares 的概念
    # 因為這不是單純的 y-model，所以這裡保留 sqrt 或是回傳該距離向量
    return np.sqrt((x - model2)**2 + (y - model)**2)

# --- Damping Fitting ---

def fit_data_damping(params, x):
    g = params['g'].value
    H0 = params['H0'].value
    a = params['a'].value
    
    uB = 9.27401E-21 # 注意這裡原本 code 是 E-21，Kittel 是 E-24，可能有單位轉換差異
    h = 6.62607E-34
    
    model = 2 * np.pi * a * h * x * 1E16 / (g * uB) + H0
    return model

def residual_damping(params, x, y):
    model = fit_data_damping(params, x)
    return y - model
