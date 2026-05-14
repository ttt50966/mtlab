import numpy as np
import matplotlib.pyplot as plt
import lmfit
import pandas as pd
from pathlib import Path

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
    # data["type"] 可設為 "scatter"（預設 "line"）
    for data in line:
        label = data.get("label", "")
        plot_type = data.get("type", "line")
        kwargs = {"label": label} if label else {}
        if plot_type == "scatter":
            plt.scatter(data["x"], data["y"], **kwargs)
        else:
            plt.plot(data["x"], data["y"], **kwargs)

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
    return y - model

def diff_fitting(params, x, y):
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
    u = 9.27401E-24
    h = 6.62607E-34

    model = g * u * np.sqrt(x * (x + M)) * (1E-13) / h
    return model

def residual_kittel(params, x, y):
    model = fit_data_kittel(params, x)

    g = params['g'].value
    M = params['M'].value
    u = 9.27401E-24
    h = 6.62607E-34

    model2_list = []
    factor = h / (g * u * 1E-13)
    for i in y:
        C = (i * factor) ** 2
        poly = [1, M, -C]
        roots = np.roots(poly)
        model2_list.append(roots[1])

    model2 = np.array(model2_list)
    return np.sqrt((x - model2)**2 + (y - model)**2)

# --- Damping Fitting ---

def fit_data_damping(params, x):
    g = params['g'].value
    H0 = params['H0'].value
    a = params['a'].value

    uB = 9.27401E-21  # 注意這裡原本 code 是 E-21，Kittel 是 E-24，可能有單位轉換差異
    h = 6.62607E-34

    model = 2 * np.pi * a * h * x * 1E16 / (g * uB) + H0
    return model

def residual_damping(params, x, y):
    model = fit_data_damping(params, x)
    return y - model

# --- LabVIEW I/O ---

def read_lvm(path):
    """
    Parse a LabVIEW .lvm file and return column names + data array.

    Returns dict with keys:
        "columns" : list of str  (column headers, including Comment)
        "data"    : np.ndarray   (shape: n_rows x n_data_cols, NaN for missing/NaN cells)

    Usage:
        lvm = read_lvm("measurement.lvm")
        cols = lvm["columns"]
        data = lvm["data"]
        col_idx = next(i for i, c in enumerate(cols) if "Voltage" in c) - 1
        V = data[:, col_idx]
    """
    path = Path(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # find second ***End_of_Header*** -> column name row follows
    eoh_count = 0
    col_line = data_start = None
    for i, line in enumerate(lines):
        if "***End_of_Header***" in line:
            eoh_count += 1
        if eoh_count == 2 and col_line is None:
            col_line = i + 1
            data_start = i + 2
            break

    columns = lines[col_line].strip().split("\t")
    n_data_cols = len(columns) - 1  # exclude trailing Comment column

    rows = []
    for line in lines[data_start:]:
        parts = line.strip().split("\t")
        if not parts:
            continue
        if parts[0] == "":
            parts = parts[1:]  # strip leading X_Value tab
        vals = []
        for x in parts[:n_data_cols]:
            try:
                vals.append(float(x))
            except (ValueError, TypeError):
                vals.append(np.nan)
        if len(vals) < n_data_cols:
            vals += [np.nan] * (n_data_cols - len(vals))
        if vals:
            rows.append(vals)

    return {"columns": columns, "data": np.array(rows, dtype=float)}


# --- Axes Styling ---

def apply_style(ax, n_xticks=5):
    """
    Apply publication-ready style to an Axes object.

    Sets spine/tick width to 2, ticks inward, bold tick labels,
    and places n_xticks evenly-spaced integer ticks on the x-axis.

    Call this after all ax.plot() calls and before plt.tight_layout().
    """
    xmin, xmax = ax.get_xlim()
    n_intervals = n_xticks - 1
    raw_step = (xmax - xmin) / n_intervals
    magnitude = 10 ** np.floor(np.log10(raw_step))
    nice_step = np.floor(raw_step / magnitude) * magnitude
    center = np.round((xmin + xmax) / 2 / nice_step) * nice_step
    half = n_intervals // 2
    ticks = center + np.arange(-half, half + 1) * nice_step
    ax.set_xticks(ticks)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.tick_params(direction="in", labelsize=18)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
