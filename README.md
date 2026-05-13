# mtlab

A Python utility for publication-quality plotting and FMR/ST-FMR data fitting.

## Installation

```bash
git clone https://github.com/ttt50966/mtlab.git
pip install -r requirements.txt
```

Copy `mtlab.py` to your working directory, then `import mtlab`.

## Functions

| Function | Description |
|---|---|
| `plot(line, dpiValue, title, xlabel, ylabel, saveFig)` | Publication-style line plot |
| `moving_average(interval, windowsize)` | Moving average smoothing |
| `diff_fitting(params, x, y)` | Fit derivative Lorentzian (ST-FMR) |
| `diff_fit_data(params, x)` | Evaluate derivative Lorentzian model |
| `residual(params, x, y)` | Fit symmetric + antisymmetric Lorentzian (FMR) |
| `fit_data(params, x)` | Evaluate Lorentzian model |
| `residual_kittel(params, x, y)` | Fit Kittel dispersion |
| `fit_data_kittel(params, x)` | Evaluate Kittel model |
| `residual_damping(params, x, y)` | Fit Gilbert damping (linewidth vs frequency) |
| `fit_data_damping(params, x)` | Evaluate damping model |

## Quick Start

```python
import mtlab
import pandas as pd
import lmfit

# Plot
df = pd.read_csv('data.csv')
line = [{'x': df['x'].values, 'y': df['y'].values, 'label': 'data'}]
mtlab.plot(line, dpiValue=300, title='FMR', xlabel='H (Oe)', ylabel='V (V)', saveFig=True)

# ST-FMR derivative fit
params = lmfit.Parameters()
params.add('A', value=-0.08)
params.add('B', value=0.1)
params.add('T', value=5)
params.add('H', value=800)
out = mtlab.diff_fitting(params, x, y)
print(out.params)
```

See [`example_usage.ipynb`](example_usage.ipynb) for a full walkthrough, or open it directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ttt50966/mtlab/blob/main/example_usage.ipynb)
