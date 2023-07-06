import numpy as np
import matplotlib.pyplot as plt
import os
import lmfit
import pandas as pd
def plot(line,dpiValue,title,xlabel,ylabel,saveFig):
    """
    This is a template for plotting
    line would be [{
        "x" : [],
        "y": [],
        "label" : ""
    }]

    dpiValue is integer like 300
    title, xlabel and ylabel are string
    saveFig is a boolean
    """

    plt.figure(dpi=dpiValue)
    ax = plt.axes()
    plt.rcParams['figure.facecolor'] = 'white'
    plt.tight_layout(pad=3, w_pad=4.8, h_pad=3.6)
    plt.tick_params(direction = "in")
    plt.xticks(fontsize = 18,fontweight='bold')
    plt.yticks(fontsize = 18,fontweight='bold')
    plt.setp(ax.spines.values(), linewidth =2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    for data in line:
        if data["label"]!="":
            plt.plot(data["x"],data["y"], label=data["label"])
        else:
            plt.plot(data["x"],data["y"])
    plt.xlabel(xlabel,fontweight="bold",fontsize="20")
    plt.ylabel(ylabel,fontweight="bold",fontsize="20")
    plt.title(title,fontweight="bold",fontsize="20")
    plt.legend(prop=dict(weight='bold'))
    if saveFig == True:
        plt.savefig(title+".png",dpi= dpiValue)

def moving_average(interval, windowsize):
    """
    This is a smooth function
    interval is the original data
    windowsize can set 3~10
    """
    import numpy as np
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re



def diff_residual(params, x, y):
    A = params['A'].value
    B = params['B'].value
    T = params['T'].value
    H = params['H'].value
    model = A*T**2*(2*H - 2*x)/(T**2 + (-H + x)**2)**2 - 2*B*T*(-H + x)*(2*H - 2*x)/(T**2 + (-H + x)**2)**2 - 2*B*T/(T**2 + (-H + x)**2)
    return np.sqrt((y-model)**2)

def diff_fit_data(params, x):
    A = params['A'].value
    B = params['B'].value
    T = params['T'].value
    H = params['H'].value
    model = A*T**2*(2*H - 2*x)/(T**2 + (-H + x)**2)**2 - 2*B*T*(-H + x)*(2*H - 2*x)/(T**2 + (-H + x)**2)**2 - 2*B*T/(T**2 + (-H + x)**2)
    return model

def diff_fitting(params,x,y):
  """
  import lmfit
  params = lmfit.Parameters() # Add new fitting objects
  params.add('A',value=-0.08)
  params.add('B',value=0.1)
  params.add('T', value=5)
  params.add('H', value=800)
  """
  #Utilize lmfit.minimize to find the local minmum "residual"
  out = lmfit.minimize(diff_residual, params, args=(x,y)) 
  return out
def residual(params, x, y):
  """
  x = data[0]["x"][0:]
  y = data[0]["y"][0:]
  title = data[0]["label"]
  params = lmfit.Parameters() # 新增Fitting參數物件
  params.add('S',value=0)
  params.add('A',value=0)
  params.add('T', value=50)
  params.add('c', value=-5)
  params.add('Hfmr', value=200)
  out = lmfit.minimize(mtlab.residual, params, args=(x,y)) #利用lmfit.minimize 找尋residual的局部極小值
  plt.plot(x,y,label="data")
  x_fit = np.arange(0,400,0.1)
  plt.plot(x_fit,mtlab.fit_data(out.params,x_fit),label="fit")
  plt.title(title)
  out.params
  """
  S = params['S'].value
  A = params['A'].value
  T = params['T'].value
  Hfmr = params['Hfmr'].value
  c = params['c'].value
  model = S* T**2/((x-Hfmr)**2+T**2) + A *T*(x-Hfmr)/((x-Hfmr)**2+T**2) + c
  return np.sqrt((y-model)**2)
def fit_data(params, x):
  """
  x = data[0]["x"][0:]
  y = data[0]["y"][0:]
  title = data[0]["label"]
  params = lmfit.Parameters() # 新增Fitting參數物件
  params.add('S',value=0)
  params.add('A',value=0)
  params.add('T', value=50)
  params.add('c', value=-5)
  params.add('Hfmr', value=200)
  out = lmfit.minimize(mtlab.residual, params, args=(x,y)) #利用lmfit.minimize 找尋residual的局部極小值
  plt.plot(x,y,label="data")
  x_fit = np.arange(0,400,0.1)
  plt.plot(x_fit,mtlab.fit_data(out.params,x_fit),label="fit")
  plt.title(title)
  out.params
  """
  S = params['S'].value
  A = params['A'].value
  T = params['T'].value
  Hfmr = params['Hfmr'].value
  c = params['c'].value
  model = S* T**2/((x-Hfmr)**2+T**2) + A *T*(x-Hfmr)/((x-Hfmr)**2+T**2) + c
  return model

def residual_kittel(params, x, y):
    u = 9.27401E-24
    h = 6.62607E-34
    g = params['g'].value
    M = params['M'].value
    model = g*u*np.sqrt(x*(x+M))*(1E-13)/h
    model2 = np.array([np.roots([1, M, -(i/g/u*h/(1E-13))**2])[1] for i in y])
    return np.sqrt((x-model2)**2+(y-model)**2)
def fit_data_kittel(params, x):
    u = 9.27401E-24
    h = 6.62607E-34
    g = params['g'].value
    M = params['M'].value
    model = g*u*np.sqrt(x*(x+M))*(1E-13)/h
    return model
def residual_damping(params, x, y):
    uB = 9.27401E-21
    h = 6.62607E-34
    pi = 3.14159
    g = params['g'].value
    H0 = params['H0'].value
    a = params['a'].value
    model = 2*pi*a*h*x*1E16/(g*uB)+H0
    return np.sqrt((y-model)**2)
def fit_data_damping(params, x):
    uB = 9.27401E-21
    h = 6.62607E-34
    pi = 3.14159
    g = params['g'].value
    H0 = params['H0'].value
    a = params['a'].value
    model = 2*pi*a*h*x*1E16/(g*uB)+H0
    return model
