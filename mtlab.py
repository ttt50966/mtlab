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
    import matplotlib.pyplot as plt
    plt.figure(dpi=dpiValue)
    ax = plt.axes()
    ax.set_facecolor("white")
    plt.tight_layout(pad=3, w_pad=4.8, h_pad=3.6)
    plt.tick_params(direction = "in")
    plt.xticks(fontsize = 12,fontweight='bold')
    plt.yticks(fontsize = 12,fontweight='bold')
    plt.setp(ax.spines.values(), linewidth =2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    for data in line:
        if data["label"]!="":
            plt.plot(data["x"],data["y"], label=data["label"])
        else:
            plt.plot(data["x"],data["y"])
    plt.xlabel(xlabel,fontweight="bold",fontsize="12")
    plt.ylabel(ylabel,fontweight="bold",fontsize="12")
    plt.title(title,fontweight="bold",fontsize="15")
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

def fitting_init():
    import os
    os.system("pip install lmfit")
    import numpy as np

def diff_residual(params, x, y):
    import numpy as np
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
  import lmfit
  #Utilize lmfit.minimize to find the local minmum "residual"
  out = lmfit.minimize(diff_residual, params, args=(x,y)) 
  return out