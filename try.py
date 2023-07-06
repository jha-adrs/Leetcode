from scipy import stats as stats
import numpy as np
from numpy import cos, sin, sqrt as isqrt, pi
import datetime as dt


def gen_sigma(count, start=200, stop=2e5):
    return np.linspace(start, stop, count)


def gen_freq(count, start=0.1, stop=10.001, step=0.001, multiplicant=1e12):
    return np.linspace(start, stop, count) * 1e12


def calc_MA(da, kzA, pA, array_length=10):
    MA = list()
    # print("Starting calc_MA")
    for i in range(array_length):
        MA.append(np.array(
            [[cos(kzA * da[i]), (1j / pA) * sin(kzA * da[i])], [1j * pA * sin(kzA * da[i]), cos(kzA * da[i])]]))
    return MA


def calc_MB(db, kzB, pB, array_length=10):
    # VO2
    # print("Starting calc_MB")
    return np.array([[cos(kzB * db), (1j / pB) * sin(kzB * db)], [1j * pB * sin(kzB * db), cos(kzB * db)]])


def calc_MC(dc, kzC, pC, array_length=10):
    MC = list()
    # print("Starting calc_MC")
    for i in range(array_length):
        MC.append(np.array(
            [[cos(kzC * dc[i]), (1j / pC) * sin(kzC * dc[i])], [1j * pC * sin(kzC * dc[i]), cos(kzC * dc[i])]]))
    return MC


def genArray(MA, MB, MC, array_length=10):
    # print("Starting genArray")
    a1 = [x for x in MA]
    a2 = [x for x in MB]
    a3 = [x for x in MC]
    F = []
    F.extend(a1)
    F.extend(a2)
    F.extend(a3)
    # print(F[0])
    return (F)


def calc_multidot(MA, MB, MC, array_length=10):
    # print("Starting calc_multidot")
    X = list()
    X = np.dot(MA[0], MC[0])
    for i in range(1, int(array_length / 2)):
        X = np.dot(X, MA[i])
        X = np.dot(X, MC[i])
    X = np.dot(X, MB)
    for i in range(int(array_length / 2), array_length):
        X = np.dot(X, MC[i])
        X = np.dot(X, MA[i])
    # print(X)
    return X


# Generates random values for da and dc
def genWidth(array_length=10,sd=0.0, lambda0=70e-6, mean=6.25e-6):
    if sd == 0:
        da = list(np.random.normal(mean, sd, array_length))
        dc = list(np.random.normal(mean, sd, array_length))
    else:
        da = stats.truncnorm.rvs(((1e-6 - mean) / sd), (11.25e-6 - mean) / sd, loc=mean, scale=sd,
                                 size=array_length)
        dc = stats.truncnorm.rvs(((1e-6 - mean) / sd), (11.25e-6 - mean) / sd, loc=mean, scale=sd,
                                 size=array_length)
    return [da, dc]



def get_AsContour(fre, sigma, thicknesses, array_length =10):
    # Constants
    da = thicknesses[0]
    dc =  thicknesses[1]
    c = 3e8
    e0 = 8.85e-12
    u0 = (4 * pi) * (10 ** -7)
    e = 1.6e-19
    h = 6.63e-34
    hbar = h / (2 * pi)

    # Constants 2
    n0 = 1  # air
    nt = 1
    ut = 1
    et = nt ** 2  # t:substrate
    na = 2.25
    ua = 1
    ea = na ** 2  # SiO2
    nc = 3.3
    uc = 1
    ec = nc ** 2  # Si
    ub = 1
    lambda0 = 70e-6
    sigma_VO2 = sigma  # S/m Variable 200 or 200000
    db = lambda0 / (4 * isqrt(12))  # VO2
    # *****************************VO2
    eps_inf = 12
    gamma = 5.75e13  # rad/s
    wp = 1.4e15 * isqrt(sigma_VO2 / 3e5)
    theta = 0 * pi / 180  # Variable
    TETM = 1
    jj = 0  # set to 0 as python indices start from 0
    n = 0
    mean = 6.25e-6
    data_points = 10
    w = 2 * pi * fre
    epsVO2 = eps_inf - wp ** 2 / \
             (w ** 2 + 1j * gamma * w)  # used j instead of i
    ###
    kz0 = (w / c) * n0 * cos(theta)
    kzA = (w / c) * na * isqrt(1 - (sin(theta) ** 2 / (na ** 2)))
    kzB = (w / c) * (isqrt(epsVO2)) * (
        isqrt(1 - (sin(theta) ** 2 / epsVO2)))  # isqrt for sqrt of complex number
    kzC = (w / c) * nc * isqrt(1 - (sin(theta) ** 2 / (nc ** 2)))
    kzt = (w / c) * nt * isqrt(1 - (sin(theta) ** 2 / (et * ut)))

    if TETM == 1:
        y0 = u0
        ya = ua * u0
        yb = ub * u0
        yc = uc * u0
        yt = ut * u0  # TE
    else:
        y0 = -e0
        ya = -ea * e0
        yb = -epsVO2 * e0
        yc = -ec * e0
        yt = -et * e0  # TM

    p0 = kz0 / (w * y0)
    pA = kzA / (w * ya)
    pB = kzB / (w * yb)
    pC = kzC / (w * yc)
    pt = kzt / (w * yt)

    # 2x2 Matrix multiplied with scalar
    m1 = 0.5 * np.array([[1, -1 / p0], [1, 1 / p0]])
    mt = np.array([[1], [-pt]])
    ###
    MA = calc_MA(da, kzA, pA,array_length=array_length)
    MB = calc_MB(db, kzB, pB,array_length=array_length)
    MC = calc_MC(dc, kzC, pC,array_length=array_length)
    ###
    X = calc_multidot(MA, MB, MC, array_length=array_length)
    MM = np.linalg.multi_dot([m1, X, mt])
    r = MM[1][0] / MM[0][0]
    t = 1 / (MM[0][0])
    ar = float(abs(r) ** 2)

    ###
    Ts = (kzt / kz0) * abs(t) ** 2
    As = 1 - ar - Ts
    ##
    Tp = (kzt / (et * kz0)) * abs(t) ** 2
    Ap = 1 - r - Tp
    ##

    return As

#from other_functions import get_AsContour, genWidth, genArray
import matplotlib.pyplot as plt
import concurrent.futures
import datetime as dt
import time
import pandas as pd
import numpy as np
import pickle
# Returns a single absorption value for a given frequency and structure number
# As is the mean of 100 structures at that frequency and structure number
def save_pickle(Z,X,Y,xx):
    print("Pickling...\n")
    pickled = pickle.dump([Z,X,Y], open(f"Results/Pickles/pickle_{xx}contour_data.p", "wb"))
    print("Pickled! \n")

def save_excel(Z,X,Y,xx):
    print("Saving to excel...\n")
    df = pd.DataFrame(Z, index=Y, columns=X)
    df.to_excel(f"Results/Excel/Df_{xx}contour_data.xlsx")
    print("Saved! \n")

def get_As_using_freq_strno(freq, strno, sigma_SD, count):
    As=[]
    
    for i in range(count):
        da, dc = genWidth(array_length=strno, sd=sigma_SD[1])
        As.append(get_AsContour(freq, sigma_SD[0], [da, dc], array_length=strno))
    return np.array(As).mean()
    

def main(xx, x_length=10, noOfStructures=10, threads=8, font_size = 25):
    start_time = dt.datetime.now()
    print("Execution started at:", start_time)
    reference_dict = {0:'a', 1:'c', 2:'d', 3:'b', 4:'d', 5:'f'}

    sigma_SD = [[200, 0.5e-6], [200, 1e-6], [200, 1.5e-6], [200000, 0.5e-6], [200000, 1e-6], [200000, 1.5e-6]]
    
    X_freq = np.linspace(0.1, 10.001,x_length) * 1e12
    Y_freq = np.arange(1, 51)
    
    color_levels = np.arange(0, 0.71, 0.01)

    total_iterations = len(X_freq) 
    current_iteration = 0
    Z=[]
    for i in X_freq:
        
        with concurrent.futures.ProcessPoolExecutor(threads) as executor:
            z = executor.map(get_As_using_freq_strno, [i]*len(Y_freq), Y_freq, [sigma_SD[xx]]*len(Y_freq), [noOfStructures]*len(Y_freq))
            Z.append([x for x in z])
        current_iteration += 1
        print("Progress: {}/{}".format(current_iteration, total_iterations))
    
    Z = np.array(Z).T  #transpose Z

    save_excel(Z,X_freq,Y_freq,xx) # Saving the data

    fig = plt.figure(figsize=(12, 10))
    plt.contourf(X_freq, Y_freq, Z, cmap='afmhot', levels=color_levels)
    #plt.colorbar()

    # Customize the plot
    plt.xlabel('('+reference_dict[xx]+')', fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
   
    plt.savefig(f'Results/Plots/contour_{xx}.png',dpi=2400)
    

    end_time = dt.datetime.now()
    save_pickle(Z,X_freq,Y_freq,xx) # Saving the data
    print("Execution completed at:", end_time)
    print("Total execution time:", end_time - start_time)
    return fig

import matplotlib as mpl
def gen_colorbar(vmin =0, vmax = 0.7, font_size = 25):
    fig = plt.figure(figsize=(2,20))
    ax = fig.add_axes([0,0.2,0.4,0.6])
    cmap = mpl.cm.afmhot
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.ax.tick_params(labelsize = font_size)
    #cb1.set_label('Some Units')
    plt.savefig("Results/Plots/colorbar.png", dpi = 2400)
    plt.show()
    return fig

from matplotlib.gridspec import GridSpec

if __name__ == "__main__":
    x = 100
    structs = 100
    threads = 8

    fig0 = main(0, x_length=x, noOfStructures=structs, threads=threads)
    print("\n\n\033[91m {}\033[00m" .format("S200 & 0.5 Done"), end="\n\n")

    fig1 =  main(1, x_length=x, noOfStructures=structs, threads=threads)
    print("\n\n\033[91m {}\033[00m" .format("S200 & 1.0 Done"), end="\n\n")

    fig2 = main(2, x_length=x, noOfStructures=structs, threads=threads)
    print("\n\n\033[91m {}\033[00m" .format("S200 & 1.5 Done"), end="\n\n")

    fig3 = main(3, x_length=x, noOfStructures=structs, threads=threads)
    print("\n\n\033[91m {}\033[00m" .format("S200000 & 0.5 Done"), end="\n\n")

    fig4 = main(4, x_length=x, noOfStructures=structs, threads=threads)
    print("\n\n\033[91m {}\033[00m" .format("S200000 & 1.0 Done"), end="\n\n")

    fig5= main(5, x_length=x, noOfStructures=structs, threads=threads)
    print("\n\n\033[91m {}\033[00m" .format("S200000 & 1.5 Done"), end="\n\n")

    #colorplot = gen_colorbar()
    plt.show()

"""
    num_rows =2
    num_cols =3
    figures = [fig0, fig1, fig2, fig3, fig4, fig5]
    fig = plt.figure(figsize=(30, 20))
    grid = GridSpec(num_rows, num_cols,figure = fig)

    for i, figure in enumerate(figures):
        row = i // num_cols
        col = i% num_cols
        ax = fig.add_subplot(grid[row,col])
        ax.plot(figure)

    cax = fig.add_subplot(grid[:,-1])
    fig.colorbar(figure, cax=cax)

    plt.tight_layout()
    plt.show()

    #fig.colorbar(colorplot, ax=axs, orientation='vertical', fraction=.1)
    plt.savefig("all_plots.png", dpi = 2400)"""

    

            