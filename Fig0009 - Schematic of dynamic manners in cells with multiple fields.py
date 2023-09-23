from mylib.statistic_test import *

code_id = '0009 - Schematic of dynamic manners in cells with multiple fields'
loc = join(figpath, code_id)
mkdir(loc)

'''
def Gaussian(x=0, sigma=2, pi=3.1416, nx = 48):
    x = x * (48 / nx)
    return 1 / (sigma * np.sqrt(pi * 2)) * np.exp(- x * x / (sigma * sigma * 2))
'''

def plot_figure(x, y1: np.ndarray, y2:np.ndarray, file_name: str = ''):
    fig, axes = plt.subplots(ncols=1, nrows=4, figsize = (3,2.5), gridspec_kw={'height_ratios':[8,1,8,1]})
    ax1 = Clear_Axes(axes[0], close_spines=['top', 'right', 'bottom'], ifyticks=True)
    ax2 = Clear_Axes(axes[1])
    ax3 = Clear_Axes(axes[2], close_spines=['top', 'right', 'bottom'], ifyticks=True) 
    ax4 = Clear_Axes(axes[3])
    
    y1_max = np.max(y1)
    y2_max = np.max(y2)

    ax1.plot(x-0.5, y1, color = 'black', linewidth=0.8)
    ax1.axis([-30, np.max(x)-0.5, -0.05, 3+0.05])
    ax1.set_yticks([0, 3])

    ax3.plot(x-0.5, y1, ls=':', color = 'gray', linewidth=0.8, label = "Before drifting")
    ax3.plot(x-0.5, y2, color = 'black', linewidth=0.8, label="After drifting")
    ax3.axis([-30, np.max(x)-0.5, -0.05, 3+0.05])
    ax3.legend()
    ax3.set_yticks([0, 3])

    ax2.imshow(np.reshape(y1, [1, y1.shape[0]]), vmin=0, vmax=3)
    ax4.imshow(np.reshape(y2, [1, y2.shape[0]]), vmin=0, vmax=3)
    ax2.set_aspect("auto")
    ax4.set_aspect("auto")
    ax2.axis([-30, np.max(x)-0.5, -0.5, 0.5])
    ax4.axis([-30, np.max(x)-0.5, -0.5, 0.5])

    plt.tight_layout()
    plt.savefig(join(loc, file_name+'.png'), dpi=600)
    plt.savefig(join(loc, file_name+'.svg'), dpi=600)
    plt.close()
    
def get_y(y, ampl):
    return y / np.nanmax(y) * ampl

x = np.linspace(0,11,1101)
sigma = 3

cent1, ampl1 = 2, 3
cent2, ampl2 = 7, 3

cent3, ampl3 = 2, 0
cent4, ampl4 = 2, 1



y1 = get_y(Gaussian(x-cent1, sigma=sigma, nx=12), ampl1)
y2 = get_y(Gaussian(x-cent2, sigma=sigma, nx=12), ampl2)
y3 = get_y(Gaussian(x-cent3, sigma=sigma, nx=12), ampl3)
y4 = get_y(Gaussian(x-cent4, sigma=sigma, nx=12), ampl4)

plot_figure(x*100, y1, y2, 'PCsf - Shift')
plot_figure(x*100, y3, y2, 'PCsf - Emerge')
plot_figure(x*100, y1, y3, 'PCsf - Disappear')
plot_figure(x*100, y1, y4, 'PCsf - Weaken')



# multiple fields
cent5_1, cent5_2, cent5_3 = 2, 4, 9
ampl5_1, ampl5_2, ampl5_3 = 3, 1.3, 2.1
y5 = get_y(Gaussian(x-cent5_1, sigma=sigma, nx=1), ampl5_1) + get_y(Gaussian(x-cent5_2, sigma=sigma, nx=1), ampl5_2) + get_y(Gaussian(x-cent5_3, sigma=sigma, nx=1), ampl5_3)

cent6_1, cent6_2 = 2, 4
ampl6_1, ampl6_2 = 3, 1.3
y6 = get_y(Gaussian(x-cent6_1, sigma=sigma, nx=1), ampl6_1) + get_y(Gaussian(x-cent6_2, sigma=sigma, nx=1), ampl6_2)

# multiple fields
cent7_1, cent7_2, cent7_3, cent7_4 = 2, 4, 9, 6
ampl7_1, ampl7_2, ampl7_3, ampl7_4 = 3, 1.3, 2.1, 1.6
y7 = get_y(Gaussian(x-cent7_1, sigma=sigma, nx=1), ampl7_1) + get_y(Gaussian(x-cent7_2, sigma=sigma, nx=1), ampl7_2) + get_y(Gaussian(x-cent7_3, sigma=sigma, nx=1), ampl7_3) + get_y(Gaussian(x-cent7_4, sigma=sigma, nx=1), ampl7_4)


cent8_1, cent8_2, cent8_3 = 2, 4, 7
ampl8_1, ampl8_2, ampl8_3 = 3, 1.3, 2.1
y8 = get_y(Gaussian(x-cent8_1, sigma=sigma, nx=1), ampl8_1) + get_y(Gaussian(x-cent8_2, sigma=sigma, nx=1), ampl8_2) + get_y(Gaussian(x-cent8_3, sigma=sigma, nx=1), ampl8_3)

cent9_1, cent9_2 = 3, 7
ampl9_1, ampl9_2 = 2.4, 3
y9 = get_y(Gaussian(x-cent9_1, sigma=sigma, nx=1), ampl9_1) + get_y(Gaussian(x-cent9_2, sigma=sigma, nx=1), ampl9_2)

cent10_1, cent10_2, cent10_3 = 2, 4, 9
ampl10_1, ampl10_2, ampl10_3 = 3, 1.3, 0.5
y10 = get_y(Gaussian(x-cent10_1, sigma=sigma, nx=1), ampl10_1) + get_y(Gaussian(x-cent10_2, sigma=sigma, nx=1), ampl10_2) + get_y(Gaussian(x-cent10_3, sigma=sigma, nx=1), ampl10_3)

plot_figure(x*100, y5, y6, 'PCmf - Disappear')
plot_figure(x*100, y5, y7, 'PCmf - Emerge')
plot_figure(x*100, y5, y8, 'PCmf - Shift')
plot_figure(x*100, y5, y9, 'PCmf - Complex')
plot_figure(x*100, y5, y10, 'PCmf - Weaken')

from matplotlib.gridspec import GridSpec
import seaborn as sns
from mylib.decoder.PiecewiseConstantSigmoidRegression import TwoPiecesPiecewiseSigmoidRegression
from mylib.calcium.smooth.interpolation import interpolated_smooth

def plot_multi_line(x, x1, x2, ys, file_name: str):
    assert ys.shape[0] == 11
    
    x_mid = int((x1+x2)/2)

    fig = plt.figure(figsize = (6, 8))
    grid = GridSpec(nrows=11, ncols=2, width_ratios=[4,1], height_ratios=np.repeat(1,11))

    colors = sns.color_palette('rocket', 4)

    for i in range(11):
        ax = Clear_Axes(fig.add_subplot(grid[i, 0]))

        if i != 0:
            ax.plot(x, ys[0, :], color = 'gray', linewidth=0.8, ls=":")
        ax.set_ylim([-0.2, 3.2])
        ax.plot(x, ys[i, :], color='black', linewidth=0.8)  
        ax.fill_betweenx(y=[0,3], x1 = x[x1-15], x2 = x[x1+15], color=colors[1], alpha=0.5, edgecolor=None, label='Previous field')
        ax.fill_betweenx(y=[0,3], x1 = x[x_mid-15], x2 = x[x_mid+15], color=colors[2], alpha=0.5, edgecolor=None, label='Inter-field area')
        ax.fill_betweenx(y=[0,6], x1 = x[x2-15], x2 = x[x2+15], color=colors[3], alpha=0.5, edgecolor=None, label='New field')

        if i == 0: 
            ax.legend()
    
    t = np.linspace(0,1800, 11)
    t_smooth = np.linspace(0,1800, 1000)
    model_1 = TwoPiecesPiecewiseSigmoidRegression()
    model_1.fit(t, ys[:, x1], k=0.05)
    y1 = model_1.predict(t_smooth)

    y_mid = get_y(Gaussian(t_smooth-t[np.nanargmax(ys[:, x_mid])], sigma=sigma*5, nx=96), ampl=np.nanmax(ys[:, x_mid]))

    model_2 = TwoPiecesPiecewiseSigmoidRegression()
    model_2.fit(t, ys[:, x2], k=0.05)
    y2 = model_2.predict(t_smooth)

    ax_right = Clear_Axes(fig.add_subplot(grid[:, 1]), close_spines=['top', 'right', 'left'])
    ax_right.plot(y1, t_smooth, label='Previous field', color=colors[1], linewidth=0.8)
    ax_right.plot(y_mid, t_smooth, label='Inter-field area', ls='--', color=colors[2], linewidth=0.8)
    ax_right.plot(y2, t_smooth, label='New field', ls=':', color=colors[3], linewidth=0.8)
    ax_right.legend()
    ax_right.invert_yaxis()
    ax_right.spines['bottom'].set_position('zero')
    ax_right.set_xlim([-0.2, 3.2])
    ax_right.set_xticks([0, 3])

    plt.tight_layout()
    plt.savefig(join(loc, file_name+'.png'), dpi=600)
    plt.savefig(join(loc, file_name+'.svg'), dpi=600)
    plt.close()

# Coordinate drift model
def gradual_shift_hypothesis(file_name: str = join(loc, 'guadual shift hypothesis')):
    x = np.linspace(0,11,1101)
    xs = np.concatenate([[2,2], np.linspace(2, 9, 9)[1:-1], [9,9]])

    y = np.zeros((11, len(x)))

    for i in range(11):
        y[i, :] = get_y(Gaussian(x-xs[i], sigma=sigma, nx=1), ampl=3)

    plot_multi_line(x, 201, 901, ys=y, file_name=file_name)


# Coordinate drift model
def coordinate_hypothesis(file_name: str = join(loc, 'coordinate plastic events hypothesis')):
    x = np.linspace(0,11,1101)
    xs = np.concatenate([[2,2,2,2,2], [9,9,9,9,9,9]])

    y = np.zeros((11, len(x)))

    for i in range(11):
        y[i, :] = get_y(Gaussian(x-xs[i], sigma=sigma, nx=1), ampl=3)

    plot_multi_line(x, 201, 901, ys=y, file_name=file_name)


# Coordinate drift model
def independent_hypothesis(file_name: str = join(loc, 'independent plastic events hypothesis')):
    x = np.linspace(0,11,1101)
    xs1 = np.array([2,2,2,2,2,2,2,2,-3,-3,-3])
    xs2 = np.array([-3,-3,-3,9,9,9,9,9,9,9,9])

    y = np.zeros((11, len(x)))

    for i in range(11):
        if i in [0,1,2]:
            y[i, :] = get_y(Gaussian(x-xs1[i], sigma=sigma, nx=1), ampl=3)
        elif i in [3,4,5,6,7]:
            y[i, :] = get_y(Gaussian(x-xs1[i], sigma=sigma, nx=1), ampl=3) + get_y(Gaussian(x-xs2[i], sigma=sigma, nx=1), ampl=3)
        elif i in [8,9,10]:
            y[i, :] = get_y(Gaussian(x-xs2[i], sigma=sigma, nx=1), ampl=3)

    plot_multi_line(x, 201, 901, ys=y, file_name=file_name)


gradual_shift_hypothesis()
coordinate_hypothesis()
independent_hypothesis()