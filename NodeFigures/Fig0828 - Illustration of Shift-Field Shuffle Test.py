from mylib.statistic_test import *
from mylib.dsp.starting_cell import *


code_id = '0828 - Illustration of Shift-Field Shuffle Test'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

def gaussian(x, mu, sig=1):
    return np.exp(-(x - mu)**2 / (2 * sig**2))

max_length = np.array([CP_DSP[0].shape[0], CP_DSP[1].shape[0], CP_DSP[2].shape[0], CP_DSP[3].shape[0], CP_DSP[0].shape[0], 
                       CP_DSP[0].shape[0], CP_DSP[4].shape[0], CP_DSP[5].shape[0], CP_DSP[6].shape[0], CP_DSP[0].shape[0]])
xs = [np.linspace(0, max_length[i], max_length[i]*1000+1) for i in range(10)]
peaks = np.random.rand(10)*0.3+0.7
peaks[4] = 0
peaks[9] = 0
peaks[0] = 0
centers = np.array([3.2, 2.4, 1.8, 0.9, 2.3, 2.3, 14, 4.5, 3.9, 1.3])

ys = [gaussian(xs[i], centers[i]) * peaks[i] + i*1.2 for i in range(10)]

fig = plt.figure(figsize=(4, 6))
ax = Clear_Axes(plt.axes())
colors = [DSPPalette[0], DSPPalette[1], DSPPalette[2], DSPPalette[3], DSPPalette[0],
          DSPPalette[0], DSPPalette[4], DSPPalette[5], DSPPalette[6], DSPPalette[0]]
for i in range(10):
    ax.plot(xs[i], ys[i], color=colors[i], linewidth=0.5)
ax.set_xlim(0, max_length[0])
ax.axvline(6.5, color='k', linewidth=0.5, ls = ":")
plt.savefig(os.path.join(loc, 'Step 1.svg'), dpi=600)
plt.savefig(os.path.join(loc, 'Step 1.png'), dpi=600)
plt.close()