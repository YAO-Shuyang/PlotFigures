from mylib.statistic_test import *
from scipy.stats import f_oneway

code_id = '0831 - Starting Cell Encoded Route Density Distribution'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Route', 'Number'],
                              f=f2, file_idx=np.where(f2['MiceID'] != 10209)[0],
                              function = StartingCellEncodedRouteDensityDistribution_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

D = GetDMatrices(1, 48)
ratio = 100 * np.max(D) / 888
print(ratio)

datas = [Data['Number'][Data['Route'] == i] for i in range(7)] + [Data['Number'][Data['Route'] == i] for i in range(8, 10)]
print(f_oneway(*datas))
print_estimator(Data['Number'][Data['Route'] == 7])
print_estimator(Data['Number'][Data['Route'] != 7])
print(ttest_ind(Data['Number'][Data['Route'] == 7], Data['Number'][Data['Route'] != 7], equal_var=False))

fig = plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
order = [0, 4, 1, 5, 2, 6, 3]
sns.barplot(
    x="Route",
    y="Number",
    data=Data,
    ax=ax,
    hue="Route",
    palette=[DSPPalette[0], DSPPalette[1], DSPPalette[2], DSPPalette[3], DSPPalette[0],
              DSPPalette[0], DSPPalette[4], DSPPalette[5], DSPPalette[6], DSPPalette[0]],
    width=0.8,
    capsize=0.3,
    err_kws={"linewidth": 0.5, 'color':'k'},
    edgecolor = None,
    zorder=2
)
ax.set_ylim(0, 0.2)
ax.set_yticks(np.linspace(0, 0.2, 11))
plt.savefig(join(loc, "density.svg"))
plt.savefig(join(loc, "density.png"), dpi=600)
plt.show()
