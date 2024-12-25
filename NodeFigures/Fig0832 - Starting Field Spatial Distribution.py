from mylib.statistic_test import *

code_id = '0832 - Starting Field Spatial Distribution'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+' .pkl')) == False:
    Data = DataFrameEstablish(variable_names = ["Relative Pos", "Density"],
                              f=f2, file_idx=np.where(f2['MiceID']!=10209)[0],
                              function = StartingFieldSpatialDistribution_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
    
fig = plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x="Relative Pos",
    y="Density",
    data=Data,
    hue="Relative Pos",
    palette=["#ABD0D1"],
    ax=ax,
    capsize=0.3,
    err_kws={"linewidth": 0.5, 'color':'k'},
    edgecolor = None,
    zorder=2
)
ax.set_ylim(0, 0.07)
ax.set_yticks(np.linspace(0, 0.07, 8))
plt.savefig(join(loc, "density.svg"))
plt.savefig(join(loc, "density.png"), dpi=600)
plt.show()