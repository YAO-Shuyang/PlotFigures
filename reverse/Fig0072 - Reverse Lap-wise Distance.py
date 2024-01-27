from mylib.statistic_test import *
from matplotlib.gridspec import GridSpec

code_id = '0072 - Reverse Lap-wise Distance'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+' [Reverse].pkl')):
    with open(join(figdata, code_id+' [Reverse].pkl'), 'rb') as handle:
        RData = pickle.load(handle)
else:
    RData = DataFrameEstablish(variable_names = ['Lap-wise Distance', 'LapID'], is_behav=True,
                              f = f3_behav, function = LapwiseDistance_Reverse_Interface, 
                              file_name = code_id +' [Reverse]', behavior_paradigm = 'ReverseMaze')

if os.path.exists(join(figdata, code_id+' [Hairpin].pkl')):
    with open(join(figdata, code_id+' [Hairpin].pkl'), 'rb') as handle:
        HData = pickle.load(handle)
else:
    HData = DataFrameEstablish(variable_names = ['Lap-wise Distance', 'LapID'], is_behav=True,
                              f = f4_behav, function = LapwiseDistance_Reverse_Interface, 
                              file_name = code_id + ' [Hairpin]', behavior_paradigm = 'HairpinMaze')

color = [sns.color_palette('Blues', 9)[5], sns.color_palette('YlOrRd', 9)[5]]
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x="Training Day",
    y="Lap-wise Distance",
    data=HData,
    err_style='bars',
    palette=color,
    ax=ax, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
ax.set_ylim(0, 120)
ax.set_yticks(np.linspace(0, 120, 7))
plt.savefig(os.path.join(loc, 'Hairpin Distance.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Haripin Distance.svg'), dpi=600)
plt.close()