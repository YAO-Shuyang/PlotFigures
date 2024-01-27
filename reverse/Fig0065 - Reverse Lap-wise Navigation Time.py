from mylib.statistic_test import *
from matplotlib.gridspec import GridSpec

code_id = '0065 - Reverse Lap-wise Navigation Time'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Lap ID', 'Direction', 'Lap-wise time cost'], is_behav=True,
                              f = f3_behav, function = LearningCurve_Reverse_Interface, 
                              file_name = code_id, behavior_paradigm = 'ReverseMaze')

color = [sns.color_palette('Blues', 9)[5], sns.color_palette('YlOrRd', 9)[5]]
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x="Training Day",
    y="Lap-wise time cost",
    hue="Direction",
    data=Data,
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
plt.savefig(os.path.join(loc, 'Navigation Time.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Navigation Time.svg'), dpi=600)
plt.close()