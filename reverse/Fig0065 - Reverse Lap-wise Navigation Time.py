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

if os.path.exists(join(figdata, code_id+' [HP].pkl')):
    with open(join(figdata, code_id+' [HP].pkl'), 'rb') as handle:
        HPData = pickle.load(handle)
else:
    HPData = DataFrameEstablish(variable_names = ['Lap ID', 'Direction', 'Lap-wise time cost'], is_behav=True,
                              f = f4_behav, function = LearningCurve_Reverse_Interface, 
                              file_name = code_id+' [HP]', behavior_paradigm = 'HairpinMaze')

dates = np.unique(Data['Training Day'])
idx = np.where()

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
    estimator=np.median,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
sns.stripplot(
    x="Training Day",
    y="Lap-wise time cost",
    hue="Direction",
    data=Data,
    palette=color,
    edgecolor='black',
    size=3,
    linewidth=0.05,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 120)
ax.set_yticks(np.linspace(0, 120, 7))
plt.savefig(os.path.join(loc, 'Navigation Time.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Navigation Time.svg'), dpi=600)
plt.close()

fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x="Training Day",
    y="Lap-wise time cost",
    hue="Direction",
    data=HPData,
    err_style='bars',
    estimator=np.median,
    palette=color,
    ax=ax, 
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
ax.set_ylim(0, 120)
ax.set_yticks(np.linspace(0, 120, 7))
plt.savefig(os.path.join(loc, 'Hairpin Maze - Navigation Time.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Hairpin Maze - Navigation Time.svg'), dpi=600)
plt.close()