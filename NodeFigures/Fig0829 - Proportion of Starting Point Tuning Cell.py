from mylib.statistic_test import *
from mylib.field.tracker_v2 import Tracker2d

code_id = '0829 - Proportion of Starting Point Tuning Cell'
loc = join(figpath, "Dsp", code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Proportion'], file_idx=np.where(f2['MiceID'] != 10209)[0],
                             f = f2, file_name=code_id, behavior_paradigm="DSPMaze",
                             function = ProportionOfStartingPointTuningCell_Interface)

idx1 = np.where(Data['Training Day'] == 'Day 1')[0]
idx2 = np.where(Data['Training Day'] == 'Day 7')[0]
print_estimator(Data['Proportion'])
print_estimator(Data['Proportion'][idx1])
print_estimator(Data['Proportion'][idx2])
print(ttest_rel(Data['Proportion'][idx1], Data['Proportion'][idx2]))

fig = plt.figure(figsize = (4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Training Day',
    y = 'Proportion',
    data = Data,
    linewidth=0.5,
    err_style='bars',
    err_kws={"linewidth": 0.5, 'capsize': 3, 'capthick': 0.5},
    ax = ax
)
sns.stripplot(
    x = 'Training Day',
    y = 'Proportion',
    data = Data,
    linewidth=0.2,
    ax = ax,
    jitter=0.2,
    size=5,
    edgecolor='black',
    hue="MiceID",
    palette=['#D4C9A8', '#8E9F85', '#C3AED6', '#FED7D7']
)
ax.set_ylim(0,1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, 'Proportion of Starting Point Tuning Cell.svg'))
plt.savefig(join(loc, 'Proportion of Starting Point Tuning Cell.png'), dpi=600)
plt.show()