# Fig0023-1, Place cell percentage - Linegraph

from mylib.statistic_test import *
code_id = '0023 - Place cell percentage'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['percentage', 'place cell num', 'total cell num'], f = f1, function = PlaceCellPercentage_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')   
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,3))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right', 'left'], ifxticks=True)
ax3 = Clear_Axes(axes[2], close_spines=['top', 'right', 'left'], ifxticks=True)

x_ticks = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']
s_ticks = ['S'+str(i) for i in range(1, 20)] + ['>=S20']

idx = np.concatenate([np.where(Data['Training Day'] == session)[0] for session in s_ticks] + [np.where(Data['Training Day'] == day)[0] for day in x_ticks])
Data = SubDict(Data, Data.keys(), idx)
Data['percentage'] = Data['percentage']*100

colors = sns.color_palette("rocket", 5)[1:-1]
markercolors = sns.color_palette("Blues", 4)[1::]
idx = np.where(Data['Stage'] == 'PRE')[0]
SubData = SubDict(Data, Data.keys(), idx)

data1 = SubData['percentage'][np.where(SubData['Training Day'] == 'S1')[0]]
print(np.mean(data1), np.std(data1))
data10 = SubData['percentage'][np.where(SubData['Training Day'] == '>=S20')[0]]
print(np.mean(data10), np.std(data10))
print(levene(data1, data10))
print(ttest_ind(data1, data10, alternative='less'))

sns.lineplot(
    x='Training Day',
    y='percentage',
    data=SubData,
    hue='Maze Type',
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=4,
    #legend=False,
    err_style='bars',
    err_kws={'capsize':3},
    linewidth=1,
    ax=ax1
)
sns.stripplot(
    x = 'Training Day',
    y = 'percentage',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    size=2,
    ax = ax1,
    jitter=0.3
)
ax1.set_ylim([0,100])
ax1.set_yticks(np.linspace(0, 100, 6))

stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage1_indices)
stage1_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1']])
SubData = SubDict(SubData, SubData.keys(), idx=stage1_indices)
sns.lineplot(
    x='Training Day',
    y='percentage',
    data=SubData,
    hue='Maze Type',
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=4,
    err_style='bars',
    err_kws={'capsize':3},
    linewidth=1,
    ax=ax2
)
sns.stripplot(
    x = 'Training Day',
    y = 'percentage',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    size=2,
    ax = ax2,
    dodge=True,
    jitter=0.2
)
ax2.set_ylim([0,100])

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.lineplot(
    x='Training Day',
    y='percentage',
    data=SubData,
    hue='Maze Type',
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=4,
    err_style='bars',
    err_kws={'capsize':3},
    linewidth=1,
    ax=ax3
)
sns.stripplot(
    x = 'Training Day',
    y = 'percentage',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    size=2,
    ax = ax3,
    dodge=True,
    jitter=0.1
)
plt.savefig(join(loc, 'place cell percentage.png'), dpi=600)
plt.savefig(join(loc, 'place cell percentage.svg'), dpi=600)
plt.close()

