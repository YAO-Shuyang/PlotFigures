from mylib.statistic_test import *

code_id = '0007 - Cell Number'
p = os.path.join(figpath, code_id)
mkdir(p)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Cell Number'], f = f1, function = CellNumber_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

idx = np.where((Data['MiceID']!=11092)&(Data['MiceID']!=11094)&(Data['MiceID']!=11095))[0]
Data = SubDict(Data, Data.keys(), idx=idx)

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,3.2))
ax_pre, ax1, ax2 = axes
colors = sns.color_palette("rocket", 3)
markercolors = sns.color_palette("Blues", 3)

ax_pre = Clear_Axes(ax_pre, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax1 = Clear_Axes(ax1, close_spines=['top', 'right', 'left'], ifxticks=True)
ax2 = Clear_Axes(ax2, close_spines=['top', 'right', 'left'], ifxticks=True)

x_ticks = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']
s_ticks = ['S'+str(i) for i in range(1, 20)] + ['>=S20']

y_max = np.nanmax(Data['Cell Number'])

pre_indices = np.concatenate([np.where((Data['Stage'] == 'PRE')&(Data['Training Day'] == session))[0] for session in s_ticks])
SubData = SubDict(Data, Data.keys(), idx=pre_indices)

sns.lineplot(
    x='Training Day',
    y='Cell Number',
    hue='Maze Type',
    data=SubData,
    #style=Data['MiceID'][idx],
    palette=colors,
    err_style='bars',
    ax=ax_pre, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
"""
sns.stripplot(
    x = 'Training Day',
    y = 'Cell Number',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax_pre,
    dodge=True
)
"""
ax_pre.set_ylim([0, y_max])
ax_pre.set_yticks(ColorBarsTicks(peak_rate=y_max, is_auto=True, tick_number=5))

stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage1_indices)
stage1_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1']])
SubData = SubDict(SubData, SubData.keys(), idx=stage1_indices)
sns.lineplot(
    x='Training Day',
    y='Cell Number',
    hue='Maze Type',
    data=SubData,
    #style=Data['MiceID'][idx],
    palette=colors,
    err_style='bars',
    ax=ax1, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
"""
sns.stripplot(
    x = 'Training Day',
    y = 'Cell Number',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax1,
    dodge=True,
)
"""
ax1.set_ylim([0, y_max])

print("Stage 1 Subdata:")
print(f"  Min {min(SubData['Cell Number'])}, Max: {max(SubData['Cell Number'])}")
print(f"  mean {np.nanmean(SubData['Cell Number'])}, ±std {np.nanstd(SubData['Cell Number'])}")

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.lineplot(
    x='Training Day',
    y='Cell Number',
    hue='Maze Type',
    data=SubData,
    #style=Data['MiceID'][idx],
    palette=colors,
    err_style='bars',
    ax=ax2, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
"""
sns.stripplot(
    x = 'Training Day',
    y = 'Cell Number',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax2,
    dodge=True
)
"""
ax2.set_ylim([0, y_max])

print("Stage 2 Subdata:")
print(f"  Min {min(SubData['Cell Number'])}, Max: {max(SubData['Cell Number'])}")
print(f"  mean {np.nanmean(SubData['Cell Number'])}, ±std {np.nanstd(SubData['Cell Number'])}")
plt.tight_layout()
plt.savefig(join(p, 'Cell Number.png'), dpi=600)
plt.savefig(join(p, 'Cell Number.svg'), dpi=600)
plt.close()
