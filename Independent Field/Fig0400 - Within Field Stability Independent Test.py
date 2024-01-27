from mylib.statistic_test import *

code_id = "0400 - Within Field Stability"
loc = join(figpath, "Independent Field", code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, 'Field Pool.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['FSC Stability', 'OEC Stability', 'Field Size', 'Field Length', 'Peak Rate', 'Position'], f = f1,
                              function = WithinFieldBasicInfo_Interface, func_kwgs = {'is_placecell': True},
                              file_name = 'Field Pool', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, 'Field Pool.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata, 'Field Statistics in Cell Pool.pkl')) == False:
    CellData = DataFrameEstablish(variable_names = ['Mean FSC', 'Std. FSC', 'Median FSC', 'Error FSC',
                                                'Mean OEC', 'Std. OEC', 'Median OEC', 'Error OEC',
                                                'Mean Size', 'Std. Size', 'Median Size', 'Error Size',
                                                'Mean Length', 'Std. Length', 'Median Length', 'Error Length',
                                                'Mean Rate', 'Std. Rate', 'Median Rate', 'Error Rate',
                                                'Mean Position', 'Std. Position', 'Median Position', 'Error Position',
                                                'Mean Interdistance', 'Std. Interdistance', 'Median Interdistance', 'Error Interdistance',
                                                'Cell ID', 'Field Number'], f = f1,
                              function = WithinCellFieldStatistics_Interface, func_kwgs = {'is_placecell': True},
                              file_name = 'Field Statistics in Cell Pool', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, 'Field Statistics in Cell Pool.pkl'), 'rb') as handle:
        CellData = pickle.load(handle)
    
idx = np.where((Data['MiceID'] != 11095)&(Data['MiceID'] != 11092))[0]
Data = SubDict(Data, Data.keys(), idx)

# Mean Rate
uniq_day = ['Day 1', 'Day 2', 'Day 3', 'Day 4',
            'Day 5', 'Day 6', 'Day 7', 'Day 8',
            'Day 9', '>=Day 10']
uniq_s = ['S'+str(i) for i in range(1,20)] + ['>=S20']

idx = np.concatenate([np.where(Data['Training Day'] == day)[0] for day in uniq_day])
SubData = SubDict(Data, Data.keys(), idx)

colors = sns.color_palette("rocket", 3)
markercolors = [sns.color_palette("crest", 4)[2], sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[0]]

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,2))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[2], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.concatenate([np.where((Data['Stage'] == 'PRE')&(Data['Training Day'] == s))[0] for s in uniq_s])
SubData1 = SubDict(Data, Data.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='FSC Stability',
    data=SubData1,
    hue='Maze Type',
    palette=colors,
    marker='o',
    ax=ax1,
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
SubSample = SubDict(SubData1, SubData1.keys(), np.random.choice(np.arange(len(SubData1['FSC Stability'])), replace=True, size=10000))
sns.stripplot(
    x='Training Day',
    y='FSC Stability',
    hue='Maze Type',
    data=SubSample,
    palette=markercolors,
    edgecolor='black',
    size=1,
    linewidth=0.05,
    ax = ax1,
    dodge=True,
    jitter=0.1
)
ax1.set_ylim([-1, 1])
ax1.set_yticks(np.linspace(-1, 1, 11))

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 1')&(SubData['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1']])
SubData2 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='FSC Stability',
    data=SubData2,
    hue='Maze Type',
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax2
)
SubSample = SubDict(SubData2, SubData2.keys(), np.random.choice(np.arange(len(SubData2['FSC Stability'])), replace=True, size=10000))
sns.stripplot(
    x='Training Day',
    y='FSC Stability',
    hue='Maze Type',
    data=SubSample,
    palette=markercolors,
    edgecolor='black',
    size=1,
    linewidth=0.05,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax2.set_ylim([-1, 1])
ax2.set_yticks(np.linspace(-1, 1, 11))

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 2')&(SubData['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1', 'Maze 2']])
SubData3 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='FSC Stability',
    data=SubData3,
    hue='Maze Type',
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax3
)
SubSample = SubDict(SubData3, SubData3.keys(), np.random.choice(np.arange(len(SubData3['FSC Stability'])), replace=True, size=10000))
sns.stripplot(
    x='Training Day',
    y='FSC Stability',
    hue='Maze Type',
    data=SubSample,
    palette=markercolors,
    edgecolor='black',
    size=1,
    linewidth=0.05,
    ax = ax3,
    dodge=True,
    jitter=0.1
)
ax3.set_ylim([-1, 1])
ax3.set_yticks(np.linspace(-1, 1, 11))
plt.tight_layout()
plt.savefig(join(loc, 'FSC Stability.png'), dpi=2400)
plt.savefig(join(loc, 'FSC Stability.svg'), dpi=2400)
plt.close()

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,2))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[2], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.concatenate([np.where((Data['Stage'] == 'PRE')&(Data['Training Day'] == s))[0] for s in uniq_s])
SubData1 = SubDict(Data, Data.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='OEC Stability',
    data=SubData1,
    hue='Maze Type',
    palette=colors,
    marker='o',
    ax=ax1,
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
SubSample = SubDict(SubData1, SubData1.keys(), np.random.choice(np.arange(len(SubData1['OEC Stability'])), replace=True, size=10000))
sns.stripplot(
    x='Training Day',
    y='OEC Stability',
    hue='Maze Type',
    data=SubSample,
    palette=markercolors,
    edgecolor='black',
    size=1,
    linewidth=0.05,
    ax = ax1,
    dodge=True,
    jitter=0.1
)
ax1.set_ylim([-1, 1])
ax1.set_yticks(np.linspace(-1, 1, 11))

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 1')&(SubData['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1']])
SubData2 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='OEC Stability',
    data=SubData2,
    hue='Maze Type',
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax2
)
SubSample = SubDict(SubData2, SubData2.keys(), np.random.choice(np.arange(len(SubData2['OEC Stability'])), replace=True, size=10000))
sns.stripplot(
    x='Training Day',
    y='OEC Stability',
    hue='Maze Type',
    data=SubSample,
    palette=markercolors,
    edgecolor='black',
    size=1,
    linewidth=0.05,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax2.set_ylim([-1, 1])
ax2.set_yticks(np.linspace(-1, 1, 11))

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 2')&(SubData['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1', 'Maze 2']])
SubData3 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='OEC Stability',
    data=SubData3,
    hue='Maze Type',
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax3
)
SubSample = SubDict(SubData3, SubData3.keys(), np.random.choice(np.arange(len(SubData3['OEC Stability'])), replace=True, size=10000))
sns.stripplot(
    x='Training Day',
    y='OEC Stability',
    hue='Maze Type',
    data=SubSample,
    palette=markercolors,
    edgecolor='black',
    size=1,
    linewidth=0.05,
    ax = ax3,
    dodge=True,
    jitter=0.1
)
ax3.set_ylim([-1, 1])
ax3.set_yticks(np.linspace(-1, 1, 11))
plt.tight_layout()
plt.savefig(join(loc, 'OEC Stability.png'), dpi=2400)
plt.savefig(join(loc, 'OEC Stability.svg'), dpi=2400)
plt.close()


fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,2))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right', 'left'], ifxticks=True)
x_ticks = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']
s_ticks = ['S'+str(i) for i in range(1, 20)] + ['>=S20']    

colors = sns.color_palette("rocket", 4)
markercolors = sns.color_palette("Blues", 4)
    
d_ticks = ['Day '+str(i) for i in range(4, 10)] + ['>=Day 10']
idx = np.concatenate([np.where((Data['Training Day'] == day)&(Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 1'))[0] for day in x_ticks])

stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage1_indices)
stage1_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1']])
SubData = SubDict(SubData, SubData.keys(), idx=stage1_indices)
sns.barplot(
    x='Training Day',
    y='OEC Stability',
    data=SubData,
    hue='Maze Type',
    palette=colors,
    capsize=0.2,
    errwidth=0.8,
    errcolor='black',
    ax=ax1
)
ax1.set_ylim([0,1])

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.barplot(
    x='Training Day',
    y='OEC Stability',
    data=SubData,
    hue='Maze Type',
    palette=colors,
    capsize=0.1,
    errwidth=0.8,
    errcolor='black',
    ax=ax2
)
ax2.set_ylim([0,1])
plt.savefig(join(loc, 'Within Field OEC Stability.png'), dpi=600)
plt.savefig(join(loc, 'Within Field OEC Stability.svg'), dpi=600)
plt.close()
