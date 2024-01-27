from mylib.statistic_test import *

code_id = '0038 - Half-half or Odd-even correlation'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Half-half Correlation', 'Odd-even Correlation', 'Cell Type'], f = f1, 
                              function = InterSessionCorrelation_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

def clear_nan_value(data):
    idx = np.where(np.isnan(data))[0]
    return np.delete(data, idx)
idx = np.where((Data['MiceID'] != 11095)&(Data['MiceID'] != 11092)&(Data['Cell Type'] == 1))[0]
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

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,3))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[2], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.concatenate([np.where((Data['Stage'] == 'PRE')&(Data['Training Day'] == s))[0] for s in uniq_s])
SubData1 = SubDict(Data, Data.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Half-half Correlation',
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
SubSample = SubDict(SubData1, SubData1.keys(), np.random.choice(np.arange(len(SubData1['Half-half Correlation'])), replace=True, size=10000))
sns.stripplot(
    x='Training Day',
    y='Half-half Correlation',
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
    y='Half-half Correlation',
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
SubSample = SubDict(SubData2, SubData2.keys(), np.random.choice(np.arange(len(SubData2['Half-half Correlation'])), replace=True, size=10000))
idx = np.concatenate([np.where((SubSample['Stage'] == 'Stage 2')&(SubSample['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1']])
SubSample = SubDict(SubSample, SubSample.keys(), idx)
sns.stripplot(
    x='Training Day',
    y='Half-half Correlation',
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
    y='Half-half Correlation',
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
SubSample = SubDict(SubData3, SubData3.keys(), np.random.choice(np.arange(len(SubData3['Half-half Correlation'])), replace=True, size=10000))
idx = np.concatenate([np.where((SubSample['Stage'] == 'Stage 2')&(SubSample['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1', 'Maze 2']])
SubSample = SubDict(SubSample, SubSample.keys(), idx)
sns.stripplot(
    x='Training Day',
    y='Half-half Correlation',
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
plt.savefig(join(loc, 'Half-half Correlation.png'), dpi=2400)
plt.savefig(join(loc, 'Half-half Correlation.svg'), dpi=2400)
plt.close()


fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,3))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[2], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.concatenate([np.where((Data['Stage'] == 'PRE')&(Data['Training Day'] == s))[0] for s in uniq_s])
SubData1 = SubDict(Data, Data.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Odd-even Correlation',
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
SubSample = SubDict(SubData1, SubData1.keys(), np.random.choice(np.arange(len(SubData1['Odd-even Correlation'])), replace=True, size=10000))
sns.stripplot(
    x='Training Day',
    y='Odd-even Correlation',
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
    y='Odd-even Correlation',
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
SubSample = SubDict(SubData2, SubData2.keys(), np.random.choice(np.arange(len(SubData2['Odd-even Correlation'])), replace=True, size=10000))
idx = np.concatenate([np.where((SubSample['Stage'] == 'Stage 1')&(SubSample['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1']])
SubSample = SubDict(SubSample, SubSample.keys(), idx)
sns.stripplot(
    x='Training Day',
    y='Odd-even Correlation',
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
    y='Odd-even Correlation',
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
SubSample = SubDict(SubData3, SubData3.keys(), np.random.choice(np.arange(len(SubData3['Odd-even Correlation'])), replace=True, size=10000))
idx = np.concatenate([np.where((SubSample['Stage'] == 'Stage 2')&(SubSample['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1', 'Maze 2']])
SubSample = SubDict(SubSample, SubSample.keys(), idx)
sns.stripplot(
    x='Training Day',
    y='Odd-even Correlation',
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
plt.savefig(join(loc, 'Odd-even Correlation.png'), dpi=2400)
plt.savefig(join(loc, 'Odd-even Correlation.svg'), dpi=2400)
plt.close()