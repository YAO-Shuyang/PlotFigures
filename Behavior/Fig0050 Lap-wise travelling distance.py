from mylib.statistic_test import *

code_id = "0050 Lap-wise Traveling Distance"
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata,code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Lap ID', 'Lap-wise Distance'], f = f_pure_behav, 
                              function = LapwiseTravelDistance_Interface, is_behav = True,
                              file_name = code_id, behavior_paradigm = 'CrossMaze', func_kwgs = {'is_placecell':False})
else:
    with open(os.path.join(figdata,code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

uniq_day = ['Day 1', 'Day 2', 'Day 3', 'Day 4',
            'Day 5', 'Day 6', 'Day 7', 'Day 8',
            'Day 9', '>=Day 10']

def TransformData(Data: dict):
    x = np.zeros(Data['Training Day'].shape[0], np.int64)
    Lap1s = np.where(Data['Lap ID'] == 1)[0]
    Lap2s = np.where(Data['Lap ID'] == 2)[0]
    Lap3s = np.where(Data['Lap ID'] == 3)[0]
    Lap4s = np.where(Data['Lap ID'] == 4)[0]
    Laps_remain = np.where(Data['Lap ID'] >= 5)[0]
    x[Lap1s] = 1
    x[Lap2s] = 2
    x[Lap3s] = 3
    x[Lap4s] = 4
    x[Laps_remain] = 5
    for i, d in enumerate(uniq_day):
        idx = np.where(Data['Training Day'] == d)[0]
        x[idx] += i*7
        
    Data['x'] = x
    return Data

Data['Lap-wise Distance'] = Data['Lap-wise Distance'] / 100
Data=TransformData(Data)
"""
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,2))
ax1, ax2 = axes[0], axes[1]
ax1 = Clear_Axes(ax1, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(ax2, close_spines=['top', 'right', 'left'], ifxticks=True)
x = np.unique(Data['Training Day'])

colors = sns.color_palette("rocket", 4)
colors = [colors[0], colors[-1]]

for day in uniq_day:
    idx = np.where((Data['Training Day'] == day)&(Data['Stage'] == 'Stage 1'))[0]
    sns.lineplot(
        x=Data['x'][idx],
        y=Data['Lap-wise Distance'][idx],
        hue=Data['Maze Type'][idx],
        #style=Data['MiceID'][idx],
        palette=colors,
        #err_style='bars',
        ax=ax1, 
        legend=False,
        #err_kws={'elinewidth':1, 'capthick':1, 'capsize':2},
        linewidth=0.7,
        alpha=0.6,
        err_kws={'edgecolor':None},
    )

ax1.set_xticks(np.linspace(4,67,10)-1, uniq_day)
ax1.set_ylim([0,250])

for day in uniq_day:
    idx = np.where((Data['Training Day'] == day)&(Data['Stage'] == 'Stage 2'))[0]
    sns.lineplot(
        x=Data['x'][idx],
        y=Data['Lap-wise Distance'][idx],
        hue=Data['Maze Type'][idx],
        #style=Data['MiceID'][idx],
        palette=colors,
        #err_style='bars',
        ax=ax2, 
        legend=False,
        #err_kws={'elinewidth':1, 'capthick':1, 'capsize':2},
        linewidth=0.7,
        alpha=0.6,
        err_kws={'edgecolor':None},
    )


ax2.set_xticks(np.linspace(4,67,10)-1, uniq_day)
ax2.set_ylim([0,250])
"""
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,2))
ax1, ax2 = axes[0], axes[1],
ax1 = Clear_Axes(ax1, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(ax2, close_spines=['top', 'right'], ifxticks=True)
x = np.unique(Data['Training Day'])

colors = sns.color_palette("rocket", 4)
colors = [colors[0], colors[-1]]
markercolors = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[0]]

idx = np.where(Data['Stage'] == 'Stage 1')[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
sns.lineplot(
    x='Training Day',
    y='Lap-wise Distance',
    hue='Maze Type',
    data=SubData,
    #style=Data['MiceID'][idx],
    palette=colors,
    err_style='bars',
    ax=ax1, 
    marker='o',
    estimator=np.median,
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax1.set_ylim([0,120])
ax1.set_yticks(np.linspace(0, 120, 4))

idx = np.where(Data['Stage'] == 'Stage 2')[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
sns.lineplot(
    x='Training Day',
    y='Lap-wise Distance',
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
    estimator=np.median,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax2.set_ylim([5,10])
ax2.set_yticks(np.linspace(5, 10, 6))
plt.savefig(join(loc, 'travel distance.png'), dpi=600)
plt.savefig(join(loc, 'travel distance.svg'), dpi=600)
plt.close()


grid = GridSpec(2,2, height_ratios=[1,3])
fig = plt.figure(figsize=(4,2))
ax1 = plt.subplot(grid[:, 0])
ax2 = plt.subplot(grid[0, 1])
ax3 = plt.subplot(grid[1, 1])
ax1 = Clear_Axes(ax1, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(ax2, close_spines=['top', 'right', 'bottom'], ifyticks=True)
ax3 = Clear_Axes(ax3, close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1'))
sns.lineplot(
    x=Data['x'][idx]-1,
    y=Data['Lap-wise Distance'][idx],
    hue=Data['Maze Type'][idx],
    #style=Data['MiceID'][idx],
    palette=colors,
    err_style='bars',
    ax=ax1, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    #err_kws={'elinewidth':1, 'capthick':1, 'capsize':2},
    linewidth=0.5,
    estimator=np.median,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
)
sns.stripplot(
    x=Data['x'][idx],
    y=Data['Lap-wise Distance'][idx],
    hue=Data['Maze Type'][idx],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.05,
    ax = ax1,
    dodge=True,
    jitter=0.1
)
ax1.set_ylim([0, 300])

idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10'))
sns.lineplot(
    x=Data['x'][idx]-64.4,
    y=Data['Lap-wise Distance'][idx],
    hue=Data['Maze Type'][idx],
    #style=Data['MiceID'][idx],
    palette=colors,
    err_style='bars',
    ax=ax3, 
    marker='o',
    markeredgecolor=None,
    markersize=1,
    legend=False,
    #err_kws={'elinewidth':1, 'capthick':1, 'capsize':2},
    linewidth=0.5,
    estimator=np.median,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Lap-wise Distance']<=10))
ax3.axhline(y=np.nanmin(Data['Lap-wise Distance'][idx]), color=markercolors[0], linestyle=':')
minidx = np.argmin(Data['Lap-wise Distance'][np.where((Data['Maze Type'] == 'Maze 1')&(Data['Stage'] == 'Stage 1'))[0]])
print("Stage 1 min distance:", 
      Data['Training Day'][np.where((Data['Maze Type'] == 'Maze 1')&(Data['Stage'] == 'Stage 1'))[0]][minidx], 
      Data['Lap-wise Distance'][np.where((Data['Maze Type'] == 'Maze 1')&(Data['Stage'] == 'Stage 1'))[0]][minidx], 
      Data['Lap ID'][np.where((Data['Maze Type'] == 'Maze 1')&(Data['Stage'] == 'Stage 1'))[0]][minidx])
sns.stripplot(
    x=Data['x'][idx],
    y=Data['Lap-wise Distance'][idx],
    hue=Data['Maze Type'][idx],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.05,
    ax = ax3,
    dodge=True,
    jitter=0.1
)

idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Lap-wise Distance']>10))
sns.stripplot(
    x=Data['x'][idx],
    y=Data['Lap-wise Distance'][idx],
    hue=Data['Maze Type'][idx],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.05,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax3.set_ylim([6, 10])
ax3.set_yticks(np.linspace(6, 10, 5))
ax3.set_xticks([-0.2, 0.8, 1.8, 2.8, 3.8], labels=[1, 2, 3, 4, 5])
ax3.set_xlim([-1, 4.5])
ax2.set_ylim([5,30])
ax2.set_yticks([10, 30])
ax2.set_xlim([-1, 4.5])
plt.tight_layout()
plt.savefig(join(loc, 'travel distance [Zoom out Stage 1].png'), dpi=600)
plt.savefig(join(loc, 'travel distance [Zoom out Stage 1].svg'), dpi=600)
plt.close()





grid = GridSpec(3,1, height_ratios=[3,2,6])
fig = plt.figure(figsize=(4,7))
ax1 = plt.subplot(grid[0])
ax2 = plt.subplot(grid[1])
ax3 = plt.subplot(grid[2])
ax1 = Clear_Axes(ax1, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(ax2, close_spines=['top', 'right', 'bottom'], ifyticks=True)
ax3 = Clear_Axes(ax3, close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1'))
sns.lineplot(
    x=Data['x'][idx]-1,
    y=Data['Lap-wise Distance'][idx],
    hue=Data['Maze Type'][idx],
    #style=Data['MiceID'][idx],
    palette=colors,
    err_style='bars',
    ax=ax1, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    #err_kws={'elinewidth':1, 'capthick':1, 'capsize':2},
    linewidth=0.5,
    estimator=np.median,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
)
sns.stripplot(
    x=Data['x'][idx],
    y=Data['Lap-wise Distance'][idx],
    hue=Data['Maze Type'][idx],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.05,
    ax = ax1,
    dodge=True,
    jitter=0.1
)
ax1.set_ylim([-10, 250])
ax1.set_yticks(np.linspace(0, 250, 6))

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10'))
sns.lineplot(
    x=Data['x'][idx]-64,
    y=Data['Lap-wise Distance'][idx],
    hue=Data['Maze Type'][idx],
    #style=Data['MiceID'][idx],
    palette=colors,
    err_style='bars',
    ax=ax3, 
    marker='o',
    markeredgecolor=None,
    markersize=1,
    legend=False,
    #err_kws={'elinewidth':1, 'capthick':1, 'capsize':2},
    linewidth=0.5,
    estimator=np.median,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap-wise Distance']<=12))
SubData = SubDict(Data, Data.keys(), idx)
idx = np.where((SubData['Maze Type'] == 'Maze 1'))[0]
maze1_minarg = np.argmin(SubData['Lap-wise Distance'][idx])
print("Stage 2 min distance:",
      SubData['Training Day'][idx][maze1_minarg],
      SubData['Lap-wise Distance'][idx][maze1_minarg],
      SubData['Lap ID'][idx][maze1_minarg])
ax3.axhline(y=SubData['Lap-wise Distance'][idx][maze1_minarg], color=markercolors[0], linestyle=':', linewidth=0.5)

idx = np.where((SubData['Maze Type'] == 'Maze 2'))[0]
maze2_minarg = np.argmin(SubData['Lap-wise Distance'][idx])
print("Stage 2 min distance:",
      SubData['Training Day'][idx][maze2_minarg],
      SubData['Lap-wise Distance'][idx][maze2_minarg],
      SubData['Lap ID'][idx][maze2_minarg])
ax3.axhline(y=SubData['Lap-wise Distance'][idx][maze2_minarg], color=markercolors[1], linestyle=':', linewidth=0.5)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap-wise Distance']<=12))
sns.stripplot(
    x=Data['x'][idx],
    y=Data['Lap-wise Distance'][idx],
    hue=Data['Maze Type'][idx],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.05,
    ax = ax3,
    dodge=True,
    jitter=0.1
)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap-wise Distance']>12))
sns.stripplot(
    x=Data['x'][idx],
    y=Data['Lap-wise Distance'][idx],
    hue=Data['Maze Type'][idx],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.05,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax3.set_ylim([5, 13])
ax3.set_yticks(np.linspace(5, 12, 8))
ax3.set_xlim([-1, 5])
ax2.set_ylim([10,32])
ax2.set_yticks([12, 30])
ax2.set_xlim([-1, 5])
plt.tight_layout()
plt.savefig(join(loc, 'travel distance [Zoom out Stage 2].png'), dpi=2400)
plt.savefig(join(loc, 'travel distance [Zoom out Stage 2].svg'), dpi=2400)
plt.close()


# First Session & Second Session
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data_11 = Data['Lap-wise Distance'][idx]
print_estimator(data_11)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
data_21 = Data['Lap-wise Distance'][idx]
print_estimator(data_21)

colors = sns.color_palette("rocket", 4)
colors = [colors[0], colors[-1]]
markercolors = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[0]]

fig = plt.figure(figsize=(2,3))
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('Novel Maze 1', data_11.shape[0]), np.repeat('Novel Maze 2', data_21.shape[0])]),
    'Lap-wise Distance': np.concatenate([data_11, data_21]),
}
sns.barplot(
    x='Condition',
    y='Lap-wise Distance',
    data=compdata,
    palette=colors,
    ax=ax,
    errwidth=0.5,
    capsize=0.3,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x='Condition',
    y='Lap-wise Distance',
    data=compdata,
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    jitter=0.2
)
ax.set_ylim([0,300])
#ax.set_yticks(np.linspace(0, 50, 6))
ax.set_xticks([0, 1], ["Novel\nMaze 1", "Novel\nMaze 2"])
ax.set_ylabel('Lap-wise Travel distance')
plt.tight_layout()
plt.savefig(join(loc, "Comparison of Maze 1 and Maze 2 [S1].png"), dpi=600)
plt.savefig(join(loc, "Comparison of Maze 1 and Maze 2 [S1].svg"), dpi=600)
plt.close()
print("First Session:", levene(data_11, data_21))
print(ttest_ind(data_11, data_21, alternative='greater', equal_var=False), end='\n\n')



# First Exposure\
print("Stage 1 Novel Day 1 -----------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 1))[0]
print("First exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap1 = Data['Lap-wise Distance'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 2))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap2 = Data['Lap-wise Distance'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] >= 5))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap5 = Data['Lap-wise Distance'][idx]

print("Significant test for lap 1 and lap 2:")
print(levene(maze1_lap1, maze1_lap2))
print(ttest_ind(maze1_lap1, maze1_lap2, alternative='greater'))

print("Significant test for lap 1 and lap 5:")
print(levene(maze1_lap1, maze1_lap5))
print(ttest_ind(maze1_lap1, maze1_lap5, equal_var=False, alternative='greater'), end='\n\n')


# First Exposure
print("Stage 1 Familiar >= Day 10 -----------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] == 1))[0]
print("First exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap1 = Data['Lap-wise Distance'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] == 2))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap2 = Data['Lap-wise Distance'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] >= 5))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap5 = Data['Lap-wise Distance'][idx]

print("Significant test for lap 1 and lap 2:")
print(levene(maze1_lap1, maze1_lap2))
print(ttest_ind(maze1_lap1, maze1_lap2, alternative='greater'))

print("Significant test for lap 1 and lap 5:")
print(levene(maze1_lap1, maze1_lap5))
print(ttest_ind(maze1_lap1, maze1_lap5, alternative='greater'), end='\n\n')


# First Exposure\
print("Stage 2 Novel Maze 2 Day 1 -----------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 1)&(Data['Maze Type'] == 'Maze 2'))[0]
print("First exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap1 = Data['Lap-wise Distance'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 2)&(Data['Maze Type'] == 'Maze 2'))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap2 = Data['Lap-wise Distance'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] >= 5)&(Data['Maze Type'] == 'Maze 2'))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap5 = Data['Lap-wise Distance'][idx]

print("Significant test for lap 1 and lap 2:")
print(levene(maze1_lap1, maze1_lap2))
print(ttest_ind(maze1_lap1, maze1_lap2, alternative='greater'))

print("Significant test for lap 1 and lap 5:")
print(levene(maze1_lap1, maze1_lap5))
print(ttest_ind(maze1_lap1, maze1_lap5, equal_var=False, alternative='greater'), end='\n\n')


# First Exposure
print("Stage 2 Familiar Maze 2 >= Day 10 -----------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] == 1)&(Data['Maze Type'] == 'Maze 2'))[0]
print("First exposure: ")
print_estimator(Data['Lap-wise Distance'][idx], end='\n\n')
maze1_lap1 = Data['Lap-wise Distance'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] == 2)&(Data['Maze Type'] == 'Maze 2'))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap2 = Data['Lap-wise Distance'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] >= 5)&(Data['Maze Type'] == 'Maze 2'))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap5 = Data['Lap-wise Distance'][idx]

print("Significant test for lap 1 and lap 2:")
print(levene(maze1_lap1, maze1_lap2))
print(ttest_ind(maze1_lap1, maze1_lap2, alternative='greater'))

print("Significant test for lap 1 and lap 5:")
print(levene(maze1_lap1, maze1_lap5))
print(ttest_ind(maze1_lap1, maze1_lap5, alternative='greater'), end='\n\n')

# First Exposure\
print("Stage 2 Novel Maze 1 Day 1 -----------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 1)&(Data['Maze Type'] == 'Maze 1'))[0]
print("First exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap1 = Data['Lap-wise Distance'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 2)&(Data['Maze Type'] == 'Maze 1'))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap2 = Data['Lap-wise Distance'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] >= 5)&(Data['Maze Type'] == 'Maze 1'))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap5 = Data['Lap-wise Distance'][idx]

print("Significant test for lap 1 and lap 2:")
print(levene(maze1_lap1, maze1_lap2))
print(ttest_ind(maze1_lap1, maze1_lap2, alternative='greater'))

print("Significant test for lap 1 and lap 5:")
print(levene(maze1_lap1, maze1_lap5))
print(ttest_ind(maze1_lap1, maze1_lap5, equal_var=False, alternative='greater'), end='\n\n')


# First Exposure
print("Stage 2 Familiar Maze 1 >= Day 10 -----------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] == 1)&(Data['Maze Type'] == 'Maze 1'))[0]
print("First exposure: ")
print_estimator(Data['Lap-wise Distance'][idx], end='\n\n')
maze1_lap1 = Data['Lap-wise Distance'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] == 2)&(Data['Maze Type'] == 'Maze 1'))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap2 = Data['Lap-wise Distance'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] >= 5)&(Data['Maze Type'] == 'Maze 1'))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise Distance'][idx])
maze1_lap5 = Data['Lap-wise Distance'][idx]

print("Significant test for lap 1 and lap 2:")
print(levene(maze1_lap1, maze1_lap2))
print(ttest_ind(maze1_lap1, maze1_lap2, alternative='greater'))

print("Significant test for lap 1 and lap 5:")
print(levene(maze1_lap1, maze1_lap5))
print(ttest_ind(maze1_lap1, maze1_lap5, alternative='greater'), end='\n\n')


print("Stage 1 Maze 1 -----------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1'))[0]
data1 = Data['Lap-wise Distance'][idx]
print_estimator(data1)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 2'))[0]
data2 = Data['Lap-wise Distance'][idx]
print_estimator(data2)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 3'))[0]
data3 = Data['Lap-wise Distance'][idx]
print_estimator(data3)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 4'))[0]
data4 = Data['Lap-wise Distance'][idx]
print_estimator(data4)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 5'))[0]
data5 = Data['Lap-wise Distance'][idx]
print_estimator(data5)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 6'))[0]
data6 = Data['Lap-wise Distance'][idx]
print_estimator(data6)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 7'))[0]
data7 = Data['Lap-wise Distance'][idx]
print_estimator(data7)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 8'))[0]
data8 = Data['Lap-wise Distance'][idx]
print_estimator(data8)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 9'))[0]
data9 = Data['Lap-wise Distance'][idx]
print_estimator(data9)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10'))[0]
data10 = Data['Lap-wise Distance'][idx]
print_estimator(data10)
print("    ", ttest_ind(data1, data10, equal_var=False))
print("Day 2 vs Day 10:  ", levene(data2, data10))
print("Day 3 vs Day 10:  ", levene(data3, data10))
print("Day 4 vs Day 10:  ", levene(data4, data10))
print("Day 5 vs Day 10:  ", levene(data5, data10))
print("Day 6 vs Day 10:  ", levene(data6, data10))
print("Day 7 vs Day 10:  ", levene(data7, data10))
print("Day 8 vs Day 10:  ", levene(data8, data10))
print("Day 9 vs Day 10:  ", levene(data9, data10), end='\n\n')

print("Stage 2 Maze 2: ------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
data221 = Data['Lap-wise Distance'][idx]
print_estimator(data221)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 2'))[0]
data222 = Data['Lap-wise Distance'][idx]
print_estimator(data222)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 2'))[0]
data223 = Data['Lap-wise Distance'][idx]
print_estimator(data223)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 2'))[0]
data224 = Data['Lap-wise Distance'][idx]
print_estimator(data224)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 2'))[0]
data225 = Data['Lap-wise Distance'][idx]
print_estimator(data225)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 2'))[0]
data226 = Data['Lap-wise Distance'][idx]
print_estimator(data226)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 2'))[0]
data227 = Data['Lap-wise Distance'][idx]
print_estimator(data227)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 2'))[0]
data228 = Data['Lap-wise Distance'][idx]
print_estimator(data228)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 2'))[0]
data229 = Data['Lap-wise Distance'][idx]
print_estimator(data229)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 2'))[0]
data2210 = Data['Lap-wise Distance'][idx]
print_estimator(data2210)
print("Maze 2, Day 1 vs Day 10")
print(levene(data221, data2210))
print(ttest_ind(data221, data2210, equal_var=False), end='\n\n')

print("Stage 2 Maze 1: ------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data211 = Data['Lap-wise Distance'][idx]
print_estimator(data211)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 1'))[0]
data212 = Data['Lap-wise Distance'][idx]
print_estimator(data212)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 1'))[0]
data213 = Data['Lap-wise Distance'][idx]
print_estimator(data213)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1'))[0]
data214 = Data['Lap-wise Distance'][idx]
print_estimator(data214)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 1'))[0]
data215 = Data['Lap-wise Distance'][idx]
print_estimator(data215)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 1'))[0]
data216 = Data['Lap-wise Distance'][idx]
print_estimator(data216)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 1'))[0]
data217 = Data['Lap-wise Distance'][idx]
print_estimator(data217)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 1'))[0]
data218 = Data['Lap-wise Distance'][idx]
print_estimator(data218)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1'))[0]
data219 = Data['Lap-wise Distance'][idx]
print_estimator(data219)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1'))[0]
data2110 = Data['Lap-wise Distance'][idx]
print_estimator(data2110, end='\n\n')




# Best records
idx = np.where((Data['Maze Type'] == 'Maze 1')&(Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10'))[0]
SubData = SubDict(Data, Data.keys(), idx)
idx = np.argmin(SubData['Lap-wise Distance'])
print(f"Session: {SubData['Training Day'][idx]} date {SubData['date'][idx]}")
print(f"Mouse: {SubData['MiceID'][idx]}, value {SubData['Lap-wise Distance'][idx]} m", end='\n')
median = np.median(SubData['Lap-wise Distance'])
print(median)


idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data_111 = Data['Lap-wise Distance'][idx]
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1'))[0]
data_141 = Data['Lap-wise Distance'][idx]
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1'))[0]
data_2101 = Data['Lap-wise Distance'][idx]

fig = plt.figure(figsize=(2,3))
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('S1', data_111.shape[0]), np.repeat('S4', data_141.shape[0]), np.repeat('S23-26', data_2101.shape[0])]),
    'Trajectory Length': np.concatenate([data_111, data_141, data_2101]),
}
sns.barplot(
    x='Condition',
    y='Trajectory Length',
    data=compdata,
    palette=colors,
    ax=ax,
    errwidth=0.8,
    capsize=0.3,
    errcolor='black',
    width = 0.8,
    estimator=np.median
)
sns.stripplot(
    x='Condition',
    y='Trajectory Length',
    data=compdata,
    palette=markercolor,
    size=1.5,
    ax = ax,
    jitter=0.2
)
ax.semilogy()
ax.set_ylim([5, 300])
ax.set_ylabel('Lap-wise trajectory length / m')
plt.tight_layout()
plt.savefig(join(loc, "R-S phase [Stage 1].png"), dpi=600)
plt.savefig(join(loc, "R-S phase [Stage 1].svg"), dpi=600)
plt.close()

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
data_111 = Data['Lap-wise Distance'][idx]
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 2'))[0]
data_141 = Data['Lap-wise Distance'][idx]
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 2'))[0]
data_2101 = Data['Lap-wise Distance'][idx]

fig = plt.figure(figsize=(2,3))
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('S1', data_111.shape[0]), np.repeat('S4', data_141.shape[0]), np.repeat('S10-13', data_2101.shape[0])]),
    'Trajectory Length': np.concatenate([data_111, data_141, data_2101]),
}
sns.barplot(
    x='Condition',
    y='Trajectory Length',
    data=compdata,
    palette=colors,
    ax=ax,
    errwidth=0.8,
    capsize=0.3,
    errcolor='black',
    width = 0.8,
    estimator=np.median
)
sns.stripplot(
    x='Condition',
    y='Trajectory Length',
    data=compdata,
    palette=markercolor,
    size=1.5,
    ax = ax,
    jitter=0.2
)
ax.semilogy()
ax.set_ylim([5, 300])
ax.set_ylabel('Lap-wise trajectory length / m')
plt.tight_layout()
plt.savefig(join(loc, "R-S phase [Stage 2].png"), dpi=600)
plt.savefig(join(loc, "R-S phase [Stage 2].svg"), dpi=600)
plt.close()