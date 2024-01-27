# Fig0020-1, Plotting learning curve in behavioral pattern, and explore time of each lap are plotted
# Fig0020-2, Learning curve ploting the mean of laps on each training day.

from mylib.statistic_test import *
from matplotlib.gridspec import GridSpec

code_id = '0020 - Learning Curve'
loc = os.path.join(figpath, code_id)
mkdir(loc)

maze_indices = np.where(f_pure_behav['maze_type'] != 0)[0]
if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Lap ID', 'Lap-wise time cost'], is_behav=True,
                              file_idx=maze_indices,
                              f = f_pure_behav, function = LearningCurve_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')

uniq_day = ['Day 1', 'Day 2', 'Day 3', 'Day 4',
            'Day 5', 'Day 6', 'Day 7', 'Day 8',
            'Day 9', '>=Day 10']

def TransformData(Data: dict):
    x = np.zeros(Data['Training Day'].shape[0], np.int64)
    Lap1s = np.where(Data['Lap ID'] == 1)[0]
    Lap2s = np.where(Data['Lap ID'] == 2)[0]
    Lap3s = np.where(Data['Lap ID'] == 3)[0]
    Lap4s = np.where(Data['Lap ID'] == 4)[0]
    Lap5s = np.where(Data['Lap ID'] >= 5)[0]

    x[Lap1s] = 1
    x[Lap2s] = 2
    x[Lap3s] = 3
    x[Lap4s] = 4
    x[Lap5s] = 5
    

    for i, d in enumerate(uniq_day):
        idx = np.where(Data['Training Day'] == d)[0]
        x[idx] += i*7
   
    Data['x'] = x
    return Data

Data=TransformData(Data)

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,2))
ax1, ax2 = axes[0], axes[1],
ax1 = Clear_Axes(ax1, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(ax2, close_spines=['top', 'right'], ifxticks=True)
x = np.unique(Data['Training Day'])

colors = sns.color_palette("rocket", 3)
colors = [colors[1], colors[-1]]
markercolors = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[0]]

idx = np.where(Data['Stage'] == 'Stage 1')[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
sns.lineplot(
    x='Training Day',
    y='Lap-wise time cost',
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
"""
idx = np.where(Data['Stage'] == 'Stage 1')[0]
sns.stripplot(
    x=Data['x'][idx],
    y=Data['Lap-wise time cost'][idx],
    hue=Data['Maze Type'][idx],
    palette=colors,
    size=0.5,
    ax = ax1,
    dodge=True,
    jitter=0.2,
    native_scale=True
)
"""
ax1.set_ylim([0,1200])
ax1.set_yticks(np.linspace(0, 1200, 5))

idx = np.where(Data['Stage'] == 'Stage 2')[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
sns.lineplot(
    x='Training Day',
    y='Lap-wise time cost',
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
ax2.set_ylim([15,35])
ax2.set_yticks(np.linspace(15, 35, 5))
"""
idx = np.where(Data['Stage'] == 'Stage 2')[0]
sns.stripplot(
    x=Data['x'][idx],
    y=Data['Lap-wise time cost'][idx],
    hue=Data['Maze Type'][idx],
    palette=colors,
    size=0.5,
    ax = ax2,
    dodge=True,
    jitter=0,
    native_scale=True
)
"""

plt.savefig(join(loc, 'learning_curve.png'), dpi=600)
plt.savefig(join(loc, 'learning_curve.svg'), dpi=600)
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
    y=Data['Lap-wise time cost'][idx],
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
    y=Data['Lap-wise time cost'][idx],
    hue=Data['Maze Type'][idx],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.05,
    ax = ax1,
    dodge=True,
    jitter=0.1
)
ax1.set_ylim([0, 3000])

idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10'))
sns.lineplot(
    x=Data['x'][idx]-64.4,
    y=Data['Lap-wise time cost'][idx],
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
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Lap-wise time cost']<=80))
sns.stripplot(
    x=Data['x'][idx],
    y=Data['Lap-wise time cost'][idx],
    hue=Data['Maze Type'][idx],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.05,
    ax = ax3,
    dodge=True,
    jitter=0.1
)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Lap-wise time cost']>80))
sns.stripplot(
    x=Data['x'][idx],
    y=Data['Lap-wise time cost'][idx],
    hue=Data['Maze Type'][idx],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.05,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax3.set_ylim([0, 90])
ax3.set_yticks(np.linspace(0, 80, 5))
ax3.set_xticks([-0.2, 0.8, 1.8, 2.8, 3.8], labels=[1, 2, 3, 4, 5])
ax3.set_xlim([-1, 4.5])
ax2.set_ylim([60,200])
ax2.set_yticks([80, 200])
ax2.set_xlim([-1, 4.5])
plt.tight_layout()
plt.savefig(join(loc, 'learning_curve [Zoom out Stage 1].png'), dpi=600)
plt.savefig(join(loc, 'learning_curve [Zoom out Stage 1].svg'), dpi=600)
plt.close()





grid = GridSpec(2,2, height_ratios=[1,3])
fig = plt.figure(figsize=(5,2))
ax1 = plt.subplot(grid[:, 0])
ax2 = plt.subplot(grid[0, 1])
ax3 = plt.subplot(grid[1, 1])
ax1 = Clear_Axes(ax1, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(ax2, close_spines=['top', 'right', 'bottom'], ifyticks=True)
ax3 = Clear_Axes(ax3, close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1'))
sns.lineplot(
    x=Data['x'][idx]-1,
    y=Data['Lap-wise time cost'][idx],
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
    y=Data['Lap-wise time cost'][idx],
    hue=Data['Maze Type'][idx],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.05,
    ax = ax1,
    dodge=True,
    jitter=0.1
)
ax1.set_ylim([-100, 1500])
ax1.set_yticks([0, 500, 1000, 1500])

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10'))
sns.lineplot(
    x=Data['x'][idx]-64,
    y=Data['Lap-wise time cost'][idx],
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
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap-wise time cost']<=60))
sns.stripplot(
    x=Data['x'][idx],
    y=Data['Lap-wise time cost'][idx],
    hue=Data['Maze Type'][idx],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.05,
    ax = ax3,
    dodge=True,
    jitter=0.1
)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap-wise time cost']>60))
sns.stripplot(
    x=Data['x'][idx],
    y=Data['Lap-wise time cost'][idx],
    hue=Data['Maze Type'][idx],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.05,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax3.set_ylim([0, 65])
ax3.set_yticks(np.linspace(0, 60, 5))
ax3.set_xlim([-1, 5])
ax2.set_ylim([50,200])
ax2.set_yticks([60, 200])
ax2.set_xlim([-1, 5])
plt.tight_layout()
plt.savefig(join(loc, 'learning_curve [Zoom out Stage 2].png'), dpi=2400)
plt.savefig(join(loc, 'learning_curve [Zoom out Stage 2].svg'), dpi=2400)
plt.close()

# First Session & Second Session
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data_11 = Data['Lap-wise time cost'][idx]
print_estimator(data_11)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
data_21 = Data['Lap-wise time cost'][idx]
print_estimator(data_21)

fig = plt.figure(figsize=(1.5,3))
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('Novel Maze 1', data_11.shape[0]), np.repeat('Novel Maze 2', data_21.shape[0])]),
    'Lap-wise time cost': np.concatenate([data_11, data_21]),
}
sns.barplot(
    x='Condition',
    y='Lap-wise time cost',
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
    y='Lap-wise time cost',
    data=compdata,
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    jitter=0.2
)
#ax.set_ylim([0,50])
#ax.set_yticks(np.linspace(0, 50, 6))
ax.set_xticks([0, 1], ["Novel\nMaze 1", "Novel\nMaze 2"])
ax.set_ylabel('Lap-wise Time of Novel Maze (cm/s)')
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
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap1 = Data['Lap-wise time cost'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 2))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap2 = Data['Lap-wise time cost'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] >= 5))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap5 = Data['Lap-wise time cost'][idx]

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
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap1 = Data['Lap-wise time cost'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] == 2))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap2 = Data['Lap-wise time cost'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] >= 5))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap5 = Data['Lap-wise time cost'][idx]

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
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap1 = Data['Lap-wise time cost'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 2)&(Data['Maze Type'] == 'Maze 2'))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap2 = Data['Lap-wise time cost'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] >= 5)&(Data['Maze Type'] == 'Maze 2'))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap5 = Data['Lap-wise time cost'][idx]

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
print_estimator(Data['Lap-wise time cost'][idx], end='\n\n')
maze1_lap1 = Data['Lap-wise time cost'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] == 2)&(Data['Maze Type'] == 'Maze 2'))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap2 = Data['Lap-wise time cost'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] >= 5)&(Data['Maze Type'] == 'Maze 2'))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap5 = Data['Lap-wise time cost'][idx]

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
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap1 = Data['Lap-wise time cost'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 2)&(Data['Maze Type'] == 'Maze 1'))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap2 = Data['Lap-wise time cost'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] >= 5)&(Data['Maze Type'] == 'Maze 1'))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap5 = Data['Lap-wise time cost'][idx]

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
print_estimator(Data['Lap-wise time cost'][idx], end='\n\n')
maze1_lap1 = Data['Lap-wise time cost'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] == 2)&(Data['Maze Type'] == 'Maze 1'))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap2 = Data['Lap-wise time cost'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] >= 5)&(Data['Maze Type'] == 'Maze 1'))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise time cost'][idx])
maze1_lap5 = Data['Lap-wise time cost'][idx]

print("Significant test for lap 1 and lap 2:")
print(levene(maze1_lap1, maze1_lap2))
print(ttest_ind(maze1_lap1, maze1_lap2, alternative='greater'))

print("Significant test for lap 1 and lap 5:")
print(levene(maze1_lap1, maze1_lap5))
print(ttest_ind(maze1_lap1, maze1_lap5, alternative='greater'), end='\n\n')

print("Stage 1 Maze 1 ---------------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data1 = Data['Lap-wise time cost'][idx]
print(f"Day 1:   Mean: {np.mean(data1)}, STD: {np.std(data1)}, min: {np.min(data1)}, max: {np.max(data1)}, median: {np.median(data1)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 1'))[0]
data2 = Data['Lap-wise time cost'][idx]
print(f"Day 2:   Mean: {np.mean(data2)}, STD: {np.std(data2)}, min: {np.min(data2)}, max: {np.max(data2)}, median: {np.median(data2)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 1'))[0]
data3 = Data['Lap-wise time cost'][idx]
print(f"Day 3:   Mean: {np.mean(data3)}, STD: {np.std(data3)}, min: {np.min(data3)}, max: {np.max(data3)}, median: {np.median(data3)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1'))[0]
data4 = Data['Lap-wise time cost'][idx]
print(f"Day 4:   Mean: {np.mean(data4)}, STD: {np.std(data4)}, min: {np.min(data4)}, max: {np.max(data4)}, median: {np.median(data4)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 1'))[0]
data5 = Data['Lap-wise time cost'][idx]
print(f"Day 5:   Mean: {np.mean(data5)}, STD: {np.std(data5)}, min: {np.min(data5)}, max: {np.max(data5)}, median: {np.median(data5)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 1'))[0]
data6 = Data['Lap-wise time cost'][idx]
print(f"Day 6:   Mean: {np.mean(data6)}, STD: {np.std(data6)}, min: {np.min(data6)}, max: {np.max(data6)}, median: {np.median(data6)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 1'))[0]
data7 = Data['Lap-wise time cost'][idx]
print(f"Day 7:   Mean: {np.mean(data7)}, STD: {np.std(data7)}, min: {np.min(data7)}, max: {np.max(data7)}, median: {np.median(data7)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 1'))[0]
data8 = Data['Lap-wise time cost'][idx]
print(f"Day 8:   Mean: {np.mean(data8)}, STD: {np.std(data8)}, min: {np.min(data8)}, max: {np.max(data8)}, median: {np.median(data8)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1'))[0]
data9 = Data['Lap-wise time cost'][idx]
print(f"Day 9:   Mean: {np.mean(data9)}, STD: {np.std(data9)}, min: {np.min(data9)}, max: {np.max(data9)}, median: {np.median(data9)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1'))[0]
data10 = Data['Lap-wise time cost'][idx]
print(f"Day 10:  Mean: {np.mean(data10)}, STD: {np.std(data10)}, min: {np.min(data10)}, max: {np.max(data10)}, median: {np.median(data10)}", end='\n\n')
print("Day 1 vs Day 10:  ", ttest_ind(data1, data10, alternative='greater'))
print("Day 2 vs Day 10:  ", ttest_ind(data2, data10, alternative='greater'))
print("Day 3 vs Day 10:  ", ttest_ind(data3, data10, alternative='greater'))
print("Day 4 vs Day 10:  ", ttest_ind(data4, data10, alternative='greater'))
print("Day 5 vs Day 10:  ", ttest_ind(data5, data10, alternative='greater'))
print("Day 6 vs Day 10:  ", ttest_ind(data6, data10, alternative='greater'))
print("Day 7 vs Day 10:  ", ttest_ind(data7, data10, alternative='greater'))
print("Day 8 vs Day 10:  ", ttest_ind(data8, data10, alternative='greater'))
print("Day 9 vs Day 10:  ", ttest_ind(data9, data10, alternative='greater'), end='\n\n\n')
print("Day 1 vs Day 3:  ", ttest_ind(data1, data3, alternative='greater'))

print("Stage 2 Maze 2 ---------------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
data1 = Data['Lap-wise time cost'][idx]
print(f"Day 1:   Mean: {np.mean(data1)}, STD: {np.std(data1)}, min: {np.min(data1)}, max: {np.max(data1)}, median: {np.median(data1)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 2'))[0]
data2 = Data['Lap-wise time cost'][idx]
print(f"Day 2:   Mean: {np.mean(data2)}, STD: {np.std(data2)}, min: {np.min(data2)}, max: {np.max(data2)}, median: {np.median(data2)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 2'))[0]
data3 = Data['Lap-wise time cost'][idx]
print(f"Day 3:   Mean: {np.mean(data3)}, STD: {np.std(data3)}, min: {np.min(data3)}, max: {np.max(data3)}, median: {np.median(data3)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 2'))[0]
data4 = Data['Lap-wise time cost'][idx]
print(f"Day 4:   Mean: {np.mean(data4)}, STD: {np.std(data4)}, min: {np.min(data4)}, max: {np.max(data4)}, median: {np.median(data4)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 2'))[0]
data5 = Data['Lap-wise time cost'][idx]
print(f"Day 5:   Mean: {np.mean(data5)}, STD: {np.std(data5)}, min: {np.min(data5)}, max: {np.max(data5)}, median: {np.median(data5)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 2'))[0]
data6 = Data['Lap-wise time cost'][idx]
print(f"Day 6:   Mean: {np.mean(data6)}, STD: {np.std(data6)}, min: {np.min(data6)}, max: {np.max(data6)}, median: {np.median(data6)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 2'))[0]
data7 = Data['Lap-wise time cost'][idx]
print(f"Day 7:   Mean: {np.mean(data7)}, STD: {np.std(data7)}, min: {np.min(data7)}, max: {np.max(data7)}, median: {np.median(data7)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 2'))[0]
data8 = Data['Lap-wise time cost'][idx]
print(f"Day 8:   Mean: {np.mean(data8)}, STD: {np.std(data8)}, min: {np.min(data8)}, max: {np.max(data8)}, median: {np.median(data8)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 2'))[0]
data9 = Data['Lap-wise time cost'][idx]
print(f"Day 9:   Mean: {np.mean(data9)}, STD: {np.std(data9)}, min: {np.min(data9)}, max: {np.max(data9)}, median: {np.median(data9)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 2'))[0]
data10 = Data['Lap-wise time cost'][idx]
print(f"Day 10:  Mean: {np.mean(data10)}, STD: {np.std(data10)}, min: {np.min(data10)}, max: {np.max(data10)}, median: {np.median(data10)}", end='\n\n')
print("Day 1 vs Day 10:  ", ttest_ind(data1, data10, alternative='greater'))
print("Day 2 vs Day 10:  ", ttest_ind(data2, data10, alternative='greater'))
print("Day 3 vs Day 10:  ", ttest_ind(data3, data10, alternative='greater'))
print("Day 4 vs Day 10:  ", ttest_ind(data4, data10, alternative='greater'))
print("Day 5 vs Day 10:  ", ttest_ind(data5, data10, alternative='greater'))
print("Day 6 vs Day 10:  ", ttest_ind(data6, data10, alternative='greater'))
print("Day 7 vs Day 10:  ", ttest_ind(data7, data10, alternative='greater'))
print("Day 8 vs Day 10:  ", ttest_ind(data8, data10, alternative='greater'))
print("Day 9 vs Day 10:  ", ttest_ind(data9, data10, alternative='greater'), end='\n\n\n')
print("Day 1 vs Day 3:  ", ttest_ind(data1, data3, alternative='greater'))


print("Stage 2 Maze 1 ---------------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data11 = Data['Lap-wise time cost'][idx]
print(f"Day 1:   Mean: {np.mean(data11)}, STD: {np.std(data11)}, min: {np.min(data11)}, max: {np.max(data11)}, median: {np.median(data11)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 1'))[0]
data12 = Data['Lap-wise time cost'][idx]
print(f"Day 2:   Mean: {np.mean(data12)}, STD: {np.std(data12)}, min: {np.min(data12)}, max: {np.max(data12)}, median: {np.median(data12)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 1'))[0]
data13 = Data['Lap-wise time cost'][idx]
print(f"Day 3:   Mean: {np.mean(data13)}, STD: {np.std(data13)}, min: {np.min(data13)}, max: {np.max(data13)}, median: {np.median(data13)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1'))[0]
data14 = Data['Lap-wise time cost'][idx]
print(f"Day 4:   Mean: {np.mean(data14)}, STD: {np.std(data14)}, min: {np.min(data14)}, max: {np.max(data14)}, median: {np.median(data14)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 1'))[0]
data15 = Data['Lap-wise time cost'][idx]
print(f"Day 5:   Mean: {np.mean(data15)}, STD: {np.std(data15)}, min: {np.min(data15)}, max: {np.max(data15)}, median: {np.median(data15)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 1'))[0]
data16 = Data['Lap-wise time cost'][idx]
print(f"Day 6:   Mean: {np.mean(data16)}, STD: {np.std(data16)}, min: {np.min(data16)}, max: {np.max(data16)}, median: {np.median(data16)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 1'))[0]
data17 = Data['Lap-wise time cost'][idx]
print(f"Day 7:   Mean: {np.mean(data17)}, STD: {np.std(data17)}, min: {np.min(data17)}, max: {np.max(data17)}, median: {np.median(data17)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 1'))[0]
data18 = Data['Lap-wise time cost'][idx]
print(f"Day 8:   Mean: {np.mean(data18)}, STD: {np.std(data18)}, min: {np.min(data18)}, max: {np.max(data18)}, median: {np.median(data18)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1'))[0]
data19 = Data['Lap-wise time cost'][idx]
print(f"Day 9:   Mean: {np.mean(data19)}, STD: {np.std(data19)}, min: {np.min(data19)}, max: {np.max(data19)}, median: {np.median(data19)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1'))[0]
data110 = Data['Lap-wise time cost'][idx]
print(f"Day 10:  Mean: {np.mean(data110)}, STD: {np.std(data110)}, min: {np.min(data110)}, max: {np.max(data110)}, median: {np.median(data110)}", end='\n\n')
print("Day 1 vs Day 10:  ", ttest_ind(data1, data110, alternative='greater'))
print("Day 2 vs Day 10:  ", ttest_ind(data12, data110, alternative='greater'))
print("Day 3 vs Day 10:  ", ttest_ind(data13, data110, alternative='greater'))
print("Day 4 vs Day 10:  ", ttest_ind(data14, data110, alternative='greater'))
print("Day 5 vs Day 10:  ", ttest_ind(data15, data110, alternative='greater'))
print("Day 6 vs Day 10:  ", ttest_ind(data16, data110, alternative='greater'))
print("Day 7 vs Day 10:  ", ttest_ind(data17, data110, alternative='greater'))
print("Day 8 vs Day 10:  ", ttest_ind(data18, data110, alternative='greater'))
print("Day 9 vs Day 10:  ", ttest_ind(data19, data110, alternative='greater'))
print("Day 1 vs Day 3:  ", ttest_ind(data11, data13, alternative='greater'), end='\n\n\n')


print("Stage 1: t-test, maze 2 vs maze 1")
print(ttest_ind(data1, data110, alternative='greater'))
print(ttest_ind(data2, data12))
print(ttest_ind(data3, data13))
print(ttest_ind(data4, data14))
print(ttest_ind(data5, data15))
print(ttest_ind(data6, data16))
print(ttest_ind(data7, data17))
print(ttest_ind(data8, data18))
print(ttest_ind(data9, data19))
print(ttest_ind(data10, data110), end='\n\n\n')

# Total time within maze 1
idx = np.where((Data['Maze Type'] == 'Maze 1')&(Data['Stage'] == 'Stage 1')&(Data['MiceID'] != 11094)&(Data['MiceID'] != 11092))[0]
SubData = SubDict(Data, Data.keys(), idx)
mice = np.unique(SubData['MiceID'])

print("Stage 1, Maze 1 total time:")
t = []
for mouse in mice:
    idx = np.where(SubData['MiceID'] == mouse)[0]
    t.append(np.sum(SubData['Lap-wise time cost'][idx]))
print_estimator(t, end='\n\n')

print("Stage 2, Maze 1 total time:")
t = []
idx = np.where((Data['Maze Type'] == 'Maze 1')&(Data['Stage'] == 'Stage 2')&(Data['MiceID'] != 11094)&(Data['MiceID'] != 11092))[0]
SubData = SubDict(Data, Data.keys(), idx)
mice = np.unique(SubData['MiceID'])
for mouse in mice:
    idx = np.where(SubData['MiceID'] == mouse)[0]
    t.append(np.sum(SubData['Lap-wise time cost'][idx]))
print_estimator(t, end='\n\n')


print("Stage 2, Maze 2 total time:")
t = []
idx = np.where((Data['Maze Type'] == 'Maze 2')&(Data['Stage'] == 'Stage 2')&(Data['MiceID'] != 11094)&(Data['MiceID'] != 11092))[0]
SubData = SubDict(Data, Data.keys(), idx)
mice = np.unique(SubData['MiceID'])
for mouse in mice:
    idx = np.where(SubData['MiceID'] == mouse)[0]
    t.append(np.sum(SubData['Lap-wise time cost'][idx]))
print_estimator(t, end='\n\n')

# Best records
#  Maze 1:
print("Best records --------------------------------------------")
idx = np.where((Data['Maze Type'] == 'Maze 1')&(Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10'))[0]
SubData = SubDict(Data, Data.keys(), idx)
idx = np.argmin(SubData['Lap-wise time cost'])
print(f"Session: {SubData['Training Day'][idx]}")
print(f"Mouse: {SubData['MiceID'][idx]}, value {SubData['Lap-wise time cost'][idx]} s", end='\n')
median = np.median(SubData['Lap-wise time cost'])
print(median, end='\n\n')

print("First lap of first session (light and dark) --------------------------------------------")
idx = np.where((Data['Training Day'] == 'Day 1')&(Data['Stage'] == 'Stage 1')&(Data['x'] == 1))[0]
data_light = Data['Lap-wise time cost'][idx]
print(f"Mean {np.mean(data_light)}, Std {np.std(data_light)}")
print(f"Min {np.min(data_light)}, Max {np.max(data_light)}")
data_dark = np.array(f_pure_old['time'])
print(f"Mean {np.mean(data_dark)}, Std {np.std(data_dark)}")
print(f"Min {np.min(data_dark)}, Max {np.max(data_dark)}")
print(levene(data_light, data_dark))
print(ttest_ind(data_light, data_dark, alternative='less'), end='\n\n')

print("Second lap of first session --------------------------------------------")
idx = np.where((Data['Training Day'] == 'Day 1')&(Data['Stage'] == 'Stage 1')&(Data['x'] == 2))[0]
data_sec = Data['Lap-wise time cost'][idx]
print(f"Mean {np.mean(data_sec)}, Std {np.std(data_sec)}")
print(f"Min {np.min(data_sec)}, Max {np.max(data_sec)}")
print(levene(data_light, data_sec))
print(ttest_ind(data_light, data_sec, alternative='greater'), end='\n\n')

print("First lap of Stage 2 Maze 2, vs --------------------------------------------")
idx = np.where((Data['Training Day'] == 'Day 1')&(Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 2')&(Data['x'] == 1))[0]
data_1p = Data['Lap-wise time cost'][idx]
print(f"Mean {np.mean(data_1p)}, Std {np.std(data_1p)}")
print(f"Min {np.min(data_1p)}, Max {np.max(data_1p)}")
print(levene(data_light, data_1p))
print(ttest_ind(data_light, data_1p, alternative='greater'), end='\n\n')



fig = plt.figure(figsize=(2,3))
colors = sns.color_palette("rocket", 2)
markercolor = sns.color_palette("Blues", 2)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('Light', data_light.shape[0]), np.repeat('Dark', data_dark.shape[0])]),
    'Time cost': np.concatenate([data_light, data_dark]),
}
sns.barplot(
    x='Condition',
    y='Time cost',
    data=compdata,
    palette=colors,
    ax=ax,
    errwidth=0.8,
    capsize=0.3,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x='Condition',
    y='Time cost',
    data=compdata,
    palette=markercolor,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    jitter=0.2
)
ax.set_ylim([0, 4000])
ax.set_ylabel('Time cost of first lap / s')
plt.tight_layout()
plt.savefig(join(loc, "Comparison of Light and Dark.png"), dpi=600)
plt.savefig(join(loc, "Comparison of Light and Dark.svg"), dpi=600)
plt.close()


fig = plt.figure(figsize=(1.5,3))
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('Maze 1 First Lap', data_light.shape[0]), np.repeat('Maze 1 Second Lap', data_sec.shape[0]), np.repeat('Maze 2 First Lap', data_1p.shape[0])]),
    'Time cost': np.concatenate([data_light, data_sec, data_1p]),
}
sns.barplot(
    x='Condition',
    y='Time cost',
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
    y='Time cost',
    data=compdata,
    palette=markercolor,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    jitter=0.2
)
ax.set_xticks([-1, 0, 1], ["Maze 1 First Lap", "Maze 1 Second Lap", "Maze 2 First Lap"], rotation=45)
ax.set_ylabel('Time cost of first lap / s')
plt.tight_layout()
plt.savefig(join(loc, "Comparison of Maze 1 and Maze 2.png"), dpi=600)
plt.savefig(join(loc, "Comparison of Maze 1 and Maze 2.svg"), dpi=600)
plt.close()



idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data_11 = Data['Lap-wise time cost'][idx]
print_estimator(data_11)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
data_21 = Data['Lap-wise time cost'][idx]
print_estimator(data_21)



idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data_111 = Data['Lap-wise time cost'][idx]
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1'))[0]
data_141 = Data['Lap-wise time cost'][idx]
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1'))[0]
data_2101 = Data['Lap-wise time cost'][idx]

fig = plt.figure(figsize=(2,3))
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('S1', data_111.shape[0]), np.repeat('S4', data_141.shape[0]), np.repeat('S23-26', data_2101.shape[0])]),
    'Time cost': np.concatenate([data_111, data_141, data_2101]),
}
sns.barplot(
    x='Condition',
    y='Time cost',
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
    y='Time cost',
    data=compdata,
    palette=markercolor,
    size=1.5,
    ax = ax,
    jitter=0.2
)
ax.semilogy()
ax.set_ylim([1, 3000])
ax.set_ylabel('Lap-wise navigation time/s')
plt.tight_layout()
plt.savefig(join(loc, "R-S phase [Stage 1 time].png"), dpi=600)
plt.savefig(join(loc, "R-S phase [Stage 1 time].svg"), dpi=600)
plt.close()


idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
data_111 = Data['Lap-wise time cost'][idx]
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 2'))[0]
data_141 = Data['Lap-wise time cost'][idx]
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 2'))[0]
data_2101 = Data['Lap-wise time cost'][idx]

fig = plt.figure(figsize=(2,3))
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('S1', data_111.shape[0]), np.repeat('S4', data_141.shape[0]), np.repeat('S10-13', data_2101.shape[0])]),
    'Time cost': np.concatenate([data_111, data_141, data_2101]),
}
sns.barplot(
    x='Condition',
    y='Time cost',
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
    y='Time cost',
    data=compdata,
    palette=markercolor,
    size=1.5,
    ax = ax,
    jitter=0.2
)
ax.semilogy()
ax.set_ylim([10, 1500])
ax.set_ylabel('Lap-wise navigation time/s')
plt.tight_layout()
plt.savefig(join(loc, "R-S phase [Stage 2 time].png"), dpi=600)
plt.savefig(join(loc, "R-S phase [Stage 2 time].svg"), dpi=600)
plt.close()