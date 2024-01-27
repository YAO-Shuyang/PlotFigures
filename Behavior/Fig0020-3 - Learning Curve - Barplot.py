from mylib.statistic_test import *
from matplotlib.gridspec import GridSpec

code_id = '0020 - Learning Curve'
loc = os.path.join(figpath, code_id)
mkdir(loc)

maze_indices = np.where(f_pure_behav['maze_type'] != 0)[0]
if os.path.exists(join(figdata, '0020 - Learning Curve'+'.pkl')):
    with open(join(figdata, '0020 - Learning Curve'+'.pkl'), 'rb') as handle:
        TimeData = pickle.load(handle)
else:
    TimeData = DataFrameEstablish(variable_names = ['Lap ID', 'Lap-wise time cost'], is_behav=True,
                              file_idx=maze_indices,
                              f = f_pure_behav, function = LearningCurve_Interface, 
                              file_name = '0020 - Learning Curve', behavior_paradigm = 'CrossMaze')

maze_indices = np.where(f_pure_behav['maze_type'] != 0)[0]
if os.path.exists(join(figdata, '0020 - Learning Curve-2'+'.pkl')):
    with open(join(figdata, '0020 - Learning Curve-2'+'.pkl'), 'rb') as handle:
        CRData = pickle.load(handle)
else:
    CRData = DataFrameEstablish(variable_names = ['Correct Rate', 'Pass Number', 'Error Number', 'Pure Guess Correct Rate'], is_behav=True,
                              file_idx=maze_indices,
                              f = f_pure_behav, function = LearningCurveBehavioralScore_Interface, 
                              file_name = '0020 - Learning Curve'+'-2', behavior_paradigm = 'CrossMaze')

## Laps
idx = np.concatenate([np.where(((TimeData['Training Day'] == 'Day 1'))&((TimeData['Lap ID'] == 1)|(TimeData['Lap ID'] >= 5))&(TimeData['Maze Type'] == 'Maze 1'))[0],
                       np.where(((TimeData['Training Day'] == '>=Day 10'))&((TimeData['Lap ID'] == 1)|(TimeData['Lap ID'] >= 5))&(TimeData['Maze Type'] == 'Maze 1'))[0]])
LapsData = SubDict(TimeData, TimeData.keys(), idx=idx)
LapsData['Lap ID'][LapsData['Lap ID'] >= 5] = 5
hue = np.array([LapsData['Stage'][i]+' '+LapsData['Training Day'][i] for i in range(len(LapsData['Stage']))])
LapsData['x'] = hue
idx = np.concatenate([np.where((LapsData['x'] == i))[0] for i in ['Stage 1 Day 1', 'Stage 1 >=Day 10', 'Stage 2 Day 1', 'Stage 2 >=Day 10']])
LapsData = SubDict(LapsData, LapsData.keys(), idx=idx)
fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(2.5,3), gridspec_kw={'height_ratios': [1, 3]})
ax1, ax = Clear_Axes(axes[0], close_spines=['top', 'right', 'bottom'], ifyticks=True), Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette("rocket", 3)
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
markercolors = sns.color_palette("Blues", 3)

print("Maze 1 Time:")
data111 = LapsData['Lap-wise time cost'][np.where((LapsData['Lap ID'] == 1)&(LapsData['x'] == 'Stage 1 Day 1'))]
data115 = LapsData['Lap-wise time cost'][np.where((LapsData['Lap ID'] == 5)&(LapsData['x'] == 'Stage 1 Day 1'))]
print("  Stage 1 Day 1:")
print("    ", levene(data111, data115))
print("    ", ttest_ind(data111, data115))
print("  Stage 1 >=Day 10:")
data1101 = LapsData['Lap-wise time cost'][np.where((LapsData['Lap ID'] == 1)&(LapsData['x'] == 'Stage 1 >=Day 10'))]
data1105 = LapsData['Lap-wise time cost'][np.where((LapsData['Lap ID'] == 5)&(LapsData['x'] == 'Stage 1 >=Day 10'))]
print("    ", levene(data1101, data1105))
print("    ", ttest_ind(data1101, data1105, equal_var=False))
print("  Stage 2 Day 1:")
data211 = LapsData['Lap-wise time cost'][np.where((LapsData['Lap ID'] == 1)&(LapsData['x'] == 'Stage 2 Day 1'))]
data215 = LapsData['Lap-wise time cost'][np.where((LapsData['Lap ID'] == 5)&(LapsData['x'] == 'Stage 2 Day 1'))]
print("    ", levene(data211, data215))
print("    ", ttest_ind(data211, data215, equal_var=False))
print("  Stage 2 >=Day 10:")
data2101 = LapsData['Lap-wise time cost'][np.where((LapsData['Lap ID'] == 1)&(LapsData['x'] == 'Stage 2 >=Day 10'))]
data2105 = LapsData['Lap-wise time cost'][np.where((LapsData['Lap ID'] == 5)&(LapsData['x'] == 'Stage 2 >=Day 10'))]
print("    ", levene(data2101, data2105))
print("    ", ttest_ind(data2101, data2105, equal_var=False), end="\n\n")

idx = np.concatenate([np.where((LapsData['Lap-wise time cost'] <= 400)&(LapsData['x'] == i))[0] for i in ['Stage 1 Day 1', 'Stage 1 >=Day 10', 'Stage 2 Day 1', 'Stage 2 >=Day 10']])
SubData = SubDict(LapsData, LapsData.keys(), idx=idx)
sns.barplot(
    x='x',
    y='Lap-wise time cost',
    data=LapsData,
    hue = 'Lap ID',
    palette=colors,
    ax=ax,
    errwidth=0.5,
    capsize=0.2,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x='x',
    y='Lap-wise time cost',
    data=SubData,
    hue = 'Lap ID',
    palette=markercolors,
    edgecolor='black',
    size=2,
    linewidth=0.15,
    ax = ax,
    jitter=0.2,
    dodge=True, legend=False
)
sns.barplot(
    x='x',
    y='Lap-wise time cost',
    data=LapsData,
    hue = 'Lap ID',
    palette=colors,
    ax=ax1,
    errwidth=0.5,
    capsize=0.2,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x='x',
    y='Lap-wise time cost',
    data=LapsData,
    hue = 'Lap ID',
    palette=markercolors,
    edgecolor='black',
    size=2,
    linewidth=0.15,
    ax = ax1,
    jitter=0.2,
    dodge=True, legend=False
)
ax.set_ylim(0, 400)
ax1.set_ylim(400, 3000)
ax1.set_yticks([400, 1000, 2000, 3000])
ax.set_yticks(np.linspace(0, 400, 5))
#ax.set_ylim([0,50])
#ax.set_yticks(np.linspace(0, 50, 6))
ax.set_ylabel('Lap-wise Time of Novel Maze (cm/s)')
plt.tight_layout()
plt.savefig(join(loc, "Maze 1 Lapwise Time.png"), dpi=600)
plt.savefig(join(loc, "Maze 1 Lapwise Time.svg"), dpi=600)
plt.close()

idx = np.concatenate([np.where(((TimeData['Training Day'] == 'Day 1'))&((TimeData['Lap ID'] == 1)|(TimeData['Lap ID'] >= 5))&(TimeData['Maze Type'] == 'Maze 2'))[0],
                       np.where(((TimeData['Training Day'] == '>=Day 10'))&((TimeData['Lap ID'] == 1)|(TimeData['Lap ID'] >= 5))&(TimeData['Maze Type'] == 'Maze 2'))[0]])
LapsData = SubDict(TimeData, TimeData.keys(), idx=idx)
LapsData['Lap ID'][LapsData['Lap ID'] >= 5] = 5

print("Maze 2 Time:")
print("  Day 1:")
data211 = LapsData['Lap-wise time cost'][np.where((LapsData['Lap ID'] == 1)&(LapsData['Training Day'] == 'Day 1'))]
data215 = LapsData['Lap-wise time cost'][np.where((LapsData['Lap ID'] == 5)&(LapsData['Training Day'] == 'Day 1'))]
print("    ", levene(data211, data215))
print("    ", ttest_ind(data211, data215, equal_var=False))
print("  >=Day 10:")
data2101 = LapsData['Lap-wise time cost'][np.where((LapsData['Lap ID'] == 1)&(LapsData['Training Day'] == '>=Day 10'))]
data2105 = LapsData['Lap-wise time cost'][np.where((LapsData['Lap ID'] == 5)&(LapsData['Training Day'] == '>=Day 10'))]
print("    ", levene(data2101, data2105))
print("    ", ttest_ind(data2101, data2105), end="\n\n")
fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(2,3), gridspec_kw={'height_ratios': [1, 3]})
ax1, ax = Clear_Axes(axes[0], close_spines=['top', 'right', 'bottom'], ifyticks=True), Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette("rocket", 3)
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
markercolors = sns.color_palette("Blues", 3)
idx = np.where(LapsData['Lap-wise time cost'] <= 200)[0]
SubData = SubDict(LapsData, LapsData.keys(), idx=idx)
sns.barplot(
    x='Training Day',
    y='Lap-wise time cost',
    data=LapsData,
    hue='Lap ID',
    palette=colors,
    ax=ax,
    errwidth=0.5,
    capsize=0.2,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x='Training Day',
    y='Lap-wise time cost',
    data=SubData,
    hue='Lap ID',
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    jitter=0.2,
    dodge=True, legend=False
)
sns.barplot(
    x='Training Day',
    y='Lap-wise time cost',
    data=LapsData,
    hue='Lap ID',
    palette=colors,
    ax=ax1,
    errwidth=0.5,
    capsize=0.2,
    errcolor='black',
)
sns.stripplot(
    x='Training Day',
    y='Lap-wise time cost',
    data=LapsData,
    hue='Lap ID',
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax1,
    jitter=0.2,
    dodge=True, legend=False
)
ax.set_ylim(0, 200)
ax1.set_ylim(200, 1200)
ax1.set_yticks([200, 600, 900, 1200])
ax.set_yticks(np.linspace(0, 200, 5))
#ax.set_ylim([0,50])
#ax.set_yticks(np.linspace(0, 50, 6))
ax.set_ylabel('Lap-wise Time of Novel Maze (cm/s)')
plt.tight_layout()
plt.savefig(join(loc, "Maze 2 Lapwise Time.png"), dpi=600)
plt.savefig(join(loc, "Maze 2 Lapwise Time.svg"), dpi=600)
plt.close()



idx = np.concatenate([np.where((CRData['Training Day'] == 'Day 1'))[0]