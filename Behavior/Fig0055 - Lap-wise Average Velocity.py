from mylib.statistic_test import *

code_id = '0055 - Lap-wise Average Velocity'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Lap-wise Average Velocity', 'Lap ID'], is_behav=True,
                              f = f_pure_behav, function = LapwiseAverageVelocity_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')

x_ticks = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']

idx = np.concatenate([np.where(Data['Training Day'] == day)[0] for day in x_ticks])
Data = SubDict(Data, Data.keys(), idx=idx)

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right', 'left'], ifxticks=True)

colors = sns.color_palette("rocket", 4)
colors = [colors[0], colors[-1]]
markercolors = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[0]]

y_max = np.max(Data['Lap-wise Average Velocity'])
print(y_max)

# Stage 1 only
idx = np.where(Data['Stage'] == 'Stage 1')[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
sns.lineplot(
    x = 'Training Day',
    y = 'Lap-wise Average Velocity',
    hue='Maze Type',
    palette=colors,
    data = SubData,
    ax = ax1,
    err_style='bars',
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
sns.stripplot(
    x='Training Day',
    y='Lap-wise Average Velocity',
    hue='Maze Type',
    data=SubData,
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.05,
    ax = ax1,
    dodge=True,
    jitter=0.1
)
ax1.set_ylim([0, 50])
ax1.set_yticks(np.linspace(0, 50, 6))

idx = np.where(Data['Stage'] == 'Stage 2')[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
sns.lineplot(
    x = 'Training Day',
    y = 'Lap-wise Average Velocity',
    hue='Maze Type',
    palette=colors,
    data = SubData,
    ax = ax2,
    err_style='bars',
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
sns.stripplot(
    x='Training Day',
    y='Lap-wise Average Velocity',
    hue='Maze Type',
    data=SubData,
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax2.set_ylim([0, 50])

plt.tight_layout()
plt.savefig(join(loc, 'Lap-wise Average Velocity.png'), dpi = 600)
plt.savefig(join(loc, 'Lap-wise Average Velocity.svg'), dpi = 600)
plt.close()

idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1'))[0]
data1 = Data['Lap-wise Average Velocity'][idx]
print(np.mean(data1), np.std(data1))

idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10'))[0]
data10 = Data['Lap-wise Average Velocity'][idx]
print(np.mean(data10), np.std(data10))

print(levene(data1, data10))
print(ttest_ind(data1, data10, alternative='less', equal_var=False))


idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 1)&(Data['Maze Type'] == 'Maze 1'))[0]
data_11 = Data['Lap-wise Average Velocity'][idx]
print("First exposure")
print_estimator(data_11, end='\n\n')
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 1)&(Data['Maze Type'] == 'Maze 2'))[0]
data_21 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data_21)

fig = plt.figure(figsize=(2,3))
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('Maze 1 First Lap', data_11.shape[0]), np.repeat('Maze 2 First Lap', data_21.shape[0])]),
    'Velocity': np.concatenate([data_11, data_21]),
}
sns.barplot(
    x='Condition',
    y='Velocity',
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
    y='Velocity',
    data=compdata,
    palette=markercolor,
    size=3,
    ax = ax,
    jitter=0.2
)
ax.set_xticks([0, 1], ["Novel\nMaze 1", "Novel\nMaze 2"])
ax.set_ylabel('Velocity of first lap (cm/s)')
ax.set_ylim(0, 20)
plt.tight_layout()
plt.savefig(join(loc, "Comparison of Maze 1 and Maze 2.png"), dpi=600)
plt.savefig(join(loc, "Comparison of Maze 1 and Maze 2.svg"), dpi=600)
plt.close()
print(levene(data_11, data_21))
print("First lap: ",ttest_ind(data_11, data_21, alternative='less'), end='\n\n')


idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data_11 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data_11)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
data_21 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data_21)

fig = plt.figure(figsize=(2,3))
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('Novel Maze 1', data_11.shape[0]), np.repeat('Novel Maze 2', data_21.shape[0])]),
    'Velocity': np.concatenate([data_11, data_21]),
}
sns.barplot(
    x='Condition',
    y='Velocity',
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
    y='Velocity',
    data=compdata,
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    jitter=0.2
)
ax.set_ylim([0,50])
ax.set_yticks(np.linspace(0, 50, 6))
ax.set_xticks([0, 1], ["Novel\nMaze 1", "Novel\nMaze 2"])
ax.set_ylabel('Lap-wise Velocity of Novel Maze (cm/s)')
plt.tight_layout()
plt.savefig(join(loc, "Comparison of Maze 1 and Maze 2 [S1].png"), dpi=600)
plt.savefig(join(loc, "Comparison of Maze 1 and Maze 2 [S1].svg"), dpi=600)
plt.close()
print(levene(data_11, data_21))
print(ttest_ind(data_11, data_21, alternative='less', equal_var=False), end='\n\n')


print("Stage 1 Maze 1 -----------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1'))[0]
data1 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data1)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 2'))[0]
data2 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data2)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 3'))[0]
data3 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data3)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 4'))[0]
data4 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data4)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 5'))[0]
data5 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data5)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 6'))[0]
data6 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data6)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 7'))[0]
data7 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data7)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 8'))[0]
data8 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data8)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 9'))[0]
data9 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data9)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10'))[0]
data10 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data10)
print(ttest_ind(data1, data10, alternative='less'))
print(ttest_ind(data2, data10, alternative='less'))
print(ttest_ind(data3, data10, alternative='less'))
print(ttest_ind(data4, data10, alternative='less'), end='\n\n')

print("Stage 2 Maze 2: ------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
data221 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data221)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 2'))[0]
data222 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data222)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 2'))[0]
data223 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data223)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 2'))[0]
data224 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data224)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 2'))[0]
data225 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data225)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 2'))[0]
data226 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data226)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 2'))[0]
data227 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data227)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 2'))[0]
data228 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data228)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 2'))[0]
data229 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data229)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 2'))[0]
data2210 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data2210, end='\n\n')

print("Stage 2 Maze 1: ------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data211 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data211)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 1'))[0]
data212 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data212)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 1'))[0]
data213 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data213)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1'))[0]
data214 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data214)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 1'))[0]
data215 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data215)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 1'))[0]
data216 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data216)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 1'))[0]
data217 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data217)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 1'))[0]
data218 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data218)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1'))[0]
data219 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data219)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1'))[0]
data2110 = Data['Lap-wise Average Velocity'][idx]
print_estimator(data2110, end='\n\n')

print("Stage 1 Maze 1 vs Stage 2 Maze 2: Day 1")
print(levene(data1, data221))
print(ttest_ind(data1, data221, alternative='less', equal_var=False))

print("Stage 2: Maze 1 vs Maze 2, Day 2")
print(levene(data212, data222))
print(ttest_ind(data212, data222, alternative='greater', equal_var=False))

print("Stage 2: Maze 1 vs Maze 2, Day 3")
print(levene(data213, data223))
print(ttest_ind(data213, data223, alternative='greater', equal_var=False))

print("Stage 2: Maze 1 vs Maze 2, Day 4")
print(levene(data214, data224))
print(ttest_ind(data214, data224, alternative='greater', equal_var=False))

print("Stage 2: Maze 1 vs Maze 2, Day 10")
print(levene(data2210, data221))
print(ttest_ind(data2210, data221, alternative='less', equal_var=False))


idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data_111 = Data['Lap-wise Average Velocity'][idx]
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1'))[0]
data_141 = Data['Lap-wise Average Velocity'][idx]
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1'))[0]
data_2101 = Data['Lap-wise Average Velocity'][idx]

fig = plt.figure(figsize=(2,3))
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('S1', data_111.shape[0]), np.repeat('S4', data_141.shape[0]), np.repeat('S23-26', data_2101.shape[0])]),
    'Speed': np.concatenate([data_111, data_141, data_2101]),
}
sns.barplot(
    x='Condition',
    y='Speed',
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
    y='Speed',
    data=compdata,
    palette=markercolor,
    size=1.5,
    ax = ax,
    jitter=0.2
)
ax.set_ylim([5, 50])
ax.set_ylabel('Lap-wise average velocity (cm/s)')
plt.tight_layout()
plt.savefig(join(loc, "R-S phase [Stage 1].png"), dpi=600)
plt.savefig(join(loc, "R-S phase [Stage 1].svg"), dpi=600)
plt.close()


idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
data_111 = Data['Lap-wise Average Velocity'][idx]
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 2'))[0]
data_141 = Data['Lap-wise Average Velocity'][idx]
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 2'))[0]
data_2101 = Data['Lap-wise Average Velocity'][idx]

fig = plt.figure(figsize=(2,3))
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('S1', data_111.shape[0]), np.repeat('S4', data_141.shape[0]), np.repeat('S10-13', data_2101.shape[0])]),
    'Speed': np.concatenate([data_111, data_141, data_2101]),
}
sns.barplot(
    x='Condition',
    y='Speed',
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
    y='Speed',
    data=compdata,
    palette=markercolor,
    size=1.5,
    ax = ax,
    jitter=0.2
)
ax.set_ylim([5, 50])
ax.set_ylabel('Lap-wise average velocity (cm/s)')
plt.tight_layout()
plt.savefig(join(loc, "R-S phase [Stage 2].png"), dpi=600)
plt.savefig(join(loc, "R-S phase [Stage 2].svg"), dpi=600)
plt.close()


RS1, RS2 = RSPhaseCalculator(Data, 'Lap-wise Average Velocity')
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(np.arange(1, RS1.shape[0]+1), RS1[:, 0], label='Novel Maze 1, to start', linewidth=0.5)
ax.plot(np.arange(1, RS1.shape[0]+1), RS1[:, 1], label='Novel Maze 1, to final', linewidth=0.5)
ax.legend()
plt.savefig(join(loc, "RS Phase Calculate ['Maze 1].png"), dpi=600)
plt.savefig(join(loc, "RS Phase Calculate ['Maze 1].svg"), dpi=600)
plt.close()

fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(np.arange(1, RS2.shape[0]+1), RS2[:, 0], label='Novel Maze 2, to start', linewidth=0.5)
ax.plot(np.arange(1, RS2.shape[0]+1), RS2[:, 1], label='Novel Maze 2, to final', linewidth=0.5)
ax.legend()
plt.savefig(join(loc, "RS Phase Calculate ['Maze 2].png"), dpi=600)
plt.savefig(join(loc, "RS Phase Calculate ['Maze 2].svg"), dpi=600)
plt.close()


# First Exposure\
print("Stage 1 Novel Day 1 -----------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 1))[0]
print("First exposure: ")
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap1 = Data['Lap-wise Average Velocity'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 2))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap2 = Data['Lap-wise Average Velocity'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] >= 5))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap5 = Data['Lap-wise Average Velocity'][idx]

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
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap1 = Data['Lap-wise Average Velocity'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] == 2))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap2 = Data['Lap-wise Average Velocity'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] >= 5))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap5 = Data['Lap-wise Average Velocity'][idx]

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
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap1 = Data['Lap-wise Average Velocity'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 2)&(Data['Maze Type'] == 'Maze 2'))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap2 = Data['Lap-wise Average Velocity'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] >= 5)&(Data['Maze Type'] == 'Maze 2'))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap5 = Data['Lap-wise Average Velocity'][idx]

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
print_estimator(Data['Lap-wise Average Velocity'][idx], end='\n\n')
maze1_lap1 = Data['Lap-wise Average Velocity'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] == 2)&(Data['Maze Type'] == 'Maze 2'))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap2 = Data['Lap-wise Average Velocity'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] >= 5)&(Data['Maze Type'] == 'Maze 2'))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap5 = Data['Lap-wise Average Velocity'][idx]

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
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap1 = Data['Lap-wise Average Velocity'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 2)&(Data['Maze Type'] == 'Maze 1'))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap2 = Data['Lap-wise Average Velocity'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] >= 5)&(Data['Maze Type'] == 'Maze 1'))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap5 = Data['Lap-wise Average Velocity'][idx]

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
print_estimator(Data['Lap-wise Average Velocity'][idx], end='\n\n')
maze1_lap1 = Data['Lap-wise Average Velocity'][idx]

# Second Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] == 2)&(Data['Maze Type'] == 'Maze 1'))[0]
print("Second exposure: ")
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap2 = Data['Lap-wise Average Velocity'][idx]

# >= 5th Exposure
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Lap ID'] >= 5)&(Data['Maze Type'] == 'Maze 1'))[0]
print(">=5th exposure: ")
print_estimator(Data['Lap-wise Average Velocity'][idx])
maze1_lap5 = Data['Lap-wise Average Velocity'][idx]

print("Significant test for lap 1 and lap 2:")
print(levene(maze1_lap1, maze1_lap2))
print(ttest_ind(maze1_lap1, maze1_lap2, alternative='greater'))

print("Significant test for lap 1 and lap 5:")
print(levene(maze1_lap1, maze1_lap5))
print(ttest_ind(maze1_lap1, maze1_lap5, alternative='greater'), end='\n\n')

print("Stage 1 Maze 1 ---------------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data1 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 1:   Mean: {np.mean(data1)}, STD: {np.std(data1)}, min: {np.min(data1)}, max: {np.max(data1)}, median: {np.median(data1)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 1'))[0]
data2 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 2:   Mean: {np.mean(data2)}, STD: {np.std(data2)}, min: {np.min(data2)}, max: {np.max(data2)}, median: {np.median(data2)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 1'))[0]
data3 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 3:   Mean: {np.mean(data3)}, STD: {np.std(data3)}, min: {np.min(data3)}, max: {np.max(data3)}, median: {np.median(data3)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1'))[0]
data4 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 4:   Mean: {np.mean(data4)}, STD: {np.std(data4)}, min: {np.min(data4)}, max: {np.max(data4)}, median: {np.median(data4)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 1'))[0]
data5 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 5:   Mean: {np.mean(data5)}, STD: {np.std(data5)}, min: {np.min(data5)}, max: {np.max(data5)}, median: {np.median(data5)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 1'))[0]
data6 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 6:   Mean: {np.mean(data6)}, STD: {np.std(data6)}, min: {np.min(data6)}, max: {np.max(data6)}, median: {np.median(data6)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 1'))[0]
data7 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 7:   Mean: {np.mean(data7)}, STD: {np.std(data7)}, min: {np.min(data7)}, max: {np.max(data7)}, median: {np.median(data7)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 1'))[0]
data8 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 8:   Mean: {np.mean(data8)}, STD: {np.std(data8)}, min: {np.min(data8)}, max: {np.max(data8)}, median: {np.median(data8)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1'))[0]
data9 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 9:   Mean: {np.mean(data9)}, STD: {np.std(data9)}, min: {np.min(data9)}, max: {np.max(data9)}, median: {np.median(data9)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1'))[0]
data10 = Data['Lap-wise Average Velocity'][idx]
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
data1 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 1:   Mean: {np.mean(data1)}, STD: {np.std(data1)}, min: {np.min(data1)}, max: {np.max(data1)}, median: {np.median(data1)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 2'))[0]
data2 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 2:   Mean: {np.mean(data2)}, STD: {np.std(data2)}, min: {np.min(data2)}, max: {np.max(data2)}, median: {np.median(data2)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 2'))[0]
data3 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 3:   Mean: {np.mean(data3)}, STD: {np.std(data3)}, min: {np.min(data3)}, max: {np.max(data3)}, median: {np.median(data3)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 2'))[0]
data4 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 4:   Mean: {np.mean(data4)}, STD: {np.std(data4)}, min: {np.min(data4)}, max: {np.max(data4)}, median: {np.median(data4)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 2'))[0]
data5 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 5:   Mean: {np.mean(data5)}, STD: {np.std(data5)}, min: {np.min(data5)}, max: {np.max(data5)}, median: {np.median(data5)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 2'))[0]
data6 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 6:   Mean: {np.mean(data6)}, STD: {np.std(data6)}, min: {np.min(data6)}, max: {np.max(data6)}, median: {np.median(data6)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 2'))[0]
data7 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 7:   Mean: {np.mean(data7)}, STD: {np.std(data7)}, min: {np.min(data7)}, max: {np.max(data7)}, median: {np.median(data7)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 2'))[0]
data8 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 8:   Mean: {np.mean(data8)}, STD: {np.std(data8)}, min: {np.min(data8)}, max: {np.max(data8)}, median: {np.median(data8)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 2'))[0]
data9 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 9:   Mean: {np.mean(data9)}, STD: {np.std(data9)}, min: {np.min(data9)}, max: {np.max(data9)}, median: {np.median(data9)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 2'))[0]
data10 = Data['Lap-wise Average Velocity'][idx]
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
data11 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 1:   Mean: {np.mean(data11)}, STD: {np.std(data11)}, min: {np.min(data11)}, max: {np.max(data11)}, median: {np.median(data11)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 1'))[0]
data12 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 2:   Mean: {np.mean(data12)}, STD: {np.std(data12)}, min: {np.min(data12)}, max: {np.max(data12)}, median: {np.median(data12)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 1'))[0]
data13 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 3:   Mean: {np.mean(data13)}, STD: {np.std(data13)}, min: {np.min(data13)}, max: {np.max(data13)}, median: {np.median(data13)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1'))[0]
data14 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 4:   Mean: {np.mean(data14)}, STD: {np.std(data14)}, min: {np.min(data14)}, max: {np.max(data14)}, median: {np.median(data14)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 1'))[0]
data15 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 5:   Mean: {np.mean(data15)}, STD: {np.std(data15)}, min: {np.min(data15)}, max: {np.max(data15)}, median: {np.median(data15)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 1'))[0]
data16 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 6:   Mean: {np.mean(data16)}, STD: {np.std(data16)}, min: {np.min(data16)}, max: {np.max(data16)}, median: {np.median(data16)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 1'))[0]
data17 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 7:   Mean: {np.mean(data17)}, STD: {np.std(data17)}, min: {np.min(data17)}, max: {np.max(data17)}, median: {np.median(data17)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 1'))[0]
data18 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 8:   Mean: {np.mean(data18)}, STD: {np.std(data18)}, min: {np.min(data18)}, max: {np.max(data18)}, median: {np.median(data18)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1'))[0]
data19 = Data['Lap-wise Average Velocity'][idx]
print(f"Day 9:   Mean: {np.mean(data19)}, STD: {np.std(data19)}, min: {np.min(data19)}, max: {np.max(data19)}, median: {np.median(data19)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1'))[0]
data110 = Data['Lap-wise Average Velocity'][idx]
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