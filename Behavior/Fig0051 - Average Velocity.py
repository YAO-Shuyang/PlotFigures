from mylib.statistic_test import *

code_id = '0051 - Average Velocity'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Average Velocity'], f = f_pure_behav, function = AverageVelocity_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze', is_behav=True)
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
x_ticks = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']
s_ticks = ['S'+str(i) for i in range(1, 20)] + ['>=S20']

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,2))
colors = sns.color_palette("rocket", 5)[1:-1]
markercolors = sns.color_palette("Blues", 4)[1::]
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right', 'left'], ifxticks=True)
ax3 = Clear_Axes(axes[2], close_spines=['top', 'right', 'left'], ifxticks=True)
Pre_indices = np.concatenate([np.where((Data['Stage'] == 'PRE')&(Data['Training Day'] == session))[0] for session in s_ticks])
SubData = SubDict(Data, Data.keys(), idx=Pre_indices)

data1 = SubData['Average Velocity'][np.where(SubData['Training Day'] == 'S1')[0]]
print(np.mean(data1), np.std(data1))
data10 = SubData['Average Velocity'][np.where(SubData['Training Day'] == '>=S20')[0]]
print(np.mean(data10), np.std(data10))
print(levene(data1, data10))
print(ttest_ind(data1, data10, alternative='less'))
sns.barplot(
    x = 'Training Day',
    y = 'Average Velocity',
    hue='Maze Type',
    palette=colors,
    data = SubData,
    ax = ax1,
    errwidth=0.8,
    capsize=0.3,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x = 'Training Day',
    y = 'Average Velocity',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    size=2,
    ax = ax1,
    dodge=True,
    jitter=0.1
)
ax1.set_ylim([0, 50])
ax1.set_yticks(np.linspace(0, 50, 5))

stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage1_indices)
stage1_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1']])
SubData = SubDict(SubData, SubData.keys(), idx=stage1_indices)
sns.barplot(
    x = 'Training Day', 
    y = 'Average Velocity', 
    hue = 'Maze Type',
    data = SubData, 
    ax = ax2,
    palette=colors,
    errwidth=0.8,
    capsize=0.2,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x = 'Training Day',
    y = 'Average Velocity',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    size=2,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax2.set_ylim([0, 50])

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.barplot(
    x = 'Training Day', 
    y = 'Average Velocity', 
    hue = 'Maze Type', 
    data = SubData, 
    ax = ax3,
    palette=colors,
    errwidth=0.8,
    capsize=0.1,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x = 'Training Day',
    y = 'Average Velocity',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    size=2,
    ax = ax3,
    dodge=True,
    jitter=0.1
)
ax3.set_ylim([0, 50])

plt.tight_layout()
plt.savefig(join(loc, "Average Velocity.png"), dpi=600)
plt.savefig(join(loc, "Average Velocity.svg"), dpi=600)
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