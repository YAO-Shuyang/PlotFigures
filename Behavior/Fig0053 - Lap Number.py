from mylib.statistic_test import *

code_id = '0053 - Lap Number'
loc = os.path.join(figpath, code_id)
mkdir(loc)

maze_indices = np.where(f1['maze_type'] != 0)[0]
if os.path.exists(join(figdata, code_id+' .pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Lap Num'], is_behav=True,
                              file_idx=maze_indices,
                              f = f1, function = LapNum_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')

print("Trained lap number before DSP:")
for mouse in [10212, 10224, 10227, 10232]:
    if mouse == 10212:
        idx = np.where((Data['MiceID'] == mouse)&(Data['Maze Type'] == 'Maze 1')&(Data['Stage'] == 'Stage 1'))[0]
        print(f"{mouse}: {np.sum(Data['Lap Num'][idx])}")
    else:
        idx = np.where((Data['MiceID'] == mouse)&(Data['Maze Type'] == 'Maze 1'))[0]
        print(f"{mouse}: {np.sum(Data['Lap Num'][idx])}")

uniq_day = ['Day 1', 'Day 2', 'Day 3', 'Day 4',
            'Day 5', 'Day 6', 'Day 7', 'Day 8',
            'Day 9', '>=Day 10']

idx = np.where((Data['MiceID']!=11094)&(Data['MiceID']!=11092))[0]
Data = SubDict(Data, Data.keys(), idx)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,2))
ax1, ax2 = axes[0], axes[1]
ax1 = Clear_Axes(ax1, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(ax2, close_spines=['top', 'right', 'left'], ifxticks=True)

colors = sns.color_palette("rocket", 4)
colors = [colors[0], colors[-1]]
markercolors = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[0]]

idx = np.where(Data['Stage'] == 'Stage 1')
sns.lineplot(
    x=Data['Training Day'][idx],
    y=Data['Lap Num'][idx],
    hue=Data['Maze Type'][idx],
    ax=ax1,
    marker='o',
    palette=colors,
    err_style='bars',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax1.set_ylim([0,40])
ax1.set_yticks(np.linspace(0,40,5))

idx = np.where(Data['Stage'] == 'Stage 2')
sns.lineplot(
    x=Data['Training Day'][idx],
    y=Data['Lap Num'][idx],
    hue=Data['Maze Type'][idx],
    ax=ax2,
    marker='o',
    palette=colors,
    err_style='bars',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax2.set_ylim([0,40])
plt.savefig(join(loc, 'lap number.png'), dpi=600)
plt.savefig(join(loc, 'lap number.svg'), dpi=600)
plt.close()


idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10'))[0]
data10 = Data['Lap Num'][idx]
print(f"Day 10:  Mean: {np.mean(data10)}, STD: {np.std(data10)}")



mice = np.unique(Data['MiceID'])
total_laps = []

for m in mice:
    idx = np.where((Data['MiceID'] == m)&(Data['Stage'] == 'Stage 1'))[0]
    total_laps.append(np.sum(Data['Lap Num'][idx]))

print(mice)
print("Stage 1 Maze 1 -----------------------------------------------------------", total_laps, f" Mean: {np.mean(total_laps)}, STD: {np.std(total_laps)}", end='\n\n')

total_laps = []
for m in mice:
    idx = np.where((Data['MiceID'] == m)&(Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 1'))[0]
    total_laps.append(np.sum(Data['Lap Num'][idx]))

print(mice)
print("Stage 2 Maze 1 -----------------------------------------------------------", total_laps, f" Mean: {np.mean(total_laps)}, STD: {np.std(total_laps)}", end='\n\n')


total_laps = []
for m in mice:
    idx = np.where((Data['MiceID'] == m)&(Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 2'))[0]
    total_laps.append(np.sum(Data['Lap Num'][idx]))

print(mice)
print("Stage 2 Maze 2 -----------------------------------------------------------", total_laps, f" Mean: {np.mean(total_laps)}, STD: {np.std(total_laps)}", end='\n\n')


print("Stage 1 Maze 1 -----------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1'))[0]
data1 = Data['Lap Num'][idx]
print(f"Day 1:   Mean: {np.mean(data1)}, STD: {np.std(data1)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 2'))[0]
data2 = Data['Lap Num'][idx]
print(f"Day 2:   Mean: {np.mean(data2)}, STD: {np.std(data2)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 3'))[0]
data3 = Data['Lap Num'][idx]
print(f"Day 3:   Mean: {np.mean(data3)}, STD: {np.std(data3)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 4'))[0]
data4 = Data['Lap Num'][idx]
print(f"Day 4:   Mean: {np.mean(data4)}, STD: {np.std(data4)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10'))[0]
data10 = Data['Lap Num'][idx]
print(f"Day 10:  Mean: {np.mean(data10)}, STD: {np.std(data10)}")
print(ttest_ind(data1, data10, alternative='less'))
print(ttest_ind(data2, data10, alternative='less'))
print(ttest_ind(data3, data10, alternative='less'))
print(ttest_ind(data4, data10, alternative='less'), end='\n\n')

print("Stage 2 Maze 2 -----------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
data221 = Data['Lap Num'][idx]
print(f"Day 1:   Mean: {np.mean(data221)}, STD: {np.std(data221)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 2'))[0]
data222 = Data['Lap Num'][idx]
print(f"Day 2:   Mean: {np.mean(data222)}, STD: {np.std(data222)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 2'))[0]
data223 = Data['Lap Num'][idx]
print(f"Day 3:   Mean: {np.mean(data223)}, STD: {np.std(data223)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 2'))[0]
data224 = Data['Lap Num'][idx]
print(f"Day 4:   Mean: {np.mean(data224)}, STD: {np.std(data224)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 2'))[0]
data2210 = Data['Lap Num'][idx]
print(f"Day 10:  Mean: {np.mean(data2210)}, STD: {np.std(data2210)}", end='\n\n')

print("Stage 2 Maze 1 -----------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data211 = Data['Lap Num'][idx]
print(f"Day 1:   Mean: {np.mean(data211)}, STD: {np.std(data211)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 1'))[0]
data212 = Data['Lap Num'][idx]
print(f"Day 2:   Mean: {np.mean(data212)}, STD: {np.std(data212)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 1'))[0]
data213 = Data['Lap Num'][idx]
print(f"Day 3:   Mean: {np.mean(data213)}, STD: {np.std(data213)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1'))[0]
data214 = Data['Lap Num'][idx]
print(f"Day 4:   Mean: {np.mean(data214)}, STD: {np.std(data214)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1'))[0]
data2110 = Data['Lap Num'][idx]
print("Stage 1 Maze 1 vs Stage 2 Maze 2: Day 1")
print(f"Day 10:  Mean: {np.mean(data2110)}, STD: {np.std(data2110)}", end='\n\n')
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