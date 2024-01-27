from mylib.statistic_test import *
from scipy.stats import levene

code_id = "0012 - Total Path Length"
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')):
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Path Length'], is_behav=True,
                              f = f_pure_behav, function = TotalPathLength_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')

Data['Path Length'] = Data['Path Length'] / 100

#idx = np.where((Data['MiceID']!=11094)&(Data['MiceID']!=11092))[0]
#Data = SubDict(Data, Data.keys(), idx)

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

data1 = SubData['Path Length'][np.where(SubData['Training Day'] == 'S1')[0]]
print(np.mean(data1), np.std(data1))
data10 = SubData['Path Length'][np.where(SubData['Training Day'] == '>=S20')[0]]
print(np.mean(data10), np.std(data10))
print(levene(data1, data10))
print(ttest_ind(data1, data10, alternative='less'))
sns.barplot(
    x = 'Training Day',
    y = 'Path Length',
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
    y = 'Path Length',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    size=3,
    edgecolor='black',
    linewidth=0.3,
    ax = ax1,
    dodge=True,
    jitter=0.1
)
ax1.set_ylim([0, 400])
ax1.set_yticks(np.linspace(0, 400, 5))

stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage1_indices)
stage1_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1']])
SubData = SubDict(SubData, SubData.keys(), idx=stage1_indices)
sns.barplot(
    x = 'Training Day', 
    y = 'Path Length', 
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
    y = 'Path Length',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    size=3,
    edgecolor='black',
    linewidth=0.3,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax2.set_ylim([0, 400])

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.barplot(
    x = 'Training Day', 
    y = 'Path Length', 
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
    y = 'Path Length',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    size=3,
    edgecolor='black',
    linewidth=0.3,
    ax = ax3,
    dodge=True,
    jitter=0.1
)
ax3.set_ylim([0, 400])

plt.tight_layout()
plt.savefig(join(loc, "Total Path Length per Session.png"), dpi=600)
plt.savefig(join(loc, "Total Path Length per Session.svg"), dpi=600)
plt.close()


idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Maze Type'] == 'Maze 1'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
print(f"Maze 1: Mean: {np.mean(SubData['Path Length'])}, STD: {np.std(SubData['Path Length'])}")
print(f"Least: {np.min(SubData['Path Length'])}, Most: {np.max(SubData['Path Length'])}", end='\n\n')

l = []
for m in np.unique(SubData['MiceID']):
    idx = np.where(SubData['MiceID'] == m)[0]
    l.append(np.sum(SubData['Path Length'][idx]))
print(f"Total Distance within Maze 1: Mean: {np.mean(l)}, STD: {np.std(l)}")
print(f"Least: {np.min(l)}, Most: {np.max(l)}", end='\n\n')
    

def get_data(Data):
    mice = np.unique(Data['MiceID'])
    dates = np.unique(Data['date'])
    
    res = {
        'MiceID': [],
        'Date': [],
        'Total Path Length': [],
        'Training Day': [],
        'Stage': []
    }
    
    for m in mice:
        for d in dates:
            idx = np.where((Data['MiceID'] == m)&(Data['date'] == d))[0]
            if len(idx) == 0:
                continue
            res['MiceID'].append(m)
            res['Date'].append(d)
            res['Total Path Length'].append(np.nansum(Data['Path Length'][idx]))
            res['Training Day'].append(Data['Training Day'][idx[0]])
            res['Stage'].append(Data['Stage'][idx[0]])
    
    # transform to numpy array
    for key in res.keys():
        res[key] = np.array(res[key])
    
    return res

SubData = get_data(Data)

print("Stage 1 Maze 1 -------------------------------------------------------------------")
idx = np.where((SubData['Stage'] == 'Stage 1')&(SubData['Training Day'] == 'Day 1'))[0]
SubData1 = SubData['Total Path Length'][idx]
print(f"Day 1:   Mean: {np.mean(SubData1)}, STD: {np.std(SubData1)}")
idx = np.where((SubData['Stage'] == 'Stage 1')&(SubData['Training Day'] == 'Day 2'))[0]
SubData2 = SubData['Total Path Length'][idx]
print(f"Day 2:   Mean: {np.mean(SubData2)}, STD: {np.std(SubData2)}")
idx = np.where((SubData['Stage'] == 'Stage 1')&(SubData['Training Day'] == 'Day 3'))[0]
SubData3 = SubData['Total Path Length'][idx]
print(f"Day 3:   Mean: {np.mean(SubData3)}, STD: {np.std(SubData3)}")
idx = np.where((SubData['Stage'] == 'Stage 1')&(SubData['Training Day'] == 'Day 4'))[0]
SubData4 = SubData['Total Path Length'][idx]
print(f"Day 4:   Mean: {np.mean(SubData4)}, STD: {np.std(SubData4)}")
idx = np.where((SubData['Stage'] == 'Stage 1')&(SubData['Training Day'] == 'Day 5'))[0]
SubData5 = SubData['Total Path Length'][idx]
print(f"Day 5:   Mean: {np.mean(SubData5)}, STD: {np.std(SubData5)}")
idx = np.where((SubData['Stage'] == 'Stage 1')&(SubData['Training Day'] == 'Day 6'))[0]
SubData6 = SubData['Total Path Length'][idx]
print(f"Day 6:   Mean: {np.mean(SubData6)}, STD: {np.std(SubData6)}")
idx = np.where((SubData['Stage'] == 'Stage 1')&(SubData['Training Day'] == 'Day 7'))[0]
SubData7 = SubData['Total Path Length'][idx]
print(f"Day 7:   Mean: {np.mean(SubData7)}, STD: {np.std(SubData7)}")
idx = np.where((SubData['Stage'] == 'Stage 1')&(SubData['Training Day'] == 'Day 8'))[0]
SubData8 = SubData['Total Path Length'][idx]
print(f"Day 8:   Mean: {np.mean(SubData8)}, STD: {np.std(SubData8)}")
idx = np.where((SubData['Stage'] == 'Stage 1')&(SubData['Training Day'] == 'Day 9'))[0]
SubData9 = SubData['Total Path Length'][idx]
print(f"Day 9:   Mean: {np.mean(SubData9)}, STD: {np.std(SubData9)}")
idx = np.where((SubData['Stage'] == 'Stage 1')&(SubData['Training Day'] == '>=Day 10'))[0]
SubData10 = SubData['Total Path Length'][idx]
print(f"Day 10:  Mean: {np.mean(SubData10)}, STD: {np.std(SubData10)}")
print("Day 1 vs Day 10:  ", ttest_ind(SubData1, SubData10))
print("Day 2 vs Day 10:  ", ttest_ind(SubData2, SubData10))
print("Day 3 vs Day 10:  ", ttest_ind(SubData3, SubData10))
print("Day 4 vs Day 10:  ", ttest_ind(SubData4, SubData10))
print("Day 5 vs Day 10:  ", ttest_ind(SubData5, SubData10))
print("Day 6 vs Day 10:  ", ttest_ind(SubData6, SubData10))
print("Day 7 vs Day 10:  ", ttest_ind(SubData7, SubData10))
print("Day 8 vs Day 10:  ", ttest_ind(SubData8, SubData10))
print("Day 9 vs Day 10:  ", ttest_ind(SubData9, SubData10), end="\n\n")

idx = np.where(SubData['Stage'] == 'Stage 1')[0]
print(f"Total Path Length per Day:  Mean: {np.mean(SubData['Total Path Length'][idx])}, STD: {np.std(SubData['Total Path Length'][idx])}")
print(f"Least: {np.min(SubData['Total Path Length'][idx])}, Most: {np.max(SubData['Total Path Length'][idx])}", end="\n\n")

print("Stage 2 Maze 2 ---------------------------------------------------------------")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 1'))[0]
print(idx)
SubData1 = SubData['Total Path Length'][idx]
print(f"Day 1:   Mean: {np.mean(SubData1)}, STD: {np.std(SubData1)}, min: {np.min(SubData1)}, max: {np.max(SubData1)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 2'))[0]
SubData2 = SubData['Total Path Length'][idx]
print(f"Day 2:   Mean: {np.mean(SubData2)}, STD: {np.std(SubData2)}, min: {np.min(SubData2)}, max: {np.max(SubData2)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 3'))[0]
SubData3 = SubData['Total Path Length'][idx]
print(f"Day 3:   Mean: {np.mean(SubData3)}, STD: {np.std(SubData3)}, min: {np.min(SubData3)}, max: {np.max(SubData3)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 4'))[0]
SubData4 = SubData['Total Path Length'][idx]
print(f"Day 4:   Mean: {np.mean(SubData4)}, STD: {np.std(SubData4)}, min: {np.min(SubData4)}, max: {np.max(SubData4)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 5'))[0]
SubData5 = SubData['Total Path Length'][idx]
print(f"Day 5:   Mean: {np.mean(SubData5)}, STD: {np.std(SubData5)}, min: {np.min(SubData5)}, max: {np.max(SubData5)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 6'))[0]
SubData6 = SubData['Total Path Length'][idx]
print(f"Day 6:   Mean: {np.mean(SubData6)}, STD: {np.std(SubData6)}, min: {np.min(SubData6)}, max: {np.max(SubData6)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 7'))[0]
SubData7 = SubData['Total Path Length'][idx]
print(f"Day 7:   Mean: {np.mean(SubData7)}, STD: {np.std(SubData7)}, min: {np.min(SubData7)}, max: {np.max(SubData7)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 8'))[0]
SubData8 = SubData['Total Path Length'][idx]
print(f"Day 8:   Mean: {np.mean(SubData8)}, STD: {np.std(SubData8)}, min: {np.min(SubData8)}, max: {np.max(SubData8)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 9'))[0]
SubData9 = SubData['Total Path Length'][idx]
print(f"Day 9:   Mean: {np.mean(SubData9)}, STD: {np.std(SubData9)}, min: {np.min(SubData9)}, max: {np.max(SubData9)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == '>=Day 10'))[0]
SubData10 = SubData['Total Path Length'][idx]
print(f"Day 10:  Mean: {np.mean(SubData10)}, STD: {np.std(SubData10)}, min: {np.min(SubData10)}, max: {np.max(SubData10)}", end='\n\n')


print("Stage 2 Maze 1 ---------------------------------------------------------------")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 1'))[0]
SubData11 = SubData['Total Path Length'][idx]
print(f"Day 1:   Mean: {np.mean(SubData11)}, STD: {np.std(SubData11)}, min: {np.min(SubData11)}, max: {np.max(SubData11)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 2'))[0]
SubData12 = SubData['Total Path Length'][idx]
print(f"Day 2:   Mean: {np.mean(SubData12)}, STD: {np.std(SubData12)}, min: {np.min(SubData12)}, max: {np.max(SubData12)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 3'))[0]
SubData13 = SubData['Total Path Length'][idx]
print(f"Day 3:   Mean: {np.mean(SubData13)}, STD: {np.std(SubData13)}, min: {np.min(SubData13)}, max: {np.max(SubData13)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 4'))[0]
SubData14 = SubData['Total Path Length'][idx]
print(f"Day 4:   Mean: {np.mean(SubData14)}, STD: {np.std(SubData14)}, min: {np.min(SubData14)}, max: {np.max(SubData14)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 5'))[0]
SubData15 = SubData['Total Path Length'][idx]
print(f"Day 5:   Mean: {np.mean(SubData15)}, STD: {np.std(SubData15)}, min: {np.min(SubData15)}, max: {np.max(SubData15)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 6'))[0]
SubData16 = SubData['Total Path Length'][idx]
print(f"Day 6:   Mean: {np.mean(SubData16)}, STD: {np.std(SubData16)}, min: {np.min(SubData16)}, max: {np.max(SubData16)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 7'))[0]
SubData17 = SubData['Total Path Length'][idx]
print(f"Day 7:   Mean: {np.mean(SubData17)}, STD: {np.std(SubData17)}, min: {np.min(SubData17)}, max: {np.max(SubData17)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 8'))[0]
SubData18 = SubData['Total Path Length'][idx]
print(f"Day 8:   Mean: {np.mean(SubData18)}, STD: {np.std(SubData18)}, min: {np.min(SubData18)}, max: {np.max(SubData18)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == 'Day 9'))[0]
SubData19 = SubData['Total Path Length'][idx]
print(f"Day 9:   Mean: {np.mean(SubData19)}, STD: {np.std(SubData19)}, min: {np.min(SubData19)}, max: {np.max(SubData19)}")
idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] == '>=Day 10'))[0]
SubData110 = SubData['Total Path Length'][idx]
print(f"Day 10:  Mean: {np.mean(SubData110)}, STD: {np.std(SubData110)}, min: {np.min(SubData110)}, max: {np.max(SubData110)}", end='\n\n')

print("Stage 1: t-test, maze 2 vs maze 1")
print(ttest_ind(SubData1, SubData110))
print(ttest_ind(SubData2, SubData12))
print(ttest_ind(SubData3, SubData13))
print(ttest_ind(SubData4, SubData14))
print(ttest_ind(SubData5, SubData15))
print(ttest_ind(SubData6, SubData16))
print(ttest_ind(SubData7, SubData17))
print(ttest_ind(SubData8, SubData18))
print(ttest_ind(SubData9, SubData19))
print(ttest_ind(SubData10, SubData110), end='\n\n\n')


y_max = np.max(SubData['Total Path Length'])


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right', 'left'], ifxticks=True)

idx = np.where(SubData['Stage'] == 'Stage 1')[0]
print(idx)
AData = SubDict(SubData, SubData.keys(), idx=idx)
sns.barplot(
    x = 'Training Day',
    y = 'Total Path Length',
    data=AData,
    ax = ax1,
    capsize=0.4,
    errcolor='black',
    errwidth=0.5,
    width=0.8,
    palette='rocket'
)
sns.stripplot(
    x = 'Training Day',
    y = 'Total Path Length',
    data=AData,
    ax = ax1,
    jitter=0.3,
    edgecolor='black',
    size=5,
    linewidth=0.15,
    palette='Blues'
)
ax1.set_ylim([0, 1500])
ax1.set_yticks(np.linspace(0, 1500, 4))

idx = np.where(SubData['Stage'] == 'Stage 2')[0]
AData = SubDict(SubData, SubData.keys(), idx=idx)
sns.barplot(
    x = 'Training Day',
    y = 'Total Path Length',
    data=AData,
    ax = ax2,
    capsize=0.4,
    errcolor='black',
    errwidth=0.5,
    width=0.8,
    palette='rocket'
)
sns.stripplot(
    x = 'Training Day',
    y = 'Total Path Length',
    data=AData,
    ax = ax2,
    jitter=0.3,
    palette='Blues',
    edgecolor='black',
    size=5,
    linewidth=0.15,
)
ax2.set_ylim([0, 1500])
plt.tight_layout()
plt.savefig(join(loc, 'total path length per day.png'), dpi=600)
plt.savefig(join(loc, 'total path length per day.svg'), dpi=600)
plt.close()