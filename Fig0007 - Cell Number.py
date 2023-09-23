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

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,3.2))
ax_pre, ax1, ax2 = axes
colors = sns.color_palette("rocket", 5)[1:-1]
markercolors = sns.color_palette("Blues", 4)[1::]

ax_pre = Clear_Axes(ax_pre, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax1 = Clear_Axes(ax1, close_spines=['top', 'right', 'left'], ifxticks=True)
ax2 = Clear_Axes(ax2, close_spines=['top', 'right', 'left'], ifxticks=True)

x_ticks = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']
s_ticks = ['S'+str(i) for i in range(1, 10)] + ['>=S10']

y_max = np.nanmax(Data['Cell Number'])

pre_indices = np.concatenate([np.where((Data['Stage'] == 'PRE')&(Data['Training Day'] == session))[0] for session in s_ticks])
SubData = SubDict(Data, Data.keys(), idx=pre_indices)

sns.barplot(
    x = 'Training Day',
    y = 'Cell Number',
    hue='Maze Type',
    palette=colors,
    data = SubData,
    ax = ax_pre,
    errwidth=0.8,
    capsize=0.3,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x = 'Training Day',
    y = 'Cell Number',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    size=2,
    ax = ax_pre,
    dodge=True,
    jitter=0.1
)
ax_pre.set_ylim([0, y_max])
ax_pre.set_yticks(ColorBarsTicks(peak_rate=y_max, is_auto=True, tick_number=5))

stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage1_indices)
stage1_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1']])
SubData = SubDict(SubData, SubData.keys(), idx=stage1_indices)
sns.barplot(
    x = 'Training Day', 
    y = 'Cell Number', 
    hue = 'Maze Type',
    data = SubData, 
    ax = ax1,
    palette=colors,
    errwidth=0.8,
    capsize=0.2,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x = 'Training Day',
    y = 'Cell Number',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    size=2,
    ax = ax1,
    dodge=True,
    jitter=0.1
)
ax1.set_ylim([0, y_max])

print("Stage 1 Subdata:")
print(f"  Min {min(SubData['Cell Number'])}, Max: {max(SubData['Cell Number'])}")
print(f"  mean {np.nanmean(SubData['Cell Number'])}, ±std {np.nanstd(SubData['Cell Number'])}")

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.barplot(
    x = 'Training Day', 
    y = 'Cell Number', 
    hue = 'Maze Type', 
    data = SubData, 
    ax = ax2,
    palette=colors,
    errwidth=0.8,
    capsize=0.1,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x = 'Training Day',
    y = 'Cell Number',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    size=2,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax2.set_ylim([0, y_max])

print("Stage 2 Subdata:")
print(f"  Min {min(SubData['Cell Number'])}, Max: {max(SubData['Cell Number'])}")
print(f"  mean {np.nanmean(SubData['Cell Number'])}, ±std {np.nanstd(SubData['Cell Number'])}")
plt.tight_layout()
plt.savefig(join(p, 'Cell Number.png'), dpi=600)
plt.savefig(join(p, 'Cell Number.svg'), dpi=600)
plt.close()

stage12_indices = np.concatenate([np.where((Data['Stage'] != 'PRE')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage12_indices)
print("Stage 1+2 Subdata:")
print(f"  Min {min(SubData['Cell Number'])}, Max: {max(SubData['Cell Number'])}")
print(f"  mean {np.nanmean(SubData['Cell Number'])}, ±std {np.nanstd(SubData['Cell Number'])}")


def estimate_quality(Data, stage: str, maze: str = 'Open Field'):
    mice = np.unique(Data['MiceID'])
    session = []
    dn = []
    
    for i, m in enumerate(mice):
        idx = np.where((Data['MiceID'] == m)&(Data['Stage'] == stage)&(Data['Maze Type'] == 'Open Field'))[0]

        print(Data['Cell Number'][idx])
        print(Data['Training Day'][idx],end='\n\n')
        
        session.append(Data['Training Day'][idx][1:])
        dn.append(np.abs(np.ediff1d(Data['Cell Number'][idx])))
    
    return np.concatenate(session), np.concatenate(dn)

x, y = estimate_quality(Data, 'PRE') # 
data2 = y[np.where((x == 'S2')|(x == 'S3')|(x == 'S4')|(x == 'S5'))[0]]
print(np.mean(data2), np.std(data2))
data10 = y[np.where((x == '>=S20'))[0]]
print(np.mean(data10), np.std(data10))
print(levene(data2, data10))
print(ttest_ind(data2, data10, alternative='greater'))

fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = x,
    y = y,
    ax = ax,
    palette=colors,
    errwidth=0.8,
    capsize=0.1,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x = x,
    y = y,
    palette=markercolors,
    size=2,
    ax = ax,
    jitter=0.3
)
plt.savefig(join(p, 'ΔCell Number.png'), dpi=600)
plt.savefig(join(p, 'ΔCell Number.svg'), dpi=600)
plt.close()