from mylib.statistic_test import *

code_id = "0025 - Percentage of PCsf"
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Percentage', 'Path Type'], f = f1, 
                              function = PercentageOfPCsf_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

Data['Percentage'] = Data['Percentage'] *100
idx = np.where((Data['MiceID'] != 11095)&(Data['MiceID'] != 11092))[0]
Data = SubDict(Data, Data.keys(), idx)

data_op = Data['Percentage'][np.where((Data['Maze Type']=='Open Field')&(Data['Stage'] != 'PRE'))[0]]
data_m11 = Data['Percentage'][np.where((Data['Maze Type']=='Maze 1')&(Data['Stage'] == 'Stage 1'))[0]]
data_m12 = Data['Percentage'][np.where((Data['Maze Type']=='Maze 1')&(Data['Stage'] == 'Stage 2'))[0]]
data_m2 = Data['Percentage'][np.where((Data['Maze Type']=='Maze 2'))[0]]

print_estimator(data_op)
print_estimator(data_m11)
print_estimator(data_m12)
print_estimator(data_m2)

x_ticks = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']
s_ticks = ['S'+str(i) for i in range(1, 20)] + ['>=S20']

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,2))
colors = sns.color_palette("rocket", 5)[1:-1]
markercolors = sns.color_palette("Blues", 4)[1::]
ax2 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day)&(Data['Path Type'] != 'CP'))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage1_indices)
stage1_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1']])
SubData = SubDict(SubData, SubData.keys(), idx=stage1_indices)
sns.barplot(
    x = 'Training Day', 
    y = 'Percentage', 
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
    y = 'Percentage',
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
ax2.set_ylim(0,30)
ax2.set_yticks(np.linspace(0, 30, 7))

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day)&(Data['Path Type'] != 'CP'))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.barplot(
    x = 'Training Day', 
    y = 'Percentage', 
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
    y = 'Percentage',
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
ax3.set_ylim(0,30)
ax3.set_yticks(np.linspace(0, 30, 7))

plt.tight_layout()
plt.savefig(join(loc, "Percentage of PCsf.png"), dpi=600)
plt.savefig(join(loc, "Percentage of PCsf.svg"), dpi=600)
plt.close()



fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,2))
colors = sns.color_palette("rocket", 5)[1:-1]
markercolors = sns.color_palette("Blues", 4)[1::]
ax2 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

idx = np.where(Data['Maze Type'] != 'Open Field')[0]
Data = SubDict(Data, Data.keys(), idx=idx)

stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day)&(Data['Path Type'] != 'CP'))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage1_indices)
stage1_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1']])
SubData = SubDict(SubData, SubData.keys(), idx=stage1_indices)
sns.barplot(
    x = 'Training Day', 
    y = 'Percentage', 
    hue = 'Maze Type',
    data = SubData, 
    ax = ax2,
    palette=colors[1:],
    errwidth=0.8,
    capsize=0.2,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x = 'Training Day',
    y = 'Percentage',
    data=SubData,
    hue='Maze Type',
    palette=markercolors[1:],
    size=3,
    edgecolor='black',
    linewidth=0.3,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax2.set_ylim(0,8)
ax2.set_yticks(np.linspace(0, 8, 5))

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day)&(Data['Path Type'] != 'CP'))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.barplot(
    x = 'Training Day', 
    y = 'Percentage', 
    hue = 'Maze Type', 
    data = SubData, 
    ax = ax3,
    palette=colors[1:],
    errwidth=0.8,
    capsize=0.1,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x = 'Training Day',
    y = 'Percentage',
    data=SubData,
    hue='Maze Type',
    palette=markercolors[1:],
    size=3,
    edgecolor='black',
    linewidth=0.3,
    ax = ax3,
    dodge=True,
    jitter=0.1
)
ax3.set_ylim(0,20)
ax3.set_yticks(np.linspace(0, 20, 5))

plt.tight_layout()
plt.savefig(join(loc, "Percentage of PCsf [Zoom Out].png"), dpi=600)
plt.savefig(join(loc, "Percentage of PCsf [Zoom Out].svg"), dpi=600)
plt.close()