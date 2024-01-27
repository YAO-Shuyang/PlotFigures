from mylib.statistic_test import *

code_id = "0057 - Mean Rate on Different Path"
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Mean Rate', 'Path Type'], f = f1, function = MeanRateDiffPath_Interface, 
                              func_kwgs={'is_placecell':True}, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')   
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

Data['hue'] = np.array([Data['Maze Type'][i]+'-'+Data['Path Type'][i] for i in range(len(Data['Maze Type']))])

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,2))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right', 'left'], ifxticks=True)
x_ticks = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']
s_ticks = ['S'+str(i) for i in range(1, 20)] + ['>=S20']

idx = np.concatenate([np.where(Data['Training Day'] == session)[0] for session in s_ticks] + [np.where(Data['Training Day'] == day)[0] for day in x_ticks])
Data = SubDict(Data, Data.keys(), idx)

colors = sns.color_palette("rocket", 4)
markercolors = sns.color_palette("Blues", 4)

stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage1_indices)
stage1_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1']])
SubData = SubDict(SubData, SubData.keys(), idx=stage1_indices)
sns.barplot(
    x='Training Day',
    y='Mean Rate',
    data=SubData,
    hue='hue',
    palette=colors,
    capsize=0.2,
    errwidth=0.8,
    errcolor='black',
    ax=ax1
)
sns.stripplot(
    x = 'Training Day',
    y = 'Mean Rate',
    data=SubData,
    hue='hue',
    palette=markercolors,
    size=2,
    ax = ax1,
    dodge=True,
    legend=False,
    jitter=0.1
)
ax1.set_ylim([0,40])

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.barplot(
    x='Training Day',
    y='Mean Rate',
    data=SubData,
    hue='hue',
    palette=colors,
    capsize=0.1,
    errwidth=0.8,
    errcolor='black',
    ax=ax2
)
sns.stripplot(
    x = 'Training Day',
    y = 'Mean Rate',
    data=SubData,
    hue='hue',
    palette=markercolors,
    size=2,
    ax = ax2,
    dodge=True,
    legend=False,
    jitter=0.1
)
ax2.set_ylim([0,40])
plt.savefig(join(loc, 'Mean Rate.png'), dpi=600)
plt.savefig(join(loc, 'Mean Rate.svg'), dpi=600)
plt.close()