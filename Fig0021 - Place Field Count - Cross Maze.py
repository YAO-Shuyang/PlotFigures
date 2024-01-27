# Fig0021-1, Barplot of field number
# Fig0021-2, Lineplot of field number

from mylib.statistic_test import *

code_id = '0021 - Field Number Change'
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Cell','Field Number'], f = f1, function = PlaceFieldNumber_Interface, 
                              func_kwgs={'is_placecell':True}, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')   
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)


fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,2))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right', 'left'], ifxticks=True)
ax3 = Clear_Axes(axes[2], close_spines=['top', 'right', 'left'], ifxticks=True)

x_ticks = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']
s_ticks = ['S'+str(i) for i in range(1, 20)] + ['>=S20']

idx = np.concatenate([np.where(Data['Training Day'] == session)[0] for session in s_ticks] + [np.where(Data['Training Day'] == day)[0] for day in x_ticks])
Data = SubDict(Data, Data.keys(), idx)

colors = sns.color_palette("rocket", 3)
markercolors = sns.color_palette("Blues", 4)[1::]
idx = np.where(Data['Stage'] == 'PRE')[0]
SubData = SubDict(Data, Data.keys(), idx)

data1 = SubData['Field Number'][np.where(SubData['Training Day'] == 'S1')[0]]
print(np.mean(data1), np.std(data1))
data10 = SubData['Field Number'][np.where(SubData['Training Day'] == '>=S20')[0]]
print(np.mean(data10), np.std(data10))
print(levene(data1, data10))
print(ttest_ind(data1, data10, equal_var=False))

ymax = max(SubData['Field Number'])
print(SubData['Field Number'],ymax)

sns.barplot(
    x='Training Day',
    y='Field Number',
    data=SubData,
    hue='Maze Type',
    palette=colors,
    capsize=0.25,
    errwidth=0.8,
    errcolor='black',
    ax=ax1
)
ax1.set_ylim([0,12])
ax1.set_yticks(np.linspace(0, 12, 5))

stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage1_indices)
stage1_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1']])
SubData = SubDict(SubData, SubData.keys(), idx=stage1_indices)
sns.barplot(
    x='Training Day',
    y='Field Number',
    data=SubData,
    hue='Maze Type',
    palette=colors,
    capsize=0.16,
    errwidth=0.8,
    errcolor='black',
    ax=ax2
)
ax2.set_ylim([0,12])

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.barplot(
    x='Training Day',
    y='Field Number',
    data=SubData,
    hue='Maze Type',
    palette=colors,
    capsize=0.12,
    errwidth=0.8,
    errcolor='black',
    ax=ax3
)
ax3.set_ylim([0,12])
plt.savefig(join(loc, 'Field Number.png'), dpi=600)
plt.savefig(join(loc, 'Field Number.svg'), dpi=600)
plt.close()