from mylib.statistic_test import *

code_id = '0036 - Coverage Change'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Coverage', 'Bin Number'], f = f_pure_behav, function = Coverage_Interface, 
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
Pre_indices = np.concatenate([np.where((Data['Stage'] == 'PRE')&(Data['Training Day'] == session)&(Data['Bin Number'] == 48))[0] for session in s_ticks])
SubData = SubDict(Data, Data.keys(), idx=Pre_indices)

data1 = SubData['Coverage'][np.concatenate([np.where((SubData['Training Day'] == 'S'+str(i))&(SubData['Bin Number'] == 48))[0] for i in range(1, 6)])]
print(np.mean(data1), np.std(data1))
data10 = SubData['Coverage'][np.concatenate([np.where((SubData['Training Day'] == 'S'+str(i))&(SubData['Bin Number'] == 48))[0] for i in range(16, 20)] + [np.where(SubData['Training Day'] == '>=S20')[0]])]
print(np.mean(data10), np.std(data10))
print(levene(data1, data10))
print(ttest_ind(data1, data10, equal_var=False, alternative='less'))
sns.barplot(
    x = 'Training Day',
    y = 'Coverage',
    hue='Maze Type',
    palette=colors,
    data = SubData,
    ax = ax1,
    errwidth=0.5,
    capsize=0.3,
    errcolor='black',
    width = 0.7
)
sns.stripplot(
    x = 'Training Day',
    y = 'Coverage',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.3,
    ax = ax1,
    dodge=True,
    jitter=0.3
)
ax1.set_ylim([0, 110])
ax1.set_yticks(np.linspace(0, 100, 5))

stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day)&(Data['Bin Number'] == 48))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage1_indices)
stage1_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1']])
SubData = SubDict(SubData, SubData.keys(), idx=stage1_indices)
sns.barplot(
    x = 'Training Day', 
    y = 'Coverage', 
    hue = 'Maze Type',
    data = SubData, 
    ax = ax2,
    palette=colors,
    errwidth=0.5,
    capsize=0.2,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x = 'Training Day',
    y = 'Coverage',
    data=SubData,
    hue='Maze Type',
    palette=markercolors,
    size=3,
    edgecolor='black',
    linewidth=0.3,
    ax = ax2,
    dodge=True,
    jitter=0.2
)
ax2.set_ylim([0, 110])

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day)&(Data['Bin Number'] == 48))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.barplot(
    x = 'Training Day', 
    y = 'Coverage', 
    hue = 'Maze Type', 
    data = SubData, 
    ax = ax3,
    palette=colors,
    errwidth=0.5,
    capsize=0.1,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x = 'Training Day',
    y = 'Coverage',
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
ax3.set_ylim([0, 110])

plt.tight_layout()
plt.savefig(join(loc, "Coverage.png"), dpi=600)
plt.savefig(join(loc, "Coverage.svg"), dpi=600)
plt.close()

"""
def plot_figure(Data, key, save_loc, file_name):
    idx = np.where((Data['Maze Type'] == key)&(Data['MiceID']!='11094'))[0]
    SubData = SubDict(Data, keys=['MiceID', 'Training Day', 'Coverage', 'bin size', 'Maze Type', 'date'], idx=idx)
    fig = plt.figure(figsize=(4,1.5))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    sns.lineplot(x='date', y='Coverage', data = SubData, hue = 'bin size', ax = ax, err_style='bars', 
                 linewidth = 0.5, err_kws={'elinewidth':0.5, 'capsize':1.5, 'capthick':0.5})
    ax.legend(facecolor = 'white', edgecolor = 'white', bbox_to_anchor = (1,1), 
              loc = 'upper left', title = 'Bin Size', title_fontsize = 8, 
              fontsize = 8)
    days = np.unique(SubData['date']).shape[0]
    ax.set_xticks(np.arange(days), labels = ['D'+str(i) for i in range(1,days+1)])
    ax.set_yticks(np.linspace(40,100,4))
    ax.axis([-0.5, days - 0.5, 40, 100])
    ax.set_ylabel("Coverage %")
    ax.set_xlabel("Training Day")
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, file_name+'.png'), dpi = 600)
    plt.savefig(os.path.join(save_loc, file_name+'.svg'), dpi = 600)
    plt.close()
    
plot_figure(Data, 'Open Field', loc, 'Coverage - Open Field')
plot_figure(Data, 'Maze 1', loc, 'Coverage - Maze 1')
plot_figure(Data, 'Maze 2', loc, 'Coverage - Maze 2')
"""