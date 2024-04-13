from mylib.statistic_test import *

code_id = "0025 - Percentage of PCsf"
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Percentage', 'Criteria'], f = f1, 
                              function = PercentageOfPCsf_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

Data['Percentage'] = Data['Percentage'] *100
idx = np.where((Data['MiceID'] != 11095)&(Data['MiceID'] != 11092))[0]
Data = SubDict(Data, Data.keys(), idx)

idx = np.where(Data['Stage'] != 'PRE')[0]
Data = SubDict(Data, Data.keys(), idx)

idx = np.where(Data['Maze Type']=='Open Field')[0]
Data['Percentage'][idx[np.arange(0, len(idx), 2)]] = (Data['Percentage'][idx[np.arange(0, len(idx), 2)]] + Data['Percentage'][idx[np.arange(1, len(idx), 2)]]) / 2

idx = np.concatenate([idx[np.arange(0, len(idx), 2)],
                      np.where(Data['Maze Type']=='Maze 1')[0],
                      np.where(Data['Maze Type']=='Maze 2')[0]])
Data = SubDict(Data, Data.keys(), idx)

data_op_s1 = Data['Percentage'][np.where((Data['Maze Type']=='Open Field') & (Data['Stage'] == 'Stage 1'))[0]]
data_op_s2 = Data['Percentage'][np.where((Data['Maze Type']=='Open Field') & (Data['Stage'] == 'Stage 2'))[0]]
data_m1_s1 = Data['Percentage'][np.where((Data['Maze Type']=='Maze 1') & (Data['Stage'] == 'Stage 1'))[0]]
data_m1_s2 = Data['Percentage'][np.where((Data['Maze Type']=='Maze 1') & (Data['Stage'] == 'Stage 2'))[0]]
data_m2 = Data['Percentage'][np.where((Data['Maze Type']=='Maze 2'))[0]]

print_estimator(data_op_s1)
print_estimator(data_op_s2)
print_estimator(data_m1_s1)
print_estimator(data_m1_s2)
print_estimator(data_m2)



x_ticks = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']
s_ticks = ['S'+str(i) for i in range(1, 20)] + ['>=S20']

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,2))
colors = sns.color_palette("rocket", 5)[1:-1]
markercolors = sns.color_palette("Blues", 4)[1::]
ax2 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day))[0] for day in x_ticks])
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
ax2.set_ylim(0, 15)
ax2.set_yticks(np.linspace(0, 15, 6))

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.barplot(
    x = 'Training Day', 
    y = 'Percentage', 
    hue = 'Maze Type', 
    hue_order= ['Open Field', 'Maze 1', 'Maze 2'],
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
    hue_order=['Open Field', 'Maze 1', 'Maze 2'],
    palette=markercolors,
    size=3,
    edgecolor='black',
    linewidth=0.3,
    ax = ax3,
    dodge=True,
    jitter=0.1
)
ax3.set_ylim(0, 15)
ax3.set_yticks(np.linspace(0, 15, 6))

plt.tight_layout()
plt.savefig(join(loc, "Percentage of PCsf.png"), dpi=600)
plt.savefig(join(loc, "Percentage of PCsf.svg"), dpi=600)
plt.close()


fig = plt.figure(figsize=(2,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x='Maze Type',
    y='Percentage',
    data=SubData,
    palette=colors,
    ax=ax,
    errcolor='black',
    errwidth=0.5,
    capsize=0.1,
    width=0.8
)
sns.stripplot(
    x='Maze Type',
    y='Percentage',
    data=SubData,
    palette=markercolors,
    edgecolor='black',
    size=4,
    linewidth=0.15,
    ax = ax,
    jitter=0.2
)
ax.set_ylim(0, 15)
ax.set_yticks(np.linspace(0, 15, 6))
plt.savefig(join(loc, 'Comparison of Environments.png'), dpi=2400)
plt.savefig(join(loc, 'Comparison of Environments.svg'), dpi=2400)
plt.close()

print("\nCross Environments:")
data_op = SubData['Percentage'][np.where((SubData['Maze Type']=='Open Field')&
                                           ((SubData['Stage'] != 'Stage 1') | (SubData['Training Day'] != 'Day 6')))[0]]
data_m1 = SubData['Percentage'][np.where((SubData['Maze Type']=='Maze 1')&
                                           ((SubData['Stage'] != 'Stage 1') | (SubData['Training Day'] != 'Day 6')))[0]]

print_estimator(data_op)
print_estimator(data_m1)
print("OF vs M1: ", ttest_rel(data_op, data_m1), end='\n\n')
data_op_m2 = SubData['Percentage'][np.where((SubData['Maze Type']=='Open Field')&(SubData['Stage'] == 'Stage 2'))[0]]
data_m2 = SubData['Percentage'][np.where((SubData['Maze Type']=='Maze 2')&(SubData['Stage'] == 'Stage 2'))[0]]
print_estimator(data_op_m2)
print_estimator(data_m2)
print("OF vs M2: ", ttest_rel(data_op_m2, data_m2), end='\n\n')