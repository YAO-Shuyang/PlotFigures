# Fig0023-1, Place cell percentage - Linegraph

from mylib.statistic_test import *
code_id = '0023 - Place cell percentage'
loc = os.path.join(figpath, code_id)
mkdir(loc)

file_idx = np.where(np.isin(f1['MiceID'], [10209, 10212, 10224, 10226, 10228, 10232, 10234]))[0]
if os.path.exists(os.path.join(figdata, code_id+' .pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['percentage', 'place cell num', 'total cell num'], f = f1, 
                              function = PlaceCellPercentage_Interface, file_idx = file_idx,
                              file_name = code_id, behavior_paradigm = 'CrossMaze')   
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata, code_id+'-CP.pkl')) == False:
    CPData = DataFrameEstablish(variable_names = ['percentage', 'place cell num', 'total cell num'], f = f1, function = PlaceCellPercentageCP_Interface, 
                              file_name = code_id+'-CP', behavior_paradigm = 'CrossMaze')   
else:
    with open(os.path.join(figdata, code_id+'-CP.pkl'), 'rb') as handle:
        CPData = pickle.load(handle)

idx = np.where((Data['MiceID'] != 11092)&(Data['MiceID'] != 11095))[0]
Data = SubDict(Data, Data.keys(), idx=idx)
CPData = SubDict(CPData, CPData.keys(), idx=idx)

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,2))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True)
ax3 = Clear_Axes(axes[2], close_spines=['top', 'right'], ifxticks=True)

x_ticks = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']
s_ticks = ['S'+str(i) for i in range(1, 20)] + ['>=S20']

idx = np.concatenate([np.where(Data['Training Day'] == session)[0] for session in s_ticks] + [np.where(Data['Training Day'] == day)[0] for day in x_ticks])
Data = SubDict(Data, Data.keys(), idx)
Data['percentage'] = Data['percentage']*100

colors = sns.color_palette("rocket", 3)
markercolors = sns.color_palette("Blues", 4)[1::]
idx = np.where(Data['Stage'] == 'PRE')[0]
SubData = SubDict(Data, Data.keys(), idx)

sns.lineplot(
    x='Training Day',
    y='percentage',
    hue='Maze Type',
    data=SubData,
    #style=Data['MiceID'][idx],
    palette=colors,
    err_style='bars',
    ax=ax1, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax1.set_ylim([0,100])
ax1.set_yticks(np.linspace(0, 100, 6))

stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage1_indices)
stage1_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1']])
SubData = SubDict(SubData, SubData.keys(), idx=stage1_indices)
sns.lineplot(
    x='Training Day',
    y='percentage',
    hue='Maze Type',
    data=SubData,
    #style=Data['MiceID'][idx],
    palette=colors,
    err_style='bars',
    ax=ax2, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax2.set_ylim([0,100])
ax2.set_yticks(np.linspace(0, 100, 6))

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.lineplot(
    x='Training Day',
    y='percentage',
    hue='Maze Type',
    data=SubData,
    #style=Data['MiceID'][idx],
    palette=colors,
    err_style='bars',
    ax=ax3, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax3.set_ylim([0,100])
ax3.set_yticks(np.linspace(0, 100, 6))
plt.savefig(join(loc, 'place cell percentage.png'), dpi=600)
plt.savefig(join(loc, 'place cell percentage.svg'), dpi=600)
plt.close()


fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,2))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True)
ax3 = Clear_Axes(axes[2], close_spines=['top', 'right'], ifxticks=True)

x_ticks = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']
s_ticks = ['S'+str(i) for i in range(1, 20)] + ['>=S20']

idx = np.concatenate([np.where(CPData['Training Day'] == session)[0] for session in s_ticks] + [np.where(CPData['Training Day'] == day)[0] for day in x_ticks])
CPData = SubDict(CPData, Data.keys(), idx)
CPData['percentage'] = CPData['percentage']*100

colors = sns.color_palette("rocket", 3)
markercolors = sns.color_palette("Blues", 4)[1::]
idx = np.where(Data['Stage'] == 'PRE')[0]
SubData = SubDict(CPData, CPData.keys(), idx)

sns.lineplot(
    x='Training Day',
    y='percentage',
    hue='Maze Type',
    data=SubData,
    #style=Data['MiceID'][idx],
    palette=colors,
    err_style='bars',
    ax=ax1, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax1.set_ylim([0,100])
ax1.set_yticks(np.linspace(0, 100, 6))

stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage1_indices)
stage1_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1']])
SubData = SubDict(SubData, SubData.keys(), idx=stage1_indices)
sns.lineplot(
    x='Training Day',
    y='percentage',
    hue='Maze Type',
    data=SubData,
    #style=Data['MiceID'][idx],
    palette=colors,
    err_style='bars',
    ax=ax2, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax2.set_ylim([0,100])
ax2.set_yticks(np.linspace(0, 100, 6))

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.lineplot(
    x='Training Day',
    y='percentage',
    hue='Maze Type',
    data=SubData,
    #style=Data['MiceID'][idx],
    palette=colors,
    err_style='bars',
    ax=ax3, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax3.set_ylim([0,100])
ax3.set_yticks(np.linspace(0, 100, 6))
plt.savefig(join(loc, 'place cell percentage [CP].png'), dpi=600)
plt.savefig(join(loc, 'place cell percentage [CP].svg'), dpi=600)
plt.close()


data1_op = Data['percentage'][np.where((Data['Training Day'] == 'Day 1')&(Data['Stage'] == 'Stage 1')&(Data['Maze Type'] == 'Open Field'))[0]]
data3_op = Data['percentage'][np.where((Data['Training Day'] == 'Day 3')&(Data['Stage'] == 'Stage 1')&(Data['Maze Type'] == 'Open Field'))[0]]
data1_m1 = Data['percentage'][np.where((Data['Training Day'] == 'Day 1')&(Data['Stage'] == 'Stage 1')&(Data['Maze Type'] == 'Maze 1'))[0]]
data3_m1 = Data['percentage'][np.where((Data['Training Day'] == 'Day 3')&(Data['Stage'] == 'Stage 1')&(Data['Maze Type'] == 'Maze 1'))[0]]

print_estimator(data1_op)
print_estimator(data3_op)
print_estimator(data1_m1)
print_estimator(data3_m1)
print("Stage 1 Day 1 vs Day 3 [Open Field]:")
print(levene(data1_op, data3_op))
print(ttest_ind(data1_op, data3_op))
print("Stage 1 Day 1 vs Day 3 [Maze 1]:")
print(levene(data1_m1, data3_m1))
print(ttest_ind(data1_m1, data3_m1, alternative='less'))


days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9', '>=Day 10']

print("Stage 1 Daily comparison --------------------------------------------------------------------------------------")
for d in days:
    dataop = Data['percentage'][np.where((Data['Training Day'] == d)&(Data['Stage'] == 'Stage 1')&(Data['Maze Type'] == 'Open Field'))[0]]
    datam1 = Data['percentage'][np.where((Data['Training Day'] == d)&(Data['Stage'] == 'Stage 1')&(Data['Maze Type'] == 'Maze 1'))[0]]
    dataop = (dataop[np.arange(0, len(dataop), 2)] + dataop[np.arange(1, len(dataop), 2)])/2
    print_estimator(dataop)
    print_estimator(datam1)
    print("  "+d+" [Open Field vs Maze 1]:")
    print("    ", levene(dataop, datam1))
    try:
        print("    ", ttest_rel(dataop, datam1), end="\n\n")
    except:
        pass
    
print("Stage 2 Daily comparison --------------------------------------------------------------------------------------")
for d in days:
    print("  ", d)
    dataop = Data['percentage'][np.where((Data['Training Day'] == d)&(Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Open Field'))[0]]
    datam1 = Data['percentage'][np.where((Data['Training Day'] == d)&(Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 1'))[0]]
    datam2 = Data['percentage'][np.where((Data['Training Day'] == d)&(Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 2'))[0]]
    dataop = (dataop[np.arange(0, len(dataop), 2)] + dataop[np.arange(1, len(dataop), 2)])/2
    print_estimator(datam1)
    print_estimator(datam2)
    
    print("     op vs m1:", ttest_rel(dataop, datam1))
    print("     op vs m2:", ttest_rel(dataop, datam2))
    print("     m1 vs m2:", ttest_rel(datam1, datam2), end="\n\n")
    
print("Percentage Stage 1 Maze 1:")
print_estimator(Data['percentage'][np.where((Data['Stage'] == 'Stage 1')&(Data['Maze Type'] == 'Maze 1'))[0]])
print("Percentage Stage 1 Open Field:")
print_estimator(Data['percentage'][np.where((Data['Stage'] == 'Stage 1')&(Data['Maze Type'] == 'Open Field'))[0]])
print("Percentage Stage 2 Maze 1:")
print_estimator(Data['percentage'][np.where((Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 1'))[0]])
print("Percentage Stage 2 Maze 2:")
print_estimator(Data['percentage'][np.where((Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 2'))[0]])
print("Percentage Stage 2 Open Field:")
print_estimator(Data['percentage'][np.where((Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Open Field'))[0]])