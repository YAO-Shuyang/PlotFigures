from mylib.statistic_test import *

code_id = '0038 - Half-half or Odd-even correlation'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Half-half Correlation', 'Odd-even Correlation'], f = f1, 
                              function = InterSessionCorrelation_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

def clear_nan_value(data):
    idx = np.where(np.isnan(data))[0]
    return np.delete(data, idx)
idx = np.where((Data['MiceID'] != 11095)&(Data['MiceID'] != 11092))[0]
Data = SubDict(Data, Data.keys(), idx)

# Mean Rate
uniq_day = ['Day 1', 'Day 2', 'Day 3', 'Day 4',
            'Day 5', 'Day 6', 'Day 7', 'Day 8',
            'Day 9', '>=Day 10']
uniq_s = ['S'+str(i) for i in range(1,20)] + ['>=S20']

idx = np.concatenate([np.where(Data['Training Day'] == day)[0] for day in uniq_day])
SubData = SubDict(Data, Data.keys(), idx)

colors = sns.color_palette("rocket", 3)
markercolors = [sns.color_palette("crest", 4)[2], sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[0]]

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,3))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[2], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.concatenate([np.where((Data['Stage'] == 'PRE')&(Data['Training Day'] == s))[0] for s in uniq_s])
SubData1 = SubDict(Data, Data.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Half-half Correlation',
    data=SubData1,
    hue='Maze Type',
    hue_order=['Open Field'],
    palette=colors,
    marker='o',
    ax=ax1,
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
sns.stripplot(
    x='Training Day',
    y='Half-half Correlation',
    hue='Maze Type',
    data=SubData1,
    hue_order=['Open Field'],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax1,
    dodge=True,
    jitter=0.1
)
ax1.set_ylim([-0.2, 1])
ax1.set_yticks(np.linspace(-0.2, 1, 7))

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 1')&(SubData['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1']])
SubData2 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Half-half Correlation',
    data=SubData2,
    hue='Maze Type',
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax2
)
sns.stripplot(
    x='Training Day',
    y='Half-half Correlation',
    hue='Maze Type',
    data=SubData2,
    hue_order=['Open Field', 'Maze 1'],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax2.set_ylim([-0.2, 1])
ax2.set_yticks(np.linspace(-0.2, 1, 7))

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 2')&(SubData['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1', 'Maze 2']])
SubData3 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Half-half Correlation',
    data=SubData3,
    hue_order=['Open Field', 'Maze 1', 'Maze 2'],
    hue='Maze Type',
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax3
)
sns.stripplot(
    x='Training Day',
    y='Half-half Correlation',
    hue='Maze Type',
    data=SubData3,
    hue_order=['Open Field', 'Maze 1', 'Maze 2'],
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax3,
    dodge=True,
    jitter=0.1
)
ax3.set_ylim([-0.2, 1])
ax3.set_yticks(np.linspace(-0.2, 1, 7))
plt.tight_layout()
plt.savefig(join(loc, 'Half-half Correlation.png'), dpi=2400)
plt.savefig(join(loc, 'Half-half Correlation.svg'), dpi=2400)
plt.close()


fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,3))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[2], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.concatenate([np.where((Data['Stage'] == 'PRE')&(Data['Training Day'] == s))[0] for s in uniq_s])
SubData1 = SubDict(Data, Data.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Odd-even Correlation',
    data=SubData1,
    hue='Maze Type',
    hue_order=['Open Field'],
    palette=colors,
    marker='o',
    ax=ax1,
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
sns.stripplot(
    x='Training Day',
    y='Odd-even Correlation',
    hue='Maze Type',
    hue_order=['Open Field'],
    data=SubData1,
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax1,
    dodge=True,
    jitter=0.1
)
ax1.set_ylim([-0.2, 1])
ax1.set_yticks(np.linspace(-0.2, 1, 7))

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 1')&(SubData['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1']])
SubData2 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Odd-even Correlation',
    data=SubData2,
    hue='Maze Type',
    hue_order=['Open Field', 'Maze 1'],
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax2
)
sns.stripplot(
    x='Training Day',
    y='Odd-even Correlation',
    hue='Maze Type',
    data=SubData2,
    palette=markercolors,
    hue_order=['Open Field', 'Maze 1'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax2.set_ylim([-0.2, 1])
ax2.set_yticks(np.linspace(-0.2, 1, 7))

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 2')&(SubData['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1', 'Maze 2']])
SubData3 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Odd-even Correlation',
    data=SubData3,
    hue='Maze Type',
    hue_order=['Open Field', 'Maze 1', 'Maze 2'],
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax3
)
sns.stripplot(
    x='Training Day',
    y='Odd-even Correlation',
    hue='Maze Type',
    data=SubData3,
    palette=markercolors,
    hue_order=['Open Field', 'Maze 1', 'Maze 2'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax3,
    dodge=True,
    jitter=0.1
)
ax3.set_ylim([-0.2, 1])
ax3.set_yticks(np.linspace(-0.2, 1, 7))
plt.tight_layout()
plt.savefig(join(loc, 'Odd-even Correlation.png'), dpi=2400)
plt.savefig(join(loc, 'Odd-even Correlation.svg'), dpi=2400)
plt.close()



data_op_d1 = SubData['Half-half Correlation'][np.where((SubData['Maze Type'] == 'Open Field')&(SubData['Training Day'] == 'Day 1')&(SubData['Stage'] == 'Stage 1'))[0]]
data_m1_d1 = SubData['Half-half Correlation'][np.where((SubData['Maze Type'] == 'Maze 1')&(SubData['Training Day'] == 'Day 1')&(SubData['Stage'] == 'Stage 1'))[0]]
data_m2_d1 = SubData['Half-half Correlation'][np.where((SubData['Maze Type'] == 'Maze 2')&(SubData['Training Day'] == 'Day 1')&(SubData['Stage'] == 'Stage 2'))[0]]

data_op_d10 = SubData['Half-half Correlation'][np.where((SubData['Maze Type'] == 'Open Field')&(SubData['Training Day'] == '>=Day 10')&(SubData['Stage'] == 'Stage 2'))[0]]
data_m1_d10 = SubData['Half-half Correlation'][np.where((SubData['Maze Type'] == 'Maze 1')&(SubData['Training Day'] == '>=Day 10')&(SubData['Stage'] == 'Stage 2'))[0]]
data_m2_d10 = SubData['Half-half Correlation'][np.where((SubData['Maze Type'] == 'Maze 2')&(SubData['Training Day'] == '>=Day 10')&(SubData['Stage'] == 'Stage 2'))[0]]

print("Open Field: Day 1 vs Day 10")
print_estimator(data_op_d1)
print_estimator(data_op_d10)
print(levene(data_op_d1, data_op_d10))
print(ttest_ind(data_op_d1, data_op_d10, equal_var=False), cohen_d(data_op_d1, data_op_d10), end='\n\n')
print("Maze 1: Day 1 vs Day 10")
print_estimator(data_m1_d1)
print_estimator(data_m1_d10)
print(levene(data_m1_d1, data_m1_d10))
print(ttest_ind(data_m1_d1, data_m1_d10), cohen_d(data_m1_d1, data_m1_d10), end='\n\n')
print("Maze 2: Day 1 vs Day 10")
print_estimator(data_m2_d1)
print_estimator(data_m2_d10)
print(levene(data_m2_d1, data_m2_d10))
print(ttest_ind(data_m2_d1, data_m2_d10), cohen_d(data_m2_d1, data_m2_d10), end='\n\n')

print("Cross Env comparison Stage 1:")
for day in uniq_day:
    print(f" OP vs M1 on Day {day} -------------------------")
    data_op = SubData['Half-half Correlation'][np.where((SubData['Maze Type'] == 'Open Field')&
                                                        (SubData['Training Day'] == day)&
                                                        (SubData['Stage'] == 'Stage 1'))[0]]
    data_m1 = SubData['Half-half Correlation'][np.where((SubData['Maze Type'] == 'Maze 1')&
                                                        (SubData['Training Day'] == day)&
                                                        (SubData['Stage'] == 'Stage 1'))[0]]
    print_estimator(data_op)
    print_estimator(data_m1)
    print(levene(data_op, data_m1))
    print(ttest_ind(data_op, data_m1), cohen_d(data_op, data_m1), end='\n\n')
    
print("Cross Env comparison Stage 2:")
for day in uniq_day:
    print(f" OP vs M1 on Day {day} -------------------------")
    data_op = SubData['Half-half Correlation'][np.where((SubData['Maze Type'] == 'Open Field')&
                                                        (SubData['Training Day'] == day)&
                                                        (SubData['Stage'] == 'Stage 2'))[0]]
    data_m1 = SubData['Half-half Correlation'][np.where((SubData['Maze Type'] == 'Maze 1')&
                                                        (SubData['Training Day'] == day)&
                                                        (SubData['Stage'] == 'Stage 2'))[0]]
    data_m2 = SubData['Half-half Correlation'][np.where((SubData['Maze Type'] == 'Maze 2')&
                                                        (SubData['Training Day'] == day)&
                                                        (SubData['Stage'] == 'Stage 2'))[0]]
    
    print("OP vs M1")
    print_estimator(data_op)
    print_estimator(data_m1)
    print(levene(data_op, data_m1))
    print(ttest_ind(data_op, data_m1), cohen_d(data_op, data_m1))

    print("OP vs M2")
    print_estimator(data_op)
    print_estimator(data_m2)
    print(levene(data_op, data_m2))
    print(ttest_ind(data_op, data_m2), cohen_d(data_op, data_m2))
