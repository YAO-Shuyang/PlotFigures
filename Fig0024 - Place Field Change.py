from mylib.statistic_test import *

code_id = "0024 - Place Field Change"
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Field Number', 'Path Type'], f = f1, 
                              function = PlaceFieldNumberChange_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
    

idx = np.where((Data['MiceID'] != 11095)&(Data['MiceID'] != 11092))[0]
Data = SubDict(Data, Data.keys(), idx)

idx = np.concatenate([np.where((Data['Maze Type']==m))[0] for m in ['Open Field', 'Maze 1', 'Maze 2']])

data_op = Data['Field Number'][np.where(Data['Maze Type']=='Open Field')[0]]
data_m1 = Data['Field Number'][np.where(Data['Maze Type']=='Maze 1')[0]]
data_m2 = Data['Field Number'][np.where(Data['Maze Type']=='Maze 2')[0]]

print_estimator(data_op)
print_estimator(data_m1)
print_estimator(data_m2)

data_m1_d1 = Data['Field Number'][np.where((Data['Maze Type']=='Maze 1')&(Data['Training Day']=='Day 1')&(Data['Path Type']=='AP')&(Data['Stage'] == 'Stage 1'))[0]]
data_m1_d10 = Data['Field Number'][np.where((Data['Maze Type']=='Maze 1')&(Data['Training Day']=='>=Day 10')&(Data['Path Type']=='AP')&(Data['Stage'] == 'Stage 2'))[0]]
print_estimator(data_m1_d1)
print_estimator(data_m1_d10)
print(levene(data_m1_d1, data_m1_d10))
print(ttest_ind(data_m1_d1, data_m1_d10, alternative='greater'), cohen_d(data_m1_d1, data_m1_d10), end='\n\n')
data_m2_d1 = Data['Field Number'][np.where((Data['Maze Type']=='Maze 2')&(Data['Training Day']=='Day 1')&(Data['Path Type']=='AP')&(Data['Stage'] == 'Stage 2'))[0]]
data_m2_d10 = Data['Field Number'][np.where((Data['Maze Type']=='Maze 2')&(Data['Training Day']=='>=Day 10')&(Data['Path Type']=='AP')&(Data['Stage'] == 'Stage 2'))[0]]
print_estimator(data_m2_d1)
print_estimator(data_m2_d10)
print(levene(data_m2_d1, data_m2_d10))
print(ttest_ind(data_m2_d1, data_m2_d10, alternative='greater'), cohen_d(data_m2_d1, data_m2_d10), end='\n\n')


data_m1_d1 = Data['Field Number'][np.where((Data['Maze Type']=='Maze 1')&(Data['Training Day']=='Day 1')&(Data['Path Type']=='CP')&(Data['Stage'] == 'Stage 1'))[0]]
data_m1_d10 = Data['Field Number'][np.where((Data['Maze Type']=='Maze 1')&(Data['Training Day']=='>=Day 10')&(Data['Path Type']=='CP')&(Data['Stage'] == 'Stage 2'))[0]]
print_estimator(data_m1_d1)
print_estimator(data_m1_d10)
print(levene(data_m1_d1, data_m1_d10))
print(ttest_ind(data_m1_d1, data_m1_d10, alternative='greater'), cohen_d(data_m1_d1, data_m1_d10), end='\n\n')
data_m2_d1 = Data['Field Number'][np.where((Data['Maze Type']=='Maze 2')&(Data['Training Day']=='Day 1')&(Data['Path Type']=='CP')&(Data['Stage'] == 'Stage 2'))[0]]
data_m2_d10 = Data['Field Number'][np.where((Data['Maze Type']=='Maze 2')&(Data['Training Day']=='>=Day 10')&(Data['Path Type']=='CP')&(Data['Stage'] == 'Stage 2'))[0]]
print_estimator(data_m2_d1)
print_estimator(data_m2_d10)
print(levene(data_m2_d1, data_m2_d10))
print(ttest_ind(data_m2_d1, data_m2_d10, alternative='greater'), cohen_d(data_m2_d1, data_m2_d10), end='\n\n')

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
    y='Field Number',
    data=SubData1,
    hue='Maze Type',
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
ax1.set_ylim([0, 15])
ax1.set_yticks(np.linspace(0, 15, 6))

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 1')&(SubData['Path Type'] != 'CP')&(SubData['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1']])
SubData2 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Field Number',
    data=SubData2,
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
    ax=ax2
)
ax2.set_ylim([0, 15])
ax2.set_yticks(np.linspace(0, 15, 6))

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 2')&(SubData['Path Type'] != 'CP')&(SubData['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1', 'Maze 2']])
SubData3 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Field Number',
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
ax3.set_ylim([0, 15])
ax3.set_yticks(np.linspace(0, 15, 6))
plt.tight_layout()
plt.savefig(join(loc, 'Field Number Change.png'), dpi=2400)
plt.savefig(join(loc, 'Field Number Change.svg'), dpi=2400)
plt.close()


fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3))
ax2 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 1')&(SubData['Path Type'] == 'CP')&(SubData['Maze Type'] == m))[0] for m in ['Maze 1']])
SubData2 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Field Number',
    data=SubData2,
    hue='Maze Type',
    palette=markercolors[1:],
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax2
)
idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 1')&(SubData['Path Type'] == 'AP')&(SubData['Maze Type'] == m))[0] for m in ['Maze 1']])
SubData3 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Field Number',
    data=SubData3,
    hue='Maze Type',
    palette=markercolors[2:],
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax2
)
ax2.set_ylim([0, 15])
ax2.set_yticks(np.linspace(0, 15, 6))

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 2')&(SubData['Path Type'] == 'CP')&(SubData['Maze Type'] == m))[0] for m in ['Maze 1']])
SubData3 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Field Number',
    data=SubData3,
    hue='Maze Type',
    palette=markercolors[1:],
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax3
)
idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 2')&(SubData['Path Type'] == 'AP')&(SubData['Maze Type'] == m))[0] for m in ['Maze 1']])
SubData3 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Field Number',
    data=SubData3,
    style='Maze Type',
    palette=markercolors[2:],
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax3
)
ax3.set_ylim([0, 15])
ax3.set_yticks(np.linspace(0, 15, 6))
plt.tight_layout()
plt.savefig(join(loc, 'Field Number Change [CP vs AP Maze 1].png'), dpi=600)
plt.savefig(join(loc, 'Field Number Change [CP vs AP Maze 1].svg'), dpi=600)
plt.close()




fig = plt.figure(figsize=(4,3))
ax3 = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 2')&(SubData['Path Type'] == 'CP')&(SubData['Maze Type'] == m))[0] for m in ['Maze 2']])
SubData3 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Field Number',
    data=SubData3,
    hue='Maze Type',
    palette=markercolors[1:],
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax3
)
idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 2')&(SubData['Path Type'] == 'AP')&(SubData['Maze Type'] == m))[0] for m in ['Maze 2']])
SubData3 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Field Number',
    data=SubData3,
    style='Maze Type',
    palette=markercolors[2:],
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax3
)
ax3.set_ylim([0, 10])
ax3.set_yticks(np.linspace(0, 10, 6))
plt.tight_layout()
plt.savefig(join(loc, 'Field Number Change [CP vs AP Maze 2].png'), dpi=600)
plt.savefig(join(loc, 'Field Number Change [CP vs AP Maze 2].svg'), dpi=600)
plt.close()



data_m1_cp_d1 = SubData['Field Number'][np.where((SubData['Stage'] == 'Stage 1')&(SubData['Path Type'] == 'CP')&(SubData['Maze Type'] == 'Maze 1')&(SubData['Training Day'] == 'Day 1'))[0]]
data_m1_cp_d10 = SubData['Field Number'][np.where((SubData['Stage'] == 'Stage 2')&(SubData['Path Type'] == 'CP')&(SubData['Maze Type'] == 'Maze 1')&(SubData['Training Day'] == '>=Day 10'))[0]]
data_m1_ap_d1 = SubData['Field Number'][np.where((SubData['Stage'] == 'Stage 1')&(SubData['Path Type'] == 'AP')&(SubData['Maze Type'] == 'Maze 1')&(SubData['Training Day'] == 'Day 1'))[0]]
data_m1_ap_d10 = SubData['Field Number'][np.where((SubData['Stage'] == 'Stage 2')&(SubData['Path Type'] == 'AP')&(SubData['Maze Type'] == 'Maze 1')&(SubData['Training Day'] == '>=Day 10'))[0]]

print(levene(data_m1_cp_d1, data_m1_cp_d10))
print(ttest_ind(data_m1_cp_d1, data_m1_cp_d10))
print(levene(data_m1_ap_d1, data_m1_ap_d10))
print(ttest_ind(data_m1_ap_d1, data_m1_ap_d10))
bardata = {
    "Path Type": np.concatenate([np.repeat('CP', data_m1_cp_d1.shape[0] + data_m1_cp_d10.shape[0]), np.repeat('AP', data_m1_ap_d1.shape[0] + data_m1_ap_d10.shape[0])]),
    "Training Day": np.concatenate([np.repeat('Day 1', data_m1_cp_d1.shape[0]), np.repeat('>=Day 10', data_m1_cp_d10.shape[0]), np.repeat('Day 1', data_m1_ap_d1.shape[0]), np.repeat('>=Day 10', data_m1_ap_d10.shape[0])]),
    "Field Number": np.concatenate([data_m1_cp_d1, data_m1_cp_d10, data_m1_ap_d1, data_m1_ap_d10])
}
fig = plt.figure(figsize=(2,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x='Training Day',
    y='Field Number',
    data=bardata,
    hue='Path Type',
    palette=markercolors[1:],
    ax=ax,
    capsize=0.2,
    errwidth=0.5,
    width=0.8
)
plt.savefig(join(loc, 'Field Number barplot.png'), dpi=600)
plt.savefig(join(loc, 'Field Number barplot.svg'), dpi=600)
plt.close()


data_m1_cp_d1 = SubData['Field Number'][np.where((SubData['Stage'] == 'Stage 2')&(SubData['Path Type'] == 'CP')&(SubData['Maze Type'] == 'Maze 2')&(SubData['Training Day'] == 'Day 1'))[0]]
data_m1_cp_d10 = SubData['Field Number'][np.where((SubData['Stage'] == 'Stage 2')&(SubData['Path Type'] == 'CP')&(SubData['Maze Type'] == 'Maze 2')&(SubData['Training Day'] == '>=Day 10'))[0]]
data_m1_ap_d1 = SubData['Field Number'][np.where((SubData['Stage'] == 'Stage 2')&(SubData['Path Type'] == 'AP')&(SubData['Maze Type'] == 'Maze 2')&(SubData['Training Day'] == 'Day 1'))[0]]
data_m1_ap_d10 = SubData['Field Number'][np.where((SubData['Stage'] == 'Stage 2')&(SubData['Path Type'] == 'AP')&(SubData['Maze Type'] == 'Maze 2')&(SubData['Training Day'] == '>=Day 10'))[0]]

print(levene(data_m1_cp_d1, data_m1_cp_d10))
print(ttest_ind(data_m1_cp_d1, data_m1_cp_d10))
print(levene(data_m1_ap_d1, data_m1_ap_d10))
print(ttest_ind(data_m1_ap_d1, data_m1_ap_d10))
bardata = {
    "Path Type": np.concatenate([np.repeat('CP', data_m1_cp_d1.shape[0] + data_m1_cp_d10.shape[0]), np.repeat('AP', data_m1_ap_d1.shape[0] + data_m1_ap_d10.shape[0])]),
    "Training Day": np.concatenate([np.repeat('Day 1', data_m1_cp_d1.shape[0]), np.repeat('>=Day 10', data_m1_cp_d10.shape[0]), np.repeat('Day 1', data_m1_ap_d1.shape[0]), np.repeat('>=Day 10', data_m1_ap_d10.shape[0])]),
    "Field Number": np.concatenate([data_m1_cp_d1, data_m1_cp_d10, data_m1_ap_d1, data_m1_ap_d10])
}
fig = plt.figure(figsize=(2,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x='Training Day',
    y='Field Number',
    data=bardata,
    hue='Path Type',
    palette=markercolors[1:],
    ax=ax,
    capsize=0.2,
    errwidth=0.5,
    width=0.8
)
plt.savefig(join(loc, 'Field Number barplot [Maze 2].png'), dpi=600)
plt.savefig(join(loc, 'Field Number barplot [Maze 2].svg'), dpi=600)
plt.close()