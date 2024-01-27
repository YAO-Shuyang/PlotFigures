from mylib.statistic_test import *

code_id = '0058 - Occupation Time on Correct&Incorrect Track'
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Occupation Time Percentage', 'Ratio', 'Bin-wise Mean Time', 'Bin-wise Ratio', 'Path Type'], 
                              f = f_pure_behav, function = OccupationTimeDiffPath_Interface, is_behav=True,
                              func_kwgs={'is_placecell':True}, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

idx = np.where((Data['MiceID'] != 11092)&(Data['MiceID'] != 11094))[0]
Data = SubDict(Data, Data.keys(), idx)

Data['hue'] = np.array([Data['Maze Type'][i]+' - '+Data['Path Type'][i] for i in range(len(Data['Maze Type']))])

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,2))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
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
    y='Bin-wise Mean Time',
    data=SubData,
    hue='hue',
    palette=colors,
    capsize=0.2,
    errwidth=0.5,
    errcolor='black',
    ax=ax1
)
ax1.set_ylim([0,40])

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.barplot(
    x='Training Day',
    y='Bin-wise Mean Time',
    data=SubData,
    hue='hue',
    palette=colors,
    capsize=0.1,
    errwidth=0.5,
    errcolor='black',
    ax=ax2
)
ax2.set_ylim([0,15])
plt.savefig(join(loc, 'Bin-wise Mean Time.png'), dpi=600)
plt.savefig(join(loc, 'Bin-wise Mean Time.svg'), dpi=600)
plt.close()


fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,2))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

idx = np.concatenate([np.where(Data['Training Day'] == session)[0] for session in s_ticks] + [np.where(Data['Training Day'] == day)[0] for day in x_ticks])
Data = SubDict(Data, Data.keys(), idx)

colors = sns.color_palette("rocket", 4)
markercolors = sns.color_palette("Blues", 4)

Data['Bin-wise Ratio'] = np.array([1/i for i in Data['Bin-wise Ratio']], dtype=np.float64)
stage1_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage1_indices)
stage1_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1']])
SubData = SubDict(SubData, SubData.keys(), idx=stage1_indices)
sns.lineplot(
    x='Training Day',
    y='Bin-wise Ratio',
    data=SubData,
    hue='Maze Type',
    palette=colors,
    err_style='bars',
    ax=ax1, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    #err_kws={'elinewidth':1, 'capthick':1, 'capsize':2},
    linewidth=0.5,
    estimator=np.median,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
)
ax1.set_ylim([0,2])

stage2_indices = np.concatenate([np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == day))[0] for day in x_ticks])
SubData = SubDict(Data, Data.keys(), idx=stage2_indices)
stage2_indices = np.concatenate([np.where(SubData['Maze Type'] == maze)[0] for maze in ['Open Field', 'Maze 1', 'Maze 2']])
SubData = SubDict(SubData, SubData.keys(), idx=stage2_indices)
sns.lineplot(
    x='Training Day',
    y='Bin-wise Ratio',
    data=SubData,
    hue='Maze Type',
    palette=colors,
    err_style='bars',
    ax=ax2, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    #err_kws={'elinewidth':1, 'capthick':1, 'capsize':2},
    linewidth=0.5,
    estimator=np.median,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
)
ax2.set_ylim([0,2])
plt.savefig(join(loc, 'Bin-wise Ratio.png'), dpi=600)
plt.savefig(join(loc, 'Bin-wise Ratio.svg'), dpi=600)
plt.close()


print("Stage 1 Maze 1 ---------------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data1_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data1_cp)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data1_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data1_ip)

idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data2_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data2_cp)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data2_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data2_ip)

idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data3_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data3_cp)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data3_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data3_ip)

idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data4_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data4_cp)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data4_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data4_ip)

idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data5_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data5_cp)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data5_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data5_ip)

idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data6_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data6_cp)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data6_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data6_ip)

idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data7_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data7_cp)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data7_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data7_ip)

idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data8_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data8_cp)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data8_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data8_ip)

idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data9_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data9_cp)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data9_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data9_ip)

idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data10_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data10_cp)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data10_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data10_ip)

print("Day 1  ", ttest_1samp(data1_cp - data1_ip, 0, alternative='greater'))
print("Day 2  ", ttest_1samp(data2_cp - data2_ip, 0, alternative='greater'))
print("Day 3  ", ttest_1samp(data3_cp - data3_ip, 0, alternative='greater'))
print("Day 4  ", ttest_1samp(data4_cp - data4_ip, 0, alternative='greater'))
print("Day 5  ", ttest_1samp(data5_cp - data5_ip, 0, alternative='greater'))
print("Day 6  ", ttest_1samp(data6_cp - data6_ip, 0, alternative='greater'))
print("Day 7  ", ttest_1samp(data7_cp - data7_ip, 0, alternative='greater'))
print("Day 8  ", ttest_1samp(data8_cp - data8_ip, 0, alternative='greater'))
print("Day 9  ", ttest_1samp(data9_cp - data9_ip, 0, alternative='greater'))
print("Day 10 ", ttest_1samp(data10_cp - data10_ip, 0, alternative='greater'), end='\n\n\n')




print("Stage 2 Maze 1 ---------------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data1_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data1_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data1_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data1_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data2_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data2_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data2_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data2_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data3_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data3_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data3_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data3_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data4_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data4_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data4_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data4_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data5_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data5_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data5_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data5_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data6_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data6_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data6_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data6_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data7_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data7_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data7_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data7_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data8_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data8_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data8_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data8_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data9_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data9_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data9_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data9_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'CP'))[0]
data10_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data10_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1')&(Data['Path Type'] == 'IP'))[0]
data10_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data10_ip)

print("Day 1  ", ttest_1samp(data1_cp - data1_ip, 0, alternative='greater'))
print("Day 2  ", ttest_1samp(data2_cp - data2_ip, 0, alternative='greater'))
print("Day 3  ", ttest_1samp(data3_cp - data3_ip, 0, alternative='greater'))
print("Day 4  ", ttest_1samp(data4_cp - data4_ip, 0, alternative='greater'))
print("Day 5  ", ttest_1samp(data5_cp - data5_ip, 0, alternative='greater'))
print("Day 6  ", ttest_1samp(data6_cp - data6_ip, 0, alternative='greater'))
print("Day 7  ", ttest_1samp(data7_cp - data7_ip, 0, alternative='greater'))
print("Day 8  ", ttest_1samp(data8_cp - data8_ip, 0, alternative='greater'))
print("Day 9  ", ttest_1samp(data9_cp - data9_ip, 0, alternative='greater'))
print("Day 10 ", ttest_1samp(data10_cp - data10_ip, 0, alternative='greater'), end='\n\n\n')



print("Stage 2 Maze 2 ---------------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'CP'))[0]
data1_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data1_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'IP'))[0]
data1_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data1_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'CP'))[0]
data2_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data2_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'IP'))[0]
data2_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data2_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'CP'))[0]
data3_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data3_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'IP'))[0]
data3_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data3_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'CP'))[0]
data4_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data4_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'IP'))[0]
data4_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data4_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'CP'))[0]
data5_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data5_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'IP'))[0]
data5_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data5_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'CP'))[0]
data6_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data6_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'IP'))[0]
data6_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data6_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'CP'))[0]
data7_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data7_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'IP'))[0]
data7_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data7_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'CP'))[0]
data8_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data8_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'IP'))[0]
data8_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data8_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'CP'))[0]
data9_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data9_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'IP'))[0]
data9_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data9_ip)

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'CP'))[0]
data10_cp = Data['Bin-wise Mean Time'][idx]
print_estimator(data10_cp)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 2')&(Data['Path Type'] == 'IP'))[0]
data10_ip = Data['Bin-wise Mean Time'][idx]
print_estimator(data10_ip)

print("Day 1  ", ttest_1samp(data1_cp - data1_ip, 0, alternative='greater'))
print("Day 2  ", ttest_1samp(data2_cp - data2_ip, 0, alternative='greater'))
print("Day 3  ", ttest_1samp(data3_cp - data3_ip, 0, alternative='greater'))
print("Day 4  ", ttest_1samp(data4_cp - data4_ip, 0, alternative='greater'))
print("Day 5  ", ttest_1samp(data5_cp - data5_ip, 0, alternative='greater'))
print("Day 6  ", ttest_1samp(data6_cp - data6_ip, 0, alternative='greater'))
print("Day 7  ", ttest_1samp(data7_cp - data7_ip, 0, alternative='greater'))
print("Day 8  ", ttest_1samp(data8_cp - data8_ip, 0, alternative='greater'))
print("Day 9  ", ttest_1samp(data9_cp - data9_ip, 0, alternative='greater'))
print("Day 10 ", ttest_1samp(data10_cp - data10_ip, 0, alternative='greater'), end='\n\n\n')