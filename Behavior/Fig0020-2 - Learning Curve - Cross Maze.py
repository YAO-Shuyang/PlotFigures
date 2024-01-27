from mylib.statistic_test import *

code_id = '0020 - Learning Curve'
loc = os.path.join(figpath, code_id)
mkdir(loc)


maze_indices = np.where(f_pure_behav['maze_type'] != 0)[0]
if os.path.exists(join(figdata, code_id+'-2.pkl')):
    with open(join(figdata, code_id+'-2.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Correct Rate', 'Pass Number', 'Error Number', 'Pure Guess Correct Rate'], is_behav=True,
                              file_idx=maze_indices,
                              f = f_pure_behav, function = LearningCurveBehavioralScore_Interface, 
                              file_name = code_id+'-2', behavior_paradigm = 'CrossMaze')
    
    with open(join(figdata, code_id+'-2.pkl'), 'wb') as f:
        pickle.dump(Data, f)      
            
uniq_day = ['Day 1', 'Day 2', 'Day 3', 'Day 4',
            'Day 5', 'Day 6', 'Day 7', 'Day 8',
            'Day 9', '>=Day 10']

idx = np.concatenate([np.where(Data['Training Day'] == day)[0] for day in uniq_day])

x = np.unique(Data['Training Day'])
Data['Correct Rate'] = Data['Correct Rate']*100
Data['Pure Guess Correct Rate'] = Data['Pure Guess Correct Rate']*100

print("Stage 1 Maze 1 ---------------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data1 = Data['Correct Rate'][idx]
print(f"Day 1:   Mean: {np.mean(data1)}, STD: {np.std(data1)}, min: {np.min(data1)}, max: {np.max(data1)}, median: {np.median(data1)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 1'))[0]
data2 = Data['Correct Rate'][idx]
print(f"Day 2:   Mean: {np.mean(data2)}, STD: {np.std(data2)}, min: {np.min(data2)}, max: {np.max(data2)}, median: {np.median(data2)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 1'))[0]
data3 = Data['Correct Rate'][idx]
print(f"Day 3:   Mean: {np.mean(data3)}, STD: {np.std(data3)}, min: {np.min(data3)}, max: {np.max(data3)}, median: {np.median(data3)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1'))[0]
data4 = Data['Correct Rate'][idx]
print(f"Day 4:   Mean: {np.mean(data4)}, STD: {np.std(data4)}, min: {np.min(data4)}, max: {np.max(data4)}, median: {np.median(data4)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 1'))[0]
data5 = Data['Correct Rate'][idx]
print(f"Day 5:   Mean: {np.mean(data5)}, STD: {np.std(data5)}, min: {np.min(data5)}, max: {np.max(data5)}, median: {np.median(data5)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 1'))[0]
data6 = Data['Correct Rate'][idx]
print(f"Day 6:   Mean: {np.mean(data6)}, STD: {np.std(data6)}, min: {np.min(data6)}, max: {np.max(data6)}, median: {np.median(data6)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 1'))[0]
data7 = Data['Correct Rate'][idx]
print(f"Day 7:   Mean: {np.mean(data7)}, STD: {np.std(data7)}, min: {np.min(data7)}, max: {np.max(data7)}, median: {np.median(data7)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 1'))[0]
data8 = Data['Correct Rate'][idx]
print(f"Day 8:   Mean: {np.mean(data8)}, STD: {np.std(data8)}, min: {np.min(data8)}, max: {np.max(data8)}, median: {np.median(data8)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1'))[0]
data9 = Data['Correct Rate'][idx]
print(f"Day 9:   Mean: {np.mean(data9)}, STD: {np.std(data9)}, min: {np.min(data9)}, max: {np.max(data9)}, median: {np.median(data9)}")
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1'))[0]
data10 = Data['Correct Rate'][idx]
print(f"Day 10:  Mean: {np.mean(data10)}, STD: {np.std(data10)}, min: {np.min(data10)}, max: {np.max(data10)}, median: {np.median(data10)}", end='\n\n')
print("Day 1 vs Day 10:  ", ttest_ind(data1, data10, alternative='less'))
print("Day 2 vs Day 10:  ", ttest_ind(data2, data10, alternative='less'))
print("Day 3 vs Day 10:  ", ttest_ind(data3, data10, alternative='less'))
print("Day 4 vs Day 10:  ", ttest_ind(data4, data10, alternative='less'))
print("Day 5 vs Day 10:  ", ttest_ind(data5, data10, alternative='less'))
print("Day 6 vs Day 10:  ", ttest_ind(data6, data10, alternative='less'))
print("Day 7 vs Day 10:  ", ttest_ind(data7, data10, alternative='less'))
print("Day 8 vs Day 10:  ", ttest_ind(data8, data10, alternative='less'))
print("Day 9 vs Day 10:  ", ttest_ind(data9, data10, alternative='less'))
print("Day 1 vs Day 3:   ", ttest_ind(data1, data3, alternative='less'), end='\n\n\n')


print("Stage 2 Maze 2 ---------------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
data1 = Data['Correct Rate'][idx]
print(f"Day 1:   Mean: {np.mean(data1)}, STD: {np.std(data1)}, min: {np.min(data1)}, max: {np.max(data1)}, median: {np.median(data1)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 2'))[0]
data2 = Data['Correct Rate'][idx]
print(f"Day 2:   Mean: {np.mean(data2)}, STD: {np.std(data2)}, min: {np.min(data2)}, max: {np.max(data2)}, median: {np.median(data2)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 2'))[0]
data3 = Data['Correct Rate'][idx]
print(f"Day 3:   Mean: {np.mean(data3)}, STD: {np.std(data3)}, min: {np.min(data3)}, max: {np.max(data3)}, median: {np.median(data3)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 2'))[0]
data4 = Data['Correct Rate'][idx]
print(f"Day 4:   Mean: {np.mean(data4)}, STD: {np.std(data4)}, min: {np.min(data4)}, max: {np.max(data4)}, median: {np.median(data4)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 2'))[0]
data5 = Data['Correct Rate'][idx]
print(f"Day 5:   Mean: {np.mean(data5)}, STD: {np.std(data5)}, min: {np.min(data5)}, max: {np.max(data5)}, median: {np.median(data5)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 2'))[0]
data6 = Data['Correct Rate'][idx]
print(f"Day 6:   Mean: {np.mean(data6)}, STD: {np.std(data6)}, min: {np.min(data6)}, max: {np.max(data6)}, median: {np.median(data6)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 2'))[0]
data7 = Data['Correct Rate'][idx]
print(f"Day 7:   Mean: {np.mean(data7)}, STD: {np.std(data7)}, min: {np.min(data7)}, max: {np.max(data7)}, median: {np.median(data7)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 2'))[0]
data8 = Data['Correct Rate'][idx]
print(f"Day 8:   Mean: {np.mean(data8)}, STD: {np.std(data8)}, min: {np.min(data8)}, max: {np.max(data8)}, median: {np.median(data8)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 2'))[0]
data9 = Data['Correct Rate'][idx]
print(f"Day 9:   Mean: {np.mean(data9)}, STD: {np.std(data9)}, min: {np.min(data9)}, max: {np.max(data9)}, median: {np.median(data9)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 2'))[0]
data10 = Data['Correct Rate'][idx]
print(f"Day 10:  Mean: {np.mean(data10)}, STD: {np.std(data10)}, min: {np.min(data10)}, max: {np.max(data10)}, median: {np.median(data10)}", end='\n\n')
print("Day 1 vs Day 10:  ", ttest_ind(data1, data10, alternative='less'))
print("Day 2 vs Day 10:  ", ttest_ind(data2, data10, alternative='less'))
print("Day 3 vs Day 10:  ", ttest_ind(data3, data10, alternative='less'))
print("Day 4 vs Day 10:  ", ttest_ind(data4, data10, alternative='less'))
print("Day 5 vs Day 10:  ", ttest_ind(data5, data10, alternative='less'))
print("Day 6 vs Day 10:  ", ttest_ind(data6, data10, alternative='less'))
print("Day 7 vs Day 10:  ", ttest_ind(data7, data10, alternative='less'))
print("Day 8 vs Day 10:  ", ttest_ind(data8, data10, alternative='less'))
print("Day 9 vs Day 10:  ", ttest_ind(data9, data10, alternative='less'))
print("Day 1 vs Day 3:   ", ttest_ind(data1, data3, alternative='less'), end='\n\n\n')

print("Stage 2 Maze 1 ---------------------------------------------------------------")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data11 = Data['Correct Rate'][idx]
print(f"Day 1:   Mean: {np.mean(data11)}, STD: {np.std(data11)}, min: {np.min(data11)}, max: {np.max(data11)}, median: {np.median(data11)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 2')&(Data['Maze Type'] == 'Maze 1'))[0]
data12 = Data['Correct Rate'][idx]
print(f"Day 2:   Mean: {np.mean(data12)}, STD: {np.std(data12)}, min: {np.min(data12)}, max: {np.max(data12)}, median: {np.median(data12)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 1'))[0]
data13 = Data['Correct Rate'][idx]
print(f"Day 3:   Mean: {np.mean(data13)}, STD: {np.std(data13)}, min: {np.min(data13)}, max: {np.max(data13)}, median: {np.median(data13)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 4')&(Data['Maze Type'] == 'Maze 1'))[0]
data14 = Data['Correct Rate'][idx]
print(f"Day 4:   Mean: {np.mean(data14)}, STD: {np.std(data14)}, min: {np.min(data14)}, max: {np.max(data14)}, median: {np.median(data14)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 5')&(Data['Maze Type'] == 'Maze 1'))[0]
data15 = Data['Correct Rate'][idx]
print(f"Day 5:   Mean: {np.mean(data15)}, STD: {np.std(data15)}, min: {np.min(data15)}, max: {np.max(data15)}, median: {np.median(data15)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 6')&(Data['Maze Type'] == 'Maze 1'))[0]
data16 = Data['Correct Rate'][idx]
print(f"Day 6:   Mean: {np.mean(data16)}, STD: {np.std(data16)}, min: {np.min(data16)}, max: {np.max(data16)}, median: {np.median(data16)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 7')&(Data['Maze Type'] == 'Maze 1'))[0]
data17 = Data['Correct Rate'][idx]
print(f"Day 7:   Mean: {np.mean(data17)}, STD: {np.std(data17)}, min: {np.min(data17)}, max: {np.max(data17)}, median: {np.median(data17)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 8')&(Data['Maze Type'] == 'Maze 1'))[0]
data18 = Data['Correct Rate'][idx]
print(f"Day 8:   Mean: {np.mean(data18)}, STD: {np.std(data18)}, min: {np.min(data18)}, max: {np.max(data18)}, median: {np.median(data18)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1'))[0]
data19 = Data['Correct Rate'][idx]
print(f"Day 9:   Mean: {np.mean(data19)}, STD: {np.std(data19)}, min: {np.min(data19)}, max: {np.max(data19)}, median: {np.median(data19)}")
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1'))[0]
data110 = Data['Correct Rate'][idx]
print(f"Day 10:  Mean: {np.mean(data110)}, STD: {np.std(data110)}, min: {np.min(data110)}, max: {np.max(data110)}, median: {np.median(data110)}", end='\n\n')
print("Day 1 vs Day 10:  ", ttest_ind(data11, data110, alternative='less'))
print("Day 2 vs Day 10:  ", ttest_ind(data12, data110, alternative='less'))
print("Day 3 vs Day 10:  ", ttest_ind(data13, data110, alternative='less'))
print("Day 4 vs Day 10:  ", ttest_ind(data14, data110, alternative='less'))
print("Day 5 vs Day 10:  ", ttest_ind(data15, data110, alternative='less'))
print("Day 6 vs Day 10:  ", ttest_ind(data16, data110, alternative='less'))
print("Day 7 vs Day 10:  ", ttest_ind(data17, data110, alternative='less'))
print("Day 8 vs Day 10:  ", ttest_ind(data18, data110, alternative='less'))
print("Day 9 vs Day 10:  ", ttest_ind(data19, data110, alternative='less'))
print("Day 1 vs Day 3:   ", ttest_ind(data11, data13, alternative='less'), end='\n\n\n')

print("Stage 1: t-test, maze 2 vs maze 1")
print(ttest_ind(data1, data110, alternative='less'))
print(ttest_ind(data2, data12))
print(ttest_ind(data3, data13))
print(ttest_ind(data4, data14))
print(ttest_ind(data5, data15))
print(ttest_ind(data6, data16))
print(ttest_ind(data7, data17))
print(ttest_ind(data8, data18))
print(ttest_ind(data9, data19))
print(ttest_ind(data10, data110), end='\n\n\n')



data2 = Data['Pure Guess Correct Rate'][idx]
print(ttest_1samp(data1, 33.33, alternative='greater'))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,2))
ax1, ax2 = axes[0], axes[1]
ax1 = Clear_Axes(ax1, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(ax2, close_spines=['top', 'right'], ifxticks=True)

colors = sns.color_palette("rocket", 3)
colors = [colors[1], colors[-1]]

idx = np.where(Data['Stage'] == 'Stage 1')
sns.lineplot(
    x=Data['Training Day'][idx],
    y=Data['Correct Rate'][idx],
    hue=Data['Maze Type'][idx],
    ax=ax1,
    marker='o',
    markeredgecolor=None,
    markersize=2,
    palette=colors,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
ax1.set_ylim([0,100])
ax1.set_yticks(np.linspace(0,100,6))

idx = np.where(Data['Stage'] == 'Stage 2')
sns.lineplot(
    x=Data['Training Day'][idx],
    y=Data['Correct Rate'][idx],
    hue=Data['Maze Type'][idx],
    #style=Data['MiceID'][idx],
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
ax2.set_ylim([0,100])
ax2.set_yticks(np.linspace(0,100,6))
plt.savefig(join(loc, 'behavioral_score.png'), dpi=600)
plt.savefig(join(loc, 'behavioral_score.svg'), dpi=600)
plt.close()



idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data_11 = Data['Correct Rate'][idx]
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
data_21 = Data['Correct Rate'][idx]

fig = plt.figure(figsize=(2,3))
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('Novel Maze 1', data_11.shape[0]), np.repeat('Novel Maze 2', data_21.shape[0])]),
    'Time cost': np.concatenate([data_11, data_21]),
}
sns.barplot(
    x='Condition',
    y='Time cost',
    data=compdata,
    palette=colors,
    ax=ax,
    errwidth=0.8,
    capsize=0.3,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x='Condition',
    y='Time cost',
    data=compdata,
    palette=markercolor,
    size=3,
    ax = ax,
    jitter=0.2
)
ax.set_ylim([0,100])
ax.set_xticks([0, 1], ["Novel\nMaze 1", "Novel\nMaze 2"])
ax.set_ylabel('correct-decision rate / %')
plt.tight_layout()
plt.savefig(join(loc, "Comparison of Maze 1 and Maze 2 [correct rate].png"), dpi=600)
plt.savefig(join(loc, "Comparison of Maze 1 and Maze 2 [correct rate].svg"), dpi=600)
plt.close()


# First Session & Second Session
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
data_11 = Data['Correct Rate'][idx]
print_estimator(data_11)
idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
data_21 = Data['Correct Rate'][idx]
print_estimator(data_21)

colors = sns.color_palette("rocket", 4)
colors = [colors[0], colors[-1]]
markercolors = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[0]]

fig = plt.figure(figsize=(1.5,3))
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('Novel Maze 1', data_11.shape[0]), np.repeat('Novel Maze 2', data_21.shape[0])]),
    'Correct Rate': np.concatenate([data_11, data_21]),
}
sns.barplot(
    x='Condition',
    y='Correct Rate',
    data=compdata,
    palette=colors,
    ax=ax,
    errwidth=0.5,
    capsize=0.3,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x='Condition',
    y='Correct Rate',
    data=compdata,
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    jitter=0.2
)
ax.set_ylim([0,100])
#ax.set_yticks(np.linspace(0, 50, 6))
ax.set_xticks([0, 1], ["Novel\nMaze 1", "Novel\nMaze 2"])
ax.set_ylabel('Lap-wise Time of Novel Maze (cm/s)')
plt.tight_layout()
plt.savefig(join(loc, "Comparison of Maze 1 and Maze 2 [S1] Correct Rate.png"), dpi=600)
plt.savefig(join(loc, "Comparison of Maze 1 and Maze 2 [S1] Correct Rate.svg"), dpi=600)
plt.close()
print("First Session:", levene(data_11, data_21))
print(ttest_ind(data_11, data_21, alternative='less', equal_var=False), end='\n\n')



data_m1_s1d1 = Data['Correct Rate'][np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))]
data_m1_s1d3 = Data['Correct Rate'][np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 1'))]
data_m1_s1d10 = Data['Correct Rate'][np.where((Data['Stage'] == 'Stage 1')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1'))]
data_m1_s2d10 = Data['Correct Rate'][np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 1'))]

print(levene(data_m1_s1d1, data_m1_s1d3), ttest_ind(data_m1_s1d1, data_m1_s1d3))
print(levene(data_m1_s1d1, data_m1_s1d10), ttest_ind(data_m1_s1d1, data_m1_s1d10, equal_var=False))
print(levene(data_m1_s1d10, data_m1_s2d10), ttest_ind(data_m1_s1d10, data_m1_s2d10))

data_m2_s2d1 = Data['Correct Rate'][np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))]
data_m2_s2d3 = Data['Correct Rate'][np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == 'Day 3')&(Data['Maze Type'] == 'Maze 2'))]
data_m2_s2d10 = Data['Correct Rate'][np.where((Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10')&(Data['Maze Type'] == 'Maze 2'))]

print(levene(data_m2_s2d1, data_m2_s2d3), ttest_ind(data_m2_s2d1, data_m2_s2d3))
print(levene(data_m2_s2d1, data_m2_s2d10), ttest_ind(data_m2_s2d1, data_m2_s2d10, equal_var=False))