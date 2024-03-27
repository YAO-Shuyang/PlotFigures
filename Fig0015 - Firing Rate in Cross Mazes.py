# Fig0015 is about firing rate, either peak rate or mean rate.
# Cross Maze

# Firing rate of rate map 12 x 12.
# Fig0015-1: Peak rate vs training day.
# Fig0015-2: Mean rate vs training day.
# Fig0015-3: Peak rate divided by path type, maze 1.
# Fig0015-4: Peak rate divided by path type, maze 2.
# Fig0015-5: Mean rate divided by path type, maze 1.
# Fig0015-6: Mean rate divided by path type, maze 2.

# Firing rate of smoothed rate map 48 x 48.
# Fig0015-7 ~ Fig0015-12: rate map corresponding to Fig0015-1~Fig0015-6

# Firing rate of old map, but cells has been divided as main field on correct path cells and main fields on incorrect path cells.
# Fig0015-13 ~ Fig0015-18: rate map corresponding to Fig0015-1~Fig0015-6
# Fig0015-13 and Fig0015-14 is the same as Fig0015-1 and Fig0015-2, respectively.

from mylib.statistic_test import *

code_id = '0015 - Firing Rate - Cross Maze'
loc = join(figpath, code_id)
mkdir(loc)

"""
Peak rate of place fields
if os.path.exists(os.path.join(figdata,code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['peak_rate'], 
                              f = f1, function = FieldPeakRateStatistic_Interface, 
                              file_name = code_id+' [field peak rate].pkl', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata,code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

idx = np.where((Data['MiceID'] != 11095)&(Data['MiceID'] != 11092))[0]
Data = SubDict(Data, Data.keys(), idx)

idx = [np.where((Data['MiceID'] == 10209)&(Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 2')&(Data['Training Day'] == 'Day 1'))[0],
       np.where((Data['MiceID'] == 10212)&(Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 2')&(Data['Training Day'] == 'Day 1'))[0],
       np.where((Data['MiceID'] == 10224)&(Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 2')&(Data['Training Day'] == 'Day 1'))[0],
       np.where((Data['MiceID'] == 10227)&(Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 2')&(Data['Training Day'] == 'Day 1'))[0]]
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,6))
mice = [10209, 10212, 10224, 10227]
for i, ax in enumerate([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]):

    ax = Clear_Axes(ax, ifxticks=True,ifyticks=True, close_spines=['top', 'right'])
    x = ax.hist(Data['peak_rate'][idx[i]], bins=100, range=(0, 6), density=True)[0]
    ax.set_ylabel(str(mice[i]))
    alpha, c, beta = gamma.fit(Data['peak_rate'][idx[i]])
    x = np.linspace(0, 6, 10000)
    y = gamma.pdf(x, alpha, loc=c, scale=beta)
    ax.plot(x, y)
    print(mice[i])
    print("    ", alpha, c, beta)
plt.show()
"""               

if os.path.exists(os.path.join(figdata,code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['peak_rate','mean_rate'], 
                              func_kwgs={'is_placecell':True},
                              f = f1, function = FiringRateProcess_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata,code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)


from scipy.stats import zscore

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
    y='mean_rate',
    data=SubData1,
    hue='Maze Type',
    palette=colors,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax1
)
ax1.set_ylim([0,1])
ax1.set_yticks(np.linspace(0,1,6))

idx = np.where(SubData['Stage'] == 'Stage 1')[0]
SubData2 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='mean_rate',
    data=SubData2,
    hue='Maze Type',
    palette=colors,
    err_style='bars',
    ax=ax2, 
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax2.set_ylim([0,1])
ax2.set_yticks(np.linspace(0,1,6))

idx = np.where(SubData['Stage'] == 'Stage 2')[0]
SubData3 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='mean_rate',
    data=SubData3,
    hue='Maze Type',
    palette=colors,
    err_style='bars',
    ax=ax3, 
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax3.set_ylim([0,1])
ax3.set_yticks(np.linspace(0,1,6))
plt.tight_layout()
plt.savefig(join(loc, 'mean rate.png'), dpi=600)
plt.savefig(join(loc, 'mean rate.svg'), dpi=600)
plt.close()


fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,3))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[2], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.concatenate([np.where((Data['Stage'] == 'PRE')&(Data['Training Day'] == s))[0] for s in uniq_s])
SubData1 = SubDict(Data, Data.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='peak_rate',
    data=SubData1,
    hue='Maze Type',
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
ax1.set_ylim([15,25])
ax1.set_yticks(np.linspace(15,25,6))

idx = np.where(SubData['Stage'] == 'Stage 1')[0]
SubData2 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='peak_rate',
    data=SubData2,
    hue='Maze Type',
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
ax2.set_ylim([15,25])
ax2.set_yticks(np.linspace(15,25,6))

idx = np.where(SubData['Stage'] == 'Stage 2')[0]
SubData3 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='peak_rate',
    data=SubData3,
    hue='Maze Type',
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
ax3.set_ylim([15,25])
ax3.set_yticks(np.linspace(15,25,6))
plt.tight_layout()
plt.savefig(join(loc, 'peak rate.png'), dpi=600)
plt.savefig(join(loc, 'peak rate.svg'), dpi=600)
plt.close()

data_op_d1 = SubData['mean_rate'][np.where((SubData['Maze Type'] == 'Open Field')&(SubData['Training Day'] == 'Day 1')&(SubData['Stage'] == 'Stage 1'))[0]]
data_op_d1 = (data_op_d1[np.arange(0, len(data_op_d1), 2)] + data_op_d1[np.arange(1, len(data_op_d1), 2)])/2
data_m1_d1 = SubData['mean_rate'][np.where((SubData['Maze Type'] == 'Maze 1')&(SubData['Training Day'] == 'Day 1')&(SubData['Stage'] == 'Stage 1'))[0]]
data_m2_d1 = SubData['mean_rate'][np.where((SubData['Maze Type'] == 'Maze 2')&(SubData['Training Day'] == 'Day 1')&(SubData['Stage'] == 'Stage 2'))[0]]

data_op_d10 = SubData['mean_rate'][np.where((SubData['Maze Type'] == 'Open Field')&(SubData['Training Day'] == '>=Day 10')&(SubData['Stage'] == 'Stage 2'))[0]]
data_op_d10 = (data_op_d10[np.arange(0, len(data_op_d10), 2)] + data_op_d10[np.arange(1, len(data_op_d10), 2)])/2
data_m1_d10 = SubData['mean_rate'][np.where((SubData['Maze Type'] == 'Maze 1')&(SubData['Training Day'] == '>=Day 10')&(SubData['Stage'] == 'Stage 2'))[0]]
data_m2_d10 = SubData['mean_rate'][np.where((SubData['Maze Type'] == 'Maze 2')&(SubData['Training Day'] == '>=Day 10')&(SubData['Stage'] == 'Stage 2'))[0]]

print("Open Field: Day 1 vs Day 10")
print_estimator(data_op_d1)
print_estimator(data_op_d10[[6, 7, 14, 15]])
print(ttest_rel(data_op_d1, data_op_d10[[6, 7, 14, 15]]), end='\n\n')
print("Maze 1: Day 1 vs Day 10")
print_estimator(data_m1_d1)
print_estimator(data_m1_d10[[6, 7, 14, 15]])
print(ttest_rel(data_m1_d1, data_m1_d10[[6, 7, 14, 15]]), end='\n\n')
print("Maze 2: Day 1 vs Day 10")
print_estimator(data_m2_d1)
print_estimator(data_m2_d10[[6, 7, 14, 15]])
print(ttest_rel(data_m2_d1, data_m2_d10[[6, 7, 14, 15]]), end='\n\n')

print("Cross Env comparison Stage 1:")
for day in uniq_day:
    print(f" OP vs M1 on Day {day} -------------------------")
    data_op = SubData['mean_rate'][np.where((SubData['Maze Type'] == 'Open Field')&
                                                        (SubData['Training Day'] == day)&
                                                        (SubData['Stage'] == 'Stage 1'))[0]]
    data_op = (data_op[np.arange(0, len(data_op), 2)] + data_op[np.arange(1, len(data_op), 2)])/2
    data_m1 = SubData['mean_rate'][np.where((SubData['Maze Type'] == 'Maze 1')&
                                                        (SubData['Training Day'] == day)&
                                                        (SubData['Stage'] == 'Stage 1'))[0]]
    try:
        print("     OP vs M1", ttest_rel(data_op, data_m1))
    except:
        print()
    
print("Cross Env comparison Stage 2:")
for day in uniq_day:
    print(f" OP vs M1 on Day {day} -------------------------")
    data_op = SubData['mean_rate'][np.where((SubData['Maze Type'] == 'Open Field')&
                                                        (SubData['Training Day'] == day)&
                                                        (SubData['Stage'] == 'Stage 2'))[0]]
    data_op = (data_op[np.arange(0, len(data_op), 2)] + data_op[np.arange(1, len(data_op), 2)])/2
    data_m1 = SubData['mean_rate'][np.where((SubData['Maze Type'] == 'Maze 1')&
                                                        (SubData['Training Day'] == day)&
                                                        (SubData['Stage'] == 'Stage 2'))[0]]
    data_m2 = SubData['mean_rate'][np.where((SubData['Maze Type'] == 'Maze 2')&
                                                        (SubData['Training Day'] == day)&
                                                        (SubData['Stage'] == 'Stage 2'))[0]]
    
    print("     OP vs M1", ttest_rel(data_op, data_m1)[1]*2)
    print("     OP vs M2", ttest_rel(data_op, data_m2)[1]*2, end='\n\n')
print()

compdata = {
    "mean_rate": np.concatenate([data_op_d1, data_m1_d1, data_m2_d1, data_op_d10[[6, 7, 14, 15]], data_m1_d10[[6, 7, 14, 15]], data_m2_d10[[6, 7, 14, 15]]]),
    "Maze Type": np.array(['Open Field']*len(data_op_d1) + ['Maze 1']*len(data_m1_d1) + ['Maze 2']*len(data_m2_d1) + ['Open Field']*len(data_op_d10[[6, 7, 14, 15]]) + ['Maze 1']*len(data_m1_d10[[6, 7, 14, 15]]) + ['Maze 2']*len(data_m2_d10[[6, 7, 14, 15]])),
    "Session": np.concatenate([np.repeat("First", len(data_op_d1)+len(data_m1_d1)+len(data_m2_d1)), np.repeat("Last", len(data_op_d10[[6, 7, 14, 15]])+len(data_m1_d10[[6, 7, 14, 15]])+len(data_m2_d10[[6, 7, 14, 15]]))]), 
    "MiceID": np.array([10209, 10212, 10224, 10227] * 6)
}
print(compdata)
fig = plt.figure(figsize=(3,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x='Maze Type',
    y='mean_rate',
    hue='Session',
    data=compdata,
    palette=['#003366', '#0099CC'],
    ax=ax,
    errcolor='black',
    errwidth=0.5,
    capsize=0.1,
    width=0.8
)

sns.stripplot(
    x='Maze Type',
    y='mean_rate',
    hue='MiceID',
    data=compdata,
    palette=['#F2E8D4', '#8E9F85', '#C3AED6', '#A7D8DE'],
    edgecolor='black',
    size=4,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.15
)
ax.set_ylim([0, 0.8])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
plt.savefig(join(loc, 'Comparison of Sessions.png'), dpi=2400)
plt.savefig(join(loc, 'Comparison of Sessions.svg'), dpi=2400)
plt.close()