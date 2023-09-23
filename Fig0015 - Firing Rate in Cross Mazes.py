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

if os.path.exists(os.path.join(figdata,code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['peak_rate','mean_rate'], 
                              func_kwgs={'is_placecell':True},
                              f = f1, function = FiringRateProcess_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata,code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)


from scipy.stats import zscore

index6s = np.where((Data['MiceID'] == '11095')|(Data['MiceID'] == '11092'))[0]
index6f = np.where((Data['MiceID'] == '10212')|(Data['MiceID'] == '10209'))[0]

Data['Mean Rate Z-score'] = np.zeros_like(Data['mean_rate'], dtype=np.int64)
Data['Mean Rate Z-score'][index6f] = zscore(Data['mean_rate'][index6f])
Data['Mean Rate Z-score'][index6s] = zscore(Data['mean_rate'][index6s])
SubData = SubDict(Data, Data.keys(), np.concatenate([index6s, index6f]))

# Mean Rate
uniq_day = ['Day 1', 'Day 2', 'Day 3', 'Day 4',
            'Day 5', 'Day 6', 'Day 7', 'Day 8',
            'Day 9', '>=Day 10']

idx = np.concatenate([np.where(SubData['Training Day'] == day)[0] for day in uniq_day])
SubData = SubDict(SubData, SubData.keys(), idx)

colors = sns.color_palette("rocket", 3)

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3))
ax1, ax2 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True), Clear_Axes(axes[1], close_spines=['top', 'right', 'left'], ifxticks=True)
idx = np.where(SubData['Stage'] == 'Stage 1')[0]
SubData1 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='mean_rate',
    data=SubData1,
    hue='Maze Type',
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=4,
    legend=False,
    err_style='bars',
    err_kws={'capsize':3},
    linewidth=1,
    ax=ax1
)

idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Training Day'] != '>=Day 10'))[0]
SubData2 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='mean_rate',
    data=SubData2,
    hue='Maze Type',
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=4,
    err_style='bars',
    err_kws={'capsize':3},
    linewidth=1,
    ax=ax2
)
ax1.set_ylim([0,1])
ax1.set_yticks(np.linspace(0,1,6))
ax2.set_ylim([0,1])
plt.tight_layout()
plt.savefig(join(loc, 'mean rate.png'), dpi=600)
plt.savefig(join(loc, 'mean rate.svg'), dpi=600)
plt.close()

'''
# Fig0015-1: Peak rate vs training day. ---------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize = (8,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette('rocket', 3)
 
plt.savefig(os.path.join(p,'[lineplot] Peak Rate.svg'), dpi = 2400)
plt.savefig(os.path.join(p,'[lineplot] Peak Rate.png'), dpi = 2400)
plt.close()

fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette('rocket', 3)
sns.barplot(x = 'Training Day',y = 'peak_rate', data = Data, hue = 'Maze Type', ax = ax, palette=colors,
            errcolor='black', errwidth=1, capsize=0.1)
plt.tight_layout()
plt.savefig(os.path.join(p,'[barplot] Peak Rate.svg'), dpi = 2400)
plt.savefig(os.path.join(p,'[barplot] Peak Rate.png'), dpi = 2400)
plt.close()


for d in ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9']:
    op_idx = np.where((Data['Training Day'] == d)&(Data['Maze Type'] == 'Open Field'))[0]
    m1_idx = np.where((Data['Training Day'] == d)&(Data['Maze Type'] == 'Maze 1'))[0]
    m2_idx = np.where((Data['Training Day'] == d)&(Data['Maze Type'] == 'Maze 2'))[0]
    
    print(d, 'Peak Rate ----------------------------------------')
    print("OP: M1    ", ttest_ind(Data['peak_rate'][op_idx], Data['peak_rate'][m1_idx]))
    print("OP: M2    ", ttest_ind(Data['peak_rate'][op_idx], Data['peak_rate'][m2_idx]))
    print("M1: M2    ", ttest_ind(Data['peak_rate'][m1_idx], Data['peak_rate'][m2_idx]), end='\n\n\n')


fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette('rocket', 3)
sns.lineplot(x = 'Training Day',y = 'mean_rate', data = Data, hue = 'Maze Type', ax = ax, palette=colors,
             err_style = 'bars', err_kws = {'elinewidth':1, 'capsize':3, 'capthick':1})
plt.tight_layout()
plt.savefig(os.path.join(p,'[lineplot] Mean Rate.svg'), dpi = 2400)
plt.savefig(os.path.join(p,'[lineplot] Mean Rate.png'), dpi = 2400)
plt.close()

fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette('rocket', 3)
sns.barplot(x = 'Training Day',y = 'mean_rate', data = Data, hue = 'Maze Type', ax = ax, palette=colors,
            errcolor='black', errwidth=1, capsize=0.1)
plt.tight_layout()
plt.savefig(os.path.join(p,'[barplot] Mean Rate.svg'), dpi = 2400)
plt.savefig(os.path.join(p,'[barplot] Mean Rate.png'), dpi = 2400)
plt.close()

for d in ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9']:
    op_idx = np.where((Data['Training Day'] == d)&(Data['Maze Type'] == 'Open Field'))[0]
    m1_idx = np.where((Data['Training Day'] == d)&(Data['Maze Type'] == 'Maze 1'))[0]
    m2_idx = np.where((Data['Training Day'] == d)&(Data['Maze Type'] == 'Maze 2'))[0]
    print(d, 'Peak Rate ----------------------------------------')
    print("OP: M1    ", ttest_ind(Data['mean_rate'][op_idx], Data['mean_rate'][m1_idx]))
    print("OP: M2    ", ttest_ind(Data['mean_rate'][op_idx], Data['mean_rate'][m2_idx]))
    print("M1: M2    ", ttest_ind(Data['mean_rate'][m1_idx], Data['mean_rate'][m2_idx]), end='\n\n\n')


print(" ========================================================================================= ")
# cross day significance test.
op_d1_idx = np.where((Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Open Field'))[0]
m1_d1_idx = np.where((Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
m2_d1_idx = np.where((Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
op_d9_idx = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Open Field'))[0]
m1_d9_idx = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1'))[0]
m2_d9_idx = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 2'))[0]

print("Day 1 vs Day 9: Peak Rate")
print(ttest_ind(Data['peak_rate'][op_d1_idx], Data['peak_rate'][op_d9_idx]))
print(ttest_ind(Data['peak_rate'][m1_d1_idx], Data['peak_rate'][m1_d9_idx]))
print(ttest_ind(Data['peak_rate'][m2_d1_idx], Data['peak_rate'][m2_d9_idx]))

print("Day 1 vs Day 9: Mean Rate")
print(ttest_ind(Data['mean_rate'][op_d1_idx], Data['mean_rate'][op_d9_idx]))
print(ttest_ind(Data['mean_rate'][m1_d1_idx], Data['mean_rate'][m1_d9_idx]))
print(ttest_ind(Data['mean_rate'][m2_d1_idx], Data['mean_rate'][m2_d9_idx]))



# Mice 11095 Only
idx = np.where(Data['MiceID'] == '11095')[0]
SubData = DivideData(Data, idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = 'Training Day',y = 'peak_rate', data = SubData, hue = 'Maze Type', ax = ax, err_style = 'bars', err_kws = {'elinewidth':1, 'capsize':3, 'capthick':1})
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze Type', loc = 'upper center', fontsize = 8, title_fontsize = 9, ncol = 3)
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Peak Rate / Hz', fontsize = fs)
ax.set_title('Peak Rate #11095', fontsize = fs)
ax.axhline(8, ls = ':', color = "orange")
ax.axhline(5, ls = ':', color = "limegreen")
ax.set_yticks([0,2,4,6,8,10,12])
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9,10], labels = ['P1','P2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.xaxis.set_major_locator(x)
ax.axis([-0.5,10.5,0,12])
plt.tight_layout()
plt.savefig(os.path.join(p,code_id+'-1-Peak Rate-CrossMaze-11095.svg'), dpi = 600)
plt.savefig(os.path.join(p,code_id+'-1-Peak Rate-CrossMaze-11095.png'), dpi = 600)
plt.close()

# T test for increasing
day1_idx_maze1 = np.where((Data['Training Day'] == 'Pre 2')&(Data['Maze Type'] == 'Maze 1'))[0]
day9_idx_maze1 = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1'))[0]
_, pvalue1 = scipy.stats.ttest_ind(Data['mean_rate'][day1_idx_maze1], Data['mean_rate'][day9_idx_maze1], nan_policy = 'omit')

day1_idx_maze2 = np.where((Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
day9_idx_maze2 = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 2'))[0]
_, pvalue2 = scipy.stats.ttest_ind(Data['mean_rate'][day1_idx_maze2], Data['mean_rate'][day9_idx_maze2], nan_policy = 'omit')

# open field maintain t test
day1_idx_op = np.where((Data['Training Day'] == 'Pre 1')&(Data['Maze Type'] == 'Open Field'))[0]
day9_idx_op = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Open Field'))[0]
_, pvalue0 = scipy.stats.ttest_ind(Data['mean_rate'][day1_idx_op], Data['mean_rate'][day9_idx_op], nan_policy = 'omit')

print('Open Field:',pvalue0)
print('Maze 1:',pvalue1)
print('Maze 2',pvalue2)

# Mice 11095 Only
idx = np.where(Data['MiceID'] == '11092')[0]
SubData = DivideData(Data, idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = 'Training Day',y = 'peak_rate', data = SubData, hue = 'Maze Type', ax = ax, err_style = 'bars', err_kws = {'elinewidth':1, 'capsize':3, 'capthick':1})
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze Type', loc = 'upper center', fontsize = 8, title_fontsize = 9, ncol = 3)
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Peak Rate / Hz', fontsize = fs)
ax.set_title('Peak Rate #11092', fontsize = fs)
ax.axhline(6, ls = ':', color = "orange")
ax.axhline(4, ls = ':', color = "limegreen")
ax.set_yticks(np.linspace(0,8,5))
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9,10], labels = ['P1','P2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.xaxis.set_major_locator(x)
ax.axis([-0.5,10.5,0,8])
plt.tight_layout()
plt.savefig(os.path.join(p,code_id+'-1-Peak Rate-CrossMaze-11092.svg'), dpi = 600)
plt.savefig(os.path.join(p,code_id+'-1-Peak Rate-CrossMaze-11092.png'), dpi = 600)
plt.close()

# Fig0015-2: Mean rate vs training day. ---------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = 'Training Day',y = 'mean_rate', data = Data, hue = 'Maze Type', ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze Type', loc = 'upper left')
ax.axis([-0.5,8.5,0,1])
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9,10], labels = ['P1','P2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.set_yticks(np.linspace(0,1,6))
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Mean Rate / Hz', fontsize = fs)
ax.set_title('Cross Maze('+str(cell_number)+' cells)', fontsize = fs)
plt.tight_layout()
plt.savefig(os.path.join(p,code_id+'-2-Mean Rate-CrossMaze.svg'), dpi = 600)
plt.savefig(os.path.join(p,code_id+'-2-Mean Rate-CrossMaze.png'), dpi = 600)
plt.close()



# Fig0015-3: Peak rate divided by path type, maze 1 ----------------------------------------------------------------------------------
idx = np.where(Data['Maze Type'] == 'Maze 1')[0]
a = len(idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = np.concatenate([Data['Training Day'][idx], Data['Training Day'][idx]]),
             y = np.concatenate([Data['peak_rate'][idx], Data['peak_rate_on_path'][idx]]),
             hue = np.concatenate([np.repeat('Entire Path', len(Data['peak_rate'][idx])),
                                   Data['path_type'][idx]]), ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze 1',
          bbox_to_anchor = (0.5,1.1))
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Peak Rate / Hz', fontsize = fs)
#ax.set_title('Simple Maze('+str(a)+' cells), Maze 1', fontsize = fs)
ax.axis([-0.5,8.5,0,8])
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9], labels = ['P2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.set_yticks([0,2,4,6,8])
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(p,code_id+'-3-Peak Rate divided by path-CrossMaze-Maze1.svg'), dpi = 600)
plt.savefig(os.path.join(p,code_id+'-3-Peak Rate divided by path-CrossMaze-Maze1.png'), dpi = 600)
plt.close()




# Fig0015-4: Peak rate divided by path type, maze 2. ------------------------------------------------------------------------------------
idx = np.where(Data['Maze Type'] == 'Maze 2')[0]
a = len(idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = np.concatenate([Data['Training Day'][idx], Data['Training Day'][idx]]),
             y = np.concatenate([Data['peak_rate'][idx], Data['peak_rate_on_path'][idx]]),
             hue = np.concatenate([np.repeat('Entire Path', len(Data['peak_rate'][idx])),
                                   Data['path_type'][idx]]), ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze 2', loc = 'upper left')
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8], labels = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.set_ylabel('Peak Rate / Hz', fontsize = fs)
#ax.set_title('Simple Maze('+str(a)+' cells), Maze 2', fontsize = fs)
ax.axis([-0.5,8.5,0,8])
ax.set_yticks([0,2,4,6,8])
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(p,code_id+'-4-Peak Rate divided by path-CrossMaze-Maze2.svg'), dpi = 600)
plt.savefig(os.path.join(p,code_id+'-4-Peak Rate divided by path-CrossMaze-Maze2.png'), dpi = 600)
plt.close()




# Fig0015-5: Mean rate divided by path type, maze 1. ----------------------------------------------------------------------------------
idx = np.where(Data['Maze Type'] == 'Maze 1')[0]
a = len(idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = np.concatenate([Data['Training Day'][idx], Data['Training Day'][idx]]),
             y = np.concatenate([Data['mean_rate'][idx], Data['mean_rate_on_path'][idx]]),
             hue = np.concatenate([np.repeat('Entire Path', len(Data['mean_rate'][idx])),
                                   Data['path_type'][idx]]), ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze 1', loc = 'upper left')
ax.xaxis.set_major_locator(x)
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9], labels = ['P2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Mean Rate / Hz', fontsize = fs)
#ax.set_title('Simple Maze('+str(a)+' cells), Maze 1', fontsize = fs)
ax.axis([-0.5,8.5,0,1])
ax.set_yticks(np.linspace(0,1,6))
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(p,code_id+'-5-Mean Rate divided by path-CrossMaze-Maze1.svg'), dpi = 600)
plt.savefig(os.path.join(p,code_id+'-5-Mean Rate divided by path-CrossMaze-Maze1.png'), dpi = 600)
plt.close()




# Fig0015-6: Mean rate divided by path type, maze 2. ------------------------------------------------------------------------------------
idx = np.where(Data['Maze Type'] == 'Maze 2')[0]
a = len(idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = np.concatenate([Data['Training Day'][idx], Data['Training Day'][idx]]),
             y = np.concatenate([Data['mean_rate'][idx], Data['mean_rate_on_path'][idx]]),
             hue = np.concatenate([np.repeat('Entire Path', len(Data['mean_rate'][idx])),
                                   Data['path_type'][idx]]), ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze 2', loc = 'upper left')
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Meak Rate / Hz', fontsize = fs)
#ax.set_title('Simple Maze('+str(a)+' cells), Maze 2', fontsize = fs)
ax.axis([-0.5,8.5,0,1])
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8], labels = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.set_yticks(np.linspace(0,1,6))
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(p,code_id+'-6-Mean Rate divided by path-CrossMaze-Maze2.svg'), dpi = 600)
plt.savefig(os.path.join(p,code_id+'-6-Mean Rate divided by path-CrossMaze-Maze2.png'), dpi = 600)
plt.close()



'''
"""
# Fig0015-3: Peak rate divided by path type, maze 1 ----------------------------------------------------------------------------------
idx = np.where(Data['Maze Type'] == 'Maze 1')[0]
a = len(idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = np.concatenate([Data['Training Day'][idx],Data['Training Day'][idx],Data['Training Day'][idx]]),
             y = np.concatenate([Data['peak_rate'][idx], Data['cor_peak_rate'][idx],Data['inc_peak_rate'][idx]]), 
             data = Data, 
             hue = np.concatenate([np.repeat('Entire Path', len(Data['peak_rate'][idx])),
                                   np.repeat('Corret Path', len(Data['cor_peak_rate'][idx])),
                                   np.repeat('Incorrect Path', len(Data['inc_peak_rate'][idx]))]), ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze 1',
          bbox_to_anchor = (0.5,1.1))
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Peak Rate / Hz', fontsize = fs)
#ax.set_title('Simple Maze('+str(a)+' cells), Maze 1', fontsize = fs)
ax.axis([-0.5,8.5,0,8])
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9], labels = ['P2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.set_yticks([0,2,4,6,8])
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(figpath,code_id+'-3-Peak Rate divided by path-CrossMaze-smooth-Maze1.svg'), dpi = 600)
plt.savefig(os.path.join(figpath,code_id+'-3-Peak Rate divided by path-CrossMaze-smooth-Maze1.png'), dpi = 600)
plt.close()




# Fig0015-4: Peak rate divided by path type, maze 2. ------------------------------------------------------------------------------------
idx = np.where(Data['Maze Type'] == 'Maze 2')[0]
a = len(idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = np.concatenate([Data['Training Day'][idx],Data['Training Day'][idx],Data['Training Day'][idx]]),
             y = np.concatenate([Data['peak_rate'][idx], Data['cor_peak_rate'][idx],Data['inc_peak_rate'][idx]]), 
             data = Data,
             hue = np.concatenate([np.repeat('Entire Path', len(Data['peak_rate'][idx])),
                                   np.repeat('Corret Path', len(Data['cor_peak_rate'][idx])),
                                   np.repeat('Incorrect Path', len(Data['inc_peak_rate'][idx]))]), ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze 2', loc = 'upper left')
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8], labels = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.set_ylabel('Peak Rate / Hz', fontsize = fs)
#ax.set_title('Simple Maze('+str(a)+' cells), Maze 2', fontsize = fs)
ax.axis([-0.5,8.5,0,8])
ax.set_yticks([0,2,4,6,8])
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(figpath,code_id+'-4-Peak Rate divided by path-CrossMaze-smooth-Maze2.svg'), dpi = 600)
plt.savefig(os.path.join(figpath,code_id+'-4-Peak Rate divided by path-CrossMaze-smooth-Maze2.png'), dpi = 600)
plt.close()




# Fig0015-5: Mean rate divided by path type, maze 1. ----------------------------------------------------------------------------------
idx = np.where(Data['Maze Type'] == 'Maze 1')[0]
a = len(idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = np.concatenate([Data['Training Day'][idx],Data['Training Day'][idx],Data['Training Day'][idx]]),
             y = np.concatenate([Data['mean_rate'][idx], Data['cor_mean_rate'][idx],Data['inc_mean_rate'][idx]]), 
             data = Data, 
             hue = np.concatenate([np.repeat('Entire Path', len(Data['mean_rate'][idx])),
                                   np.repeat('Corret Path', len(Data['cor_mean_rate'][idx])),
                                   np.repeat('Incorrect Path', len(Data['inc_mean_rate'][idx]))]), ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze 1', loc = 'upper left')
ax.xaxis.set_major_locator(x)
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9], labels = ['P2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Mean Rate / Hz', fontsize = fs)
#ax.set_title('Simple Maze('+str(a)+' cells), Maze 1', fontsize = fs)
ax.axis([-0.5,8.5,0,1])
ax.set_yticks(np.linspace(0,1,6))
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(figpath,code_id+'-5-Mean Rate divided by path-CrossMaze-smooth-Maze1.svg'), dpi = 600)
plt.savefig(os.path.join(figpath,code_id+'-5-Mean Rate divided by path-CrossMaze-smooth-Maze1.png'), dpi = 600)
plt.close()




# Fig0015-6: Mean rate divided by path type, maze 2. ------------------------------------------------------------------------------------
idx = np.where(Data['Maze Type'] == 'Maze 2')[0]
a = len(idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = np.concatenate([Data['Training Day'][idx],Data['Training Day'][idx],Data['Training Day'][idx]]),
             y = np.concatenate([Data['mean_rate'][idx], Data['cor_mean_rate'][idx],Data['inc_mean_rate'][idx]]), 
             data = Data, 
             hue = np.concatenate([np.repeat('Entire Path', len(Data['mean_rate'][idx])),
                                   np.repeat('Corret Path', len(Data['cor_mean_rate'][idx])),
                                   np.repeat('Incorrect Path', len(Data['inc_mean_rate'][idx]))]), ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze 2', loc = 'upper left')
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Meak Rate / Hz', fontsize = fs)
#ax.set_title('Simple Maze('+str(a)+' cells), Maze 2', fontsize = fs)
ax.axis([-0.5,8.5,0,1])
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8], labels = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.set_yticks(np.linspace(0,1,6))
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(figpath,code_id+'-6-Mean Rate divided by path-CrossMaze-smooth-Maze2.svg'), dpi = 600)
plt.savefig(os.path.join(figpath,code_id+'-6-Mean Rate divided by path-CrossMaze-smooth-Maze2.png'), dpi = 600)
plt.close()
"""