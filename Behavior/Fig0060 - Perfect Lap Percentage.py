from mylib.statistic_test import *

code_id = '0060 - Perfect Lap Percentage'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Lap ID', 'Perfect Lap', 'Distance', 'Navigation Time', 'Average Velocity'], is_behav=True,
                              f = f_pure_behav, function = PerfectLapIdentify_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
    
if os.path.exists(join(figdata, code_id+' [percentage].pkl')):
    with open(join(figdata, code_id+' [percentage].pkl'), 'rb') as handle:
        PercentData = pickle.load(handle)
else:
    PercentData = DataFrameEstablish(variable_names = ['Perfect Lap Percentage'], is_behav=True,
                              f = f_pure_behav, function = PerfectLapPercentage_Interface, 
                              file_name = code_id+' [percentage]', behavior_paradigm = 'CrossMaze')

idx = np.where((PercentData['MiceID']!=11092)&(PercentData['MiceID']!=11094)&(PercentData['MiceID']!=11095))[0]
PercentData = SubDict(PercentData, PercentData.keys(), idx=idx)

idx = np.where((Data['MiceID']!=11092)&(Data['MiceID']!=11094)&(Data['MiceID']!=11095))[0]
Data = SubDict(Data, Data.keys(), idx=idx)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,2))
ax1, ax2 = axes[0], axes[1]
ax1 = Clear_Axes(ax1, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(ax2, close_spines=['top', 'right'], ifxticks=True)

colors = sns.color_palette("rocket", 4)
colors = [colors[0], colors[-1]]
PercentData['Perfect Lap Percentage'] = PercentData['Perfect Lap Percentage'] * 100

idx = np.where(PercentData['Stage'] == 'Stage 1')
SubData = SubDict(PercentData, PercentData.keys(), idx=idx)

idx = np.where(SubData['Training Day'] == '>=Day 10')[0]
print_estimator(SubData['Perfect Lap Percentage'][idx])
sns.lineplot(
    x='Training Day',
    y='Perfect Lap Percentage',
    hue='Maze Type',
    data = SubData,
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
ax1.set_ylim([-3,100])
ax1.set_yticks(np.linspace(0,100,6))

idx = np.where(PercentData['Stage'] == 'Stage 2')
SubData = SubDict(PercentData, PercentData.keys(), idx=idx)
sns.lineplot(
    x='Training Day',
    y='Perfect Lap Percentage',
    hue='Maze Type',
    data=SubData,
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
ax2.set_ylim([-3,100])
ax2.set_yticks(np.linspace(0,100,6))
plt.savefig(join(loc, 'perfect lap percentage.png'), dpi=600)
plt.savefig(join(loc, 'perfect lap percentage.svg'), dpi=600)
plt.close()

idx = np.where(Data['Perfect Lap'] == 1)[0]
time_perfect = Data['Navigation Time'][idx]
speed_perfect = Data['Average Velocity'][idx]
distance_perfect = Data['Distance'][idx]

idx = np.where(Data['Perfect Lap'] == 0)[0]
time_incorrect = Data['Navigation Time'][idx]
speed_incorrect = Data['Average Velocity'][idx]
distance_incorrect = Data['Distance'][idx]

print("Speed:", levene(speed_perfect, speed_incorrect), ttest_ind(speed_perfect, speed_incorrect, equal_var=False, alternative='greater'))
print("Distance:", levene(distance_perfect, distance_incorrect), ttest_ind(distance_perfect, distance_incorrect, equal_var=False, alternative='less'))
print("Time:", levene(time_perfect, time_incorrect), ttest_ind(time_perfect, time_incorrect, equal_var=False, alternative='less'))

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(4,2), gridspec_kw={'width_ratios':[3,1]})
ax = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax1 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(time_perfect, bins=90, alpha=0.5, range=(10,100), label='Perfect')
ax.hist(time_incorrect, bins=90, alpha=0.5, range=(10,100), label='Incorrect')
ax.axis([10,100, 0,400])
ax.legend()
ax1.hist(time_perfect, bins=10, alpha=0.5, range=(100,3000), label='Perfect')
ax1.hist(time_incorrect, bins=400, alpha=0.5, range=(100,500), label='Incorrect')
ax1.set_xlim([100,500])
plt.savefig(join(loc, 'time histogram.png'), dpi=600)
plt.savefig(join(loc, 'time histogram.svg'), dpi=600)
plt.close()

fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(speed_perfect, bins=50, alpha=0.5, range=(0,50), label='Perfect')
ax.hist(speed_incorrect, bins=50, alpha=0.5, range=(0,50), label='Incorrect')
ax.set_xlim([0,60])
ax.set_ylim([0, 300])
ax.legend()
plt.savefig(join(loc, 'speed histogram.png'), dpi=600)
plt.savefig(join(loc, 'speed histogram.svg'), dpi=600)
plt.close()

idx = np.where((Data['Perfect Lap'] == 1)&(Data['Maze Type'] == 'Maze 1'))[0]
distance_perfect = Data['Distance'][idx]
idx = np.where((Data['Perfect Lap'] == 0)&(Data['Maze Type'] == 'Maze 1'))[0]
distance_incorrect = Data['Distance'][idx]
print("Distance:", levene(distance_perfect, distance_incorrect), ttest_ind(distance_perfect, distance_incorrect, equal_var=False, alternative='less'))
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(4,2), gridspec_kw={'width_ratios':[3,1]})
ax = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax1 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(distance_perfect, bins=50, alpha=0.5, range=(600,800), label='Perfect')
ax.hist(distance_incorrect, bins=50, alpha=0.5, range=(600,800), label='Incorrect')
ax.set_xlim(600, 800)
ax.set_ylim(0, 200)
ax.legend()
ax1.hist(distance_perfect, bins=190, alpha=0.5, range=(800,20000), label='Perfect')
ax1.hist(distance_incorrect, bins=190, alpha=0.5, range=(800,20000), label='Incorrect')
ax1.set_xlim([800,20000])
#ax1.set_ylim([0,60])
plt.savefig(join(loc, 'distance histogram [maze 1].png'), dpi=600)
plt.savefig(join(loc, 'distance histogram [maze 1].svg'), dpi=600)
plt.close()

idx = np.where((Data['Perfect Lap'] == 1)&(Data['Maze Type'] == 'Maze 2'))[0]
distance_perfect = Data['Distance'][idx]
idx = np.where((Data['Perfect Lap'] == 0)&(Data['Maze Type'] == 'Maze 2'))[0]
distance_incorrect = Data['Distance'][idx]
print("Distance:", levene(distance_perfect, distance_incorrect), ttest_ind(distance_perfect, distance_incorrect, equal_var=False, alternative='less'))
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(4,2), gridspec_kw={'width_ratios':[3,1]})
ax = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax1 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(distance_perfect, bins=60, alpha=0.5, range=(500,800), label='Perfect')
ax.hist(distance_incorrect, bins=60, alpha=0.5, range=(500,800), label='Incorrect')
ax.set_xlim(500, 800)
ax.set_ylim(0, 300)
ax.legend()
ax1.hist(distance_perfect, bins=190, alpha=0.5, range=(800,20000), label='Perfect')
ax1.hist(distance_incorrect, bins=190, alpha=0.5, range=(800,20000), label='Incorrect')
ax1.set_xlim([800,20000])
#ax1.set_ylim([0,60])
plt.savefig(join(loc, 'distance histogram [maze 2].png'), dpi=600)
plt.savefig(join(loc, 'distance histogram [maze 2].svg'), dpi=600)
plt.close()

# First Session & Second Session
idx = np.where((PercentData['Stage'] == 'Stage 1')&(PercentData['Training Day'] == 'Day 1')&(PercentData['Maze Type'] == 'Maze 1'))[0]
data_11 = PercentData['Perfect Lap Percentage'][idx]
print_estimator(data_11)
idx = np.where((PercentData['Stage'] == 'Stage 2')&(PercentData['Training Day'] == 'Day 1')&(PercentData['Maze Type'] == 'Maze 2'))[0]
data_21 = PercentData['Perfect Lap Percentage'][idx]
print_estimator(data_21)

colors = sns.color_palette("rocket", 4)
colors = [colors[0], colors[-1]]
markercolors = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[0]]

fig = plt.figure(figsize=(2,3))
colors = sns.color_palette("rocket", 3)
markercolor = sns.color_palette("Blues", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
compdata = {
    'Condition': np.concatenate([np.repeat('Novel Maze 1', data_11.shape[0]), np.repeat('Novel Maze 2', data_21.shape[0])]),
    'Perfect Lap Percentage': np.concatenate([data_11, data_21]),
}
sns.barplot(
    x='Condition',
    y='Perfect Lap Percentage',
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
    y='Perfect Lap Percentage',
    data=compdata,
    palette=markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    jitter=0.2
)
ax.set_ylim([-3,500])
#ax.set_yticks(np.linspace(0, 50, 6))
ax.set_xticks([0, 1], ["Novel\nMaze 1", "Novel\nMaze 2"])
ax.set_ylabel('Lap-wise Travel distance')
plt.tight_layout()
plt.savefig(join(loc, "Comparison of Maze 1 and Maze 2 [S1].png"), dpi=600)
plt.savefig(join(loc, "Comparison of Maze 1 and Maze 2 [S1].svg"), dpi=600)
plt.close()
print("First Session:", levene(data_11, data_21))
print(ttest_ind(data_11, data_21, alternative='less', equal_var=False), end='\n\n')

import gc

def plot_sample(
    f: pd.DataFrame,
    mouse: int,
    date: int,
    session: int,
    lap: int
):
    idx = np.where((f['MiceID'] == mouse)&(f['date'] == date)&(f['session'] == session))[0]
    if len(idx) == 0:
        raise ValueError(f"Do not find session with mouse {mouse}, date {date}, session {session}.")
    
    if exists(f['Trace File'][idx[0]]):
        with open(f['Trace File'][idx[0]], 'rb') as handle:
            trace = pickle.load(handle)
        
        beg, end = LapSplit(trace, 'CrossMaze')
        
        if lap > beg.shape[0]:
            raise ValueError(f"lap {lap} is out of range {beg.shape[0]}.")
        
        fig = plt.figure(figsize=(3,3))
        ax = Clear_Axes(plt.axes())
        ax.set_aspect('equal')
        DrawMazeProfile(axes=ax, maze_type=int(f['maze_type'][idx[0]]), color='black', linewidth=0.5)
        x, y = trace['correct_pos'][beg[lap-1]:end[lap-1], 0], trace['correct_pos'][beg[lap-1]:end[lap-1], 1]
        
        x, y = x/20-0.5, y/20-0.5
        ax.plot(x, y, linewidth=0.5)
        ax.axis([-0.6, 47.6, 47.6, -0.6])
        ax.set_title(f'Lap {lap}, Mouse {mouse}')
        plt.savefig(join(loc, f'{mouse}_{date}_{session}_{lap}.png'), dpi=600)
        plt.savefig(join(loc, f'{mouse}_{date}_{session}_{lap}.svg'), dpi=600)
        plt.close()
        del trace
        gc.collect()
        print(f"  Plot done. Saved as {mouse}_{date}_{session}_{lap}.png")
    else:
        raise ValueError(f"Trace file not found: {f['Trace File'][idx[0]]}")


plot_sample(f1, mouse=10209, date=20230728, session=2, lap=12)
plot_sample(f1, mouse=10209, date=20230728, session=3, lap=7)

plot_sample(f1, mouse=10212, date=20230728, session=2, lap=14)
plot_sample(f1, mouse=10212, date=20230728, session=3, lap=5)

plot_sample(f1, mouse=10224, date=20230930, session=2, lap=24)
plot_sample(f1, mouse=10224, date=20230930, session=3, lap=28)

plot_sample(f1, mouse=10227, date=20230928, session=2, lap=31)
plot_sample(f1, mouse=10227, date=20230930, session=3, lap=28)