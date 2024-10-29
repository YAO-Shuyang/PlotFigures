from mylib.statistic_test import *

code_id = '0018 - Spatial Information Regression Out Speed'
loc = os.path.join(figpath, code_id)
mkdir(loc)

from mazepy.basic.convert import coordinate_recording_time

def separate_speed_bins(trace):
    beg, end = trace['lap beg time'], trace['lap end time']
    Speed = np.zeros(end.shape[0], np.float64)
    
    for i in range(end.shape[0]):
        behav_idx = np.where((trace['correct_time'] >= beg[i]) & (trace['correct_time'] <= end[i]))[0]

        dx = np.ediff1d(trace['processed_pos_new'][behav_idx, 0])/10
        dy = np.ediff1d(trace['processed_pos_new'][behav_idx, 1])/10
        
        dt = trace['correct_time'][behav_idx[-1]] - trace['correct_time'][behav_idx[0]]
        Speed[i] = np.sum(np.sqrt(dx**2+dy**2))/dt*1000
    
    speed_labels = np.clip((Speed // 10).astype(np.int64), 0, 4)
    
    

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    for mouse in [10209, 10212, 10224, 10227]:
        for maze_type in [1, 2]:
            idx = np.where((f1['MiceID'] == mouse)&(f1['maze_type'] == maze_type))[0]

            
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

colors = sns.color_palette("rocket", 3)

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8, 3))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

idx = np.where((Data['Stage'] == "Stage 1") & (Data['Maze Type'] == "Maze 1") & (Data['Speed Level'] <= 3))[0]
SubData = SubDict(Data, Data.keys(), idx)
sns.lineplot(
    x = "Training Day",
    y = "SI",
    hue = "Speed Level",
    data = SubData,
    palette = sns.color_palette("rainbow", 4),
    ax = ax1,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
idx = np.where((Data['Stage'] == "Stage 2") & (Data['Maze Type'] == "Maze 1") & (Data['Speed Level'] <= 3))[0]
SubData = SubDict(Data, Data.keys(), idx)
sns.lineplot(
    x = "Training Day",
    y = "SI",
    hue = "Speed Level",
    data = SubData,
    palette = sns.color_palette("rainbow", 4),
    ax = ax2,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
ax1.set_ylim(0, 5)
ax1.set_yticks(np.linspace(0, 5, 6))
ax2.set_ylim(0, 5)
ax2.set_yticks(np.linspace(0, 5, 6))
plt.savefig(os.path.join(loc, 'SI [Maze 1].png'), dpi=600)
plt.savefig(os.path.join(loc, 'SI [Maze 1].svg'), dpi=600)
plt.close()

fig = plt.figure(figsize=(4, 3))
ax1 = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

idx = np.where((Data['Stage'] == "Stage 2") & (Data['Maze Type'] == "Maze 2") & (Data['Speed Level'] <= 3))[0]
SubData = SubDict(Data, Data.keys(), idx)
sns.lineplot(
    x = "Training Day",
    y = "SI",
    hue = "Speed Level",
    data = SubData,
    palette = sns.color_palette("rainbow", 4),
    ax = ax1,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
ax1.set_ylim(0, 5)
ax1.set_yticks(np.linspace(0, 5, 6))
ax2.set_ylim(0, 5)
ax2.set_yticks(np.linspace(0, 5, 6))
plt.savefig(os.path.join(loc, 'SI [Maze 2].png'), dpi=600)
plt.savefig(os.path.join(loc, 'SI [Maze 2].svg'), dpi=600)
plt.close()