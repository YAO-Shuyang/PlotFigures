from mylib.statistic_test import *

code_id = '0018 - Spatial Information Regression Out Speed'
loc = os.path.join(figpath, code_id)
mkdir(loc)

idx = np.where((np.isin(f1['MiceID'], [10209, 10212, 10224, 10227]))&(f1['maze_type'] != 0))[0]
if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ["SI", "Speed Level"], f = f1, 
                              function = SI_RegressOut_Speed_Interface, file_idx=idx,
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
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