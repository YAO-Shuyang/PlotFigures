from mylib.statistic_test import *
import scipy.stats
from scipy.stats import poisson, norm

code_id = "0049 - Place Field Size"
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Field Size', 'Std', 'Error', 'Field Number', 'Cell ID'], f = f1,
                              function = PlaceFieldSize_Interface, func_kwgs = {'is_placecell': True},
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)


fig, axes = plt.subplots(ncols=2, nrows=1, figsize = (8,3))
ax1, ax2 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True), Clear_Axes(axes[1], close_spines=['top', 'right', 'left'],ifxticks=True)
colors = sns.color_palette("rocket", 3)
uniq_day = ['Day 1', 'Day 2', 'Day 3', 'Day 4',
            'Day 5', 'Day 6', 'Day 7', 'Day 8',
            'Day 9', '>=Day 10']
idx = np.concatenate([np.where(Data['Training Day'] == day)[0] for day in uniq_day])
Data = SubDict(Data, Data.keys(), idx)
stage_indices = np.where(Data['Stage'] == 'Stage 1')[0]
SubData = SubDict(Data, Data.keys(), idx=stage_indices)
sns.barplot(
    x='Training Day',
    y='Field Size',
    data=SubData,
    hue='Maze Type',
    ax=ax1,
    palette=colors
)

stage_indices = np.where(Data['Stage'] == 'Stage 2')[0]
SubData = SubDict(Data, Data.keys(), idx=stage_indices)
sns.lineplot(
    x='Training Day',
    y='Field Size',
    data=SubData,
    hue='Maze Type',
    ax=ax2,
    palette=colors
)

plt.tight_layout()
plt.savefig(os.path.join(loc, "Cross Env Field Size Comparison.png"), dpi = 2400)
plt.savefig(os.path.join(loc, "Cross Env Field Size Comparison.svg"), dpi = 2400)
plt.close()


# Field number with size
maze= 'Maze 1'
dates = ['Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9', '>=Day 10']
indices = np.concatenate([np.where((Data['Training Day'] == day)&(Data['Maze Type'] == maze))[0] for day in dates])
SubData = SubDict(Data, Data.keys(), idx=indices)

# Distribution of Place Field Size
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
density = ax.hist(SubData['Field Size'], bins=50, range=(0.5, 500.5), rwidth=0.8, color = 'gray')[0]
ymax = np.max(density)
ax.set_xlim([0, 501])
ax.set_xticks(np.linspace(0, 500, 6))
ax.set_xlabel("Field Size / bin")
ax.set_ylabel("Field Count")
ax.set_yticks(ColorBarsTicks(peak_rate=ymax, is_auto=True, tick_number=4))
plt.tight_layout()
plt.savefig(os.path.join(loc, maze+' Field Size Distribution.png'), dpi = 600)
plt.savefig(os.path.join(loc, maze+' Field Size Distribution.svg'), dpi = 600)
plt.close()

# Distribution of Place Field Size
fig = plt.figure(figsize = (3,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(SubData['Field Size'], bins=50, range=(0.5, 500.5), rwidth=0.8, color = 'gray')
ax.set_xlim([0, 501])
ax.set_xticks(np.linspace(0, 500, 6))
ax.set_yticks(ColorBarsTicks(peak_rate=ymax, is_auto=True, tick_number=4))
ax.semilogy()
plt.tight_layout()
plt.savefig(os.path.join(loc, maze+' Field Size Distribution [semilogy].png'), dpi = 600)
plt.savefig(os.path.join(loc, maze+' Field Size Distribution [semilogy].svg'), dpi = 600)
plt.close()

maze = 'Maze 2'
indices = np.concatenate([np.where((Data['Training Day'] == day)&(Data['Maze Type'] == maze))[0] for day in dates])
SubData = SubDict(Data, Data.keys(), idx=indices)

# Distribution of Place Field Size
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
density = ax.hist(SubData['Field Size'], bins=60, range=(0.5, 600.5), rwidth=0.8, color = 'gray')[0]
ymax = np.max(density)

ax.set_xlim([0, 601])
ax.set_xticks(np.linspace(0, 600, 7))
ax.set_xlabel("Field Size / bin")
ax.set_ylabel("Field Count")
ax.set_yticks(ColorBarsTicks(peak_rate=ymax, is_auto=True, tick_number=4))
plt.tight_layout()
plt.savefig(os.path.join(loc, maze+' Field Size Distribution.png'), dpi = 600)
plt.savefig(os.path.join(loc, maze+' Field Size Distribution.svg'), dpi = 600)
plt.close()

# Distribution of Place Field Size
fig = plt.figure(figsize = (3,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(SubData['Field Size'], bins=60, range=(0.5, 600.5), rwidth=0.8, color = 'gray')
ax.set_xlim([0, 601])
ax.set_xticks(np.linspace(0, 600, 7))
ax.set_yticks(ColorBarsTicks(peak_rate=ymax, is_auto=True, tick_number=4))
ax.semilogy()
plt.tight_layout()
plt.savefig(os.path.join(loc, maze+' Field Size Distribution [semilogy].png'), dpi = 600)
plt.savefig(os.path.join(loc, maze+' Field Size Distribution [semilogy].svg'), dpi = 600)
plt.close()


maze = 'Open Field'
indices = np.concatenate([np.where((Data['Training Day'] == day)&(Data['Maze Type'] == maze))[0] for day in dates])
SubData = SubDict(Data, Data.keys(), idx=indices)

# Distribution of Place Field Size
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
density = ax.hist(SubData['Field Size'], bins=100, range=(0.5, 1000.5), rwidth=0.8, color = 'gray')[0]
ymax = np.max(density)

ax.set_xlim([0, 1001])
ax.set_xticks(np.linspace(0, 1000, 6))
ax.set_xlabel("Field Size / bin")
ax.set_ylabel("Field Count")
ax.set_yticks(ColorBarsTicks(peak_rate=ymax, is_auto=True, tick_number=4))
plt.tight_layout()
plt.savefig(os.path.join(loc, maze+' Field Size Distribution.png'), dpi = 600)
plt.savefig(os.path.join(loc, maze+' Field Size Distribution.svg'), dpi = 600)
plt.close()

# Distribution of Place Field Size
fig = plt.figure(figsize = (3,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(SubData['Field Size'], bins=120, range=(0.5, 1000.5), rwidth=0.8, color = 'gray')
ax.set_xlim([0, 1001])
ax.set_xticks(np.linspace(0, 1000, 6))
ax.set_yticks(ColorBarsTicks(peak_rate=ymax, is_auto=True, tick_number=4))
ax.semilogy()
plt.tight_layout()
plt.savefig(os.path.join(loc, maze+' Field Size Distribution [semilogy].png'), dpi = 600)
plt.savefig(os.path.join(loc, maze+' Field Size Distribution [semilogy].svg'), dpi = 600)
plt.close()