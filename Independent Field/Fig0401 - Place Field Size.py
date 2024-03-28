from mylib.statistic_test import *
import scipy.stats
from scipy.stats import poisson, norm, weibull_min, lognorm, gamma
from mylib.stats.kstest import nbinom_kstest, lognorm_kstest, gamma_kstest

code_id = "0401 - Place Field Size"
loc = os.path.join(figpath, "Independent Field", code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, 'Field Pool.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['FSC Stability', 'OEC Stability', 'Field Size', 'Field Length', 'Peak Rate', 'Position'], f = f1,
                              function = WithinFieldBasicInfo_Interface, func_kwgs = {'is_placecell': True},
                              file_name = 'Field Pool', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, 'Field Pool.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata, 'Field Statistics in Cell Pool.pkl')) == False:
    CellData = DataFrameEstablish(variable_names = ['Mean FSC', 'Std. FSC', 'Median FSC', 'Error FSC',
                                                'Mean OEC', 'Std. OEC', 'Median OEC', 'Error OEC',
                                                'Mean Size', 'Std. Size', 'Median Size', 'Error Size',
                                                'Mean Length', 'Std. Length', 'Median Length', 'Error Length',
                                                'Mean Rate', 'Std. Rate', 'Median Rate', 'Error Rate',
                                                'Mean Position', 'Std. Position', 'Median Position', 'Error Position',
                                                'Mean Interdistance', 'Std. Interdistance', 'Median Interdistance', 'Error Interdistance',
                                                'Cell ID', 'Field Number'], f = f1,
                              function = WithinCellFieldStatistics_Interface, func_kwgs = {'is_placecell': True},
                              file_name = 'Field Statistics in Cell Pool', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, 'Field Statistics in Cell Pool.pkl'), 'rb') as handle:
        CellData = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)

Data['Field Size'] = Data['Field Size']

idx = np.where((Data['MiceID'] != 11092)&(Data['MiceID'] != 11095))[0]
Data = SubDict(Data, Data.keys(), idx=idx)

colors = sns.color_palette("rocket", 3)
uniq_day = ['Day 1', 'Day 2', 'Day 3', 'Day 4',
            'Day 5', 'Day 6', 'Day 7', 'Day 8',
            'Day 9', '>=Day 10']
dates = ['Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9', '>=Day 10']
def plotfigures(
    maze: int, 
    mouse: int,
    bin_num: int = 400,
    yloglim: int = 1000
):
    dates = ['Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9', '>=Day 10']
    if maze == 1:
        indices = np.concatenate([np.where((Data['Training Day'] == day)&
                                           (Data['Stage'] == 'Stage 1')&
                                           (Data['MiceID'] == mouse)&
                                           (Data['Maze Type'] == 'Maze 1'))[0] for day in dates]+
                                 [np.where((Data['Stage'] == 'Stage 2')&
                                           (Data['MiceID'] == mouse)&
                                           (Data['Maze Type'] == 'Maze 1'))[0]])
    elif maze == 2:
        indices = np.concatenate([np.where((Data['Training Day'] == day)&
                                           (Data['MiceID'] == mouse)&
                                           (Data['Maze Type'] == 'Maze 2'))[0] for day in dates])
        
    SubData = SubDict(Data, Data.keys(), idx=indices)

    fig = plt.figure(figsize = (4,3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    freq = ax.hist(SubData['Field Size'], bins=bin_num, range=(-0.5, bin_num+0.5), color = 'gray')[0]
    ymax = np.max(freq)
    pdf = freq/np.sum(freq)
    shape, locc, scale = lognorm.fit(SubData['Field Size'], floc=0)
    y = lognorm.pdf(np.arange(bin_num), shape, loc=locc, scale=scale)
    y = y/np.sum(y)*np.sum(freq)
    #print(shape, locc, scale, np.sum(y))
    ax.plot(np.arange(bin_num), y, linewidth = 0.5, label='Lognormal')
    dist = lognorm.rvs(shape, loc=locc, scale=scale, size=int(np.sum(freq)))
    print(f"Maze {maze}, Mouse {mouse}, Shape Num {len(SubData['Field Size'])}:")
    print(f"    Lognormal: shape {shape}, loc {locc}, scale {scale}")
    statistic_p, lognorm_p = lognorm_kstest(SubData['Field Size'], resample_size=1629)
    print(f"    Lognormal Statistic, {statistic_p}, Lognormal P-value: {lognorm_p}", end="\n\n")

    alpha, c, beta = gamma.fit(SubData['Field Size'], floc=0)
    y = gamma.pdf(np.arange(bin_num), alpha, scale=beta)
    y = y/np.nansum(y)*np.nansum(freq)
    ax.plot(np.arange(bin_num), y, linewidth = 0.5, label='Gamma')
    print(f"    Gamma: alpha {alpha}, beta {beta}", end='\n\n')
    statistic_p, gamma_p = gamma_kstest(SubData['Field Size'], resample_size=1629, is_floc=True)
    print(f"    Gamma Statistic, {statistic_p}, Gamma P-value: {gamma_p}", end="\n\n")
    
    ax.legend()
    ax.set_xlim([-0.5, bin_num])
    ax.set_xlim([0, bin_num])
    ax.set_xticks(np.linspace(0, bin_num, 5))
    ax.set_xlabel("Field Size / bin")
    ax.set_ylabel("Field Count")
    plt.tight_layout()
    plt.savefig(os.path.join(loc, f'Maze {maze} Mouse {mouse} - Field Size Distribution.png'), dpi = 600)
    plt.savefig(os.path.join(loc, f'Maze {maze} Mouse {mouse} - Field Size Distribution.svg'), dpi = 600)
    plt.close()

    # Distribution of Place Field Size
    fig = plt.figure(figsize = (3,2))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.hist(SubData['Field Size'], bins=bin_num, range=(0.5, bin_num+0.5), color = 'black')
    ax.set_xlim([0, bin_num+1])
    ax.set_xticks(np.linspace(0, bin_num, 5))
    ax.set_ylim(1, yloglim)
    ax.semilogy()
    plt.tight_layout()
    plt.savefig(os.path.join(loc, f'Maze {maze} Mouse {mouse} - Field Size Distribution [semilogy].png'), dpi = 600)
    plt.savefig(os.path.join(loc, f'Maze {maze} Mouse {mouse} - Field Size Distribution [semilogy].svg'), dpi = 600)
    plt.close()
    

# draw
plotfigures(maze=1, mouse=10209)
plotfigures(maze=2, mouse=10209, yloglim=1000)

plotfigures(maze=1, mouse=10212)
plotfigures(maze=2, mouse=10212, yloglim=1000)

plotfigures(maze=1, mouse=10224)
plotfigures(maze=2, mouse=10224, yloglim=1000)

plotfigures(maze=1, mouse=10227)
plotfigures(maze=2, mouse=10227, yloglim=1000)


maze = 'Open Field'
indices = np.concatenate([np.where((Data['Training Day'] == day)&(Data['Maze Type'] == maze))[0] for day in dates])
SubData = SubDict(Data, Data.keys(), idx=indices)

# Distribution of Place Field Size
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
density = ax.hist(SubData['Field Size'], bins=100, range=(0.5, 1000.5), rwidth=0.8, color = 'black')[0]
density = density/np.sum(density)
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
ax.hist(SubData['Field Size'], bins=120, range=(0.5, 1000.5), rwidth=0.8, color = 'black')
ax.set_xlim([0, 1001])
ax.set_xticks(np.linspace(0, 1000, 6))
ax.set_yticks(ColorBarsTicks(peak_rate=ymax, is_auto=True, tick_number=4))
ax.semilogy()
plt.tight_layout()
plt.savefig(os.path.join(loc, maze+' Field Size Distribution [semilogy].png'), dpi = 600)
plt.savefig(os.path.join(loc, maze+' Field Size Distribution [semilogy].svg'), dpi = 600)
plt.close()


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
    y='Field Size',
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
SubSample = SubDict(SubData1, SubData1.keys(), np.random.choice(np.arange(len(SubData1['Field Size'])), replace=True, size=10000))
sns.stripplot(
    x='Training Day',
    y='Field Size',
    hue='Maze Type',
    data=SubSample,
    palette=markercolors,
    edgecolor='black',
    size=2,
    linewidth=0.05,
    ax = ax1,
    dodge=True,
    jitter=0.1
)
ax1.set_ylim([0, 2400])
ax1.set_yticks(np.linspace(0, 2400, 7))

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 1')&(SubData['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1']])
SubData2 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Field Size',
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
SubSample = SubDict(SubData2, SubData2.keys(), np.random.choice(np.arange(len(SubData2['Field Size'])), replace=True, size=10000))
sns.stripplot(
    x='Training Day',
    y='Field Size',
    hue='Maze Type',
    data=SubSample,
    hue_order=['Open Field', 'Maze 1'],
    palette=markercolors,
    edgecolor='black',
    size=2,
    linewidth=0.05,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax2.set_ylim([0, 2400])
ax2.set_yticks(np.linspace(0, 2400, 7))

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 2')&(SubData['Maze Type'] == m))[0] for m in ['Open Field', 'Maze 1', 'Maze 2']])
SubData3 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Field Size',
    data=SubData3,
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
SubSample = SubDict(SubData3, SubData3.keys(), np.random.choice(np.arange(len(SubData3['Field Size'])), replace=True, size=10000))
sns.stripplot(
    x='Training Day',
    y='Field Size',
    hue='Maze Type',
    hue_order=['Open Field', 'Maze 1', 'Maze 2'],
    data=SubSample,
    palette=markercolors,
    edgecolor='black',
    size=2,
    linewidth=0.05,
    ax = ax3,
    dodge=True,
    jitter=0.1
)
ax3.set_ylim([0, 2400])
ax3.set_yticks(np.linspace(0, 2400, 7))
plt.tight_layout()
plt.savefig(join(loc, 'Field Size Change.png'), dpi=600)
plt.savefig(join(loc, 'Field Size Change.svg'), dpi=600)
plt.close()


fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3))
ax2 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 1')&(SubData['Maze Type'] == m))[0] for m in ['Maze 1']])
SubData2 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Field Length',
    data=SubData2,
    hue='Maze Type',
    palette=colors[1:],
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax2
)
SubSample = SubDict(SubData2, SubData2.keys(), np.random.choice(np.arange(len(SubData2['Field Length'])), replace=True, size=10000))
sns.stripplot(
    x='Training Day',
    y='Field Length',
    hue='Maze Type',
    data=SubSample,
    palette=markercolors[1:],
    edgecolor='black',
    size=2,
    linewidth=0.05,
    ax = ax2,
    dodge=True,
    jitter=0.1
)
ax2.set_ylim([0, 400])
ax2.set_yticks(np.linspace(0, 400, 5))

idx = np.concatenate([np.where((SubData['Stage'] == 'Stage 2')&(SubData['Maze Type'] == m))[0] for m in ['Maze 1', 'Maze 2']])
SubData3 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='Field Length',
    data=SubData3,
    hue='Maze Type',
    palette=colors[1:],
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
    ax=ax3
)
SubSample = SubDict(SubData3, SubData3.keys(), np.random.choice(np.arange(len(SubData3['Field Length'])), replace=True, size=10000))
sns.stripplot(
    x='Training Day',
    y='Field Length',
    hue='Maze Type',
    hue_order=['Maze 1', 'Maze 2'],
    data=SubSample,
    palette=markercolors[1:],
    edgecolor='black',
    size=2,
    linewidth=0.05,
    ax = ax3,
    dodge=True,
    jitter=0.1
)
ax3.set_ylim([0, 300])
ax3.set_yticks(np.linspace(0, 300, 7))
plt.tight_layout()
plt.savefig(join(loc, 'Field Length Change.png'), dpi=2400)
plt.savefig(join(loc, 'Field Length Change.svg'), dpi=2400)
plt.close()