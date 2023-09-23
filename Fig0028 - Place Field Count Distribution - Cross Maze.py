# We can record tens of, if not hundreds of, cell spontaneously within a session using in vivo calcium imaging methods, and many of these cells recorded show a robust
# capacity of spatial coding. These cells are place cells which are identified by our criteria.

# Many of place cells exhibit a multifield spatial code, and the field number of each cell obeys certain distribution.
# This code is to draw the distribution figure for each session and save them.

from mylib.statistic_test import *
import scipy.stats

code_id = '0028 - Place Field Count'
loc = os.path.join(figpath, code_id)
mkdir(loc)

def mmax(a,b):
    if a >= b:
        return a
    else:
        return b

if os.path.exists(os.path.join(figdata, code_id+'.pkl')):
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = {
        'lamda':np.array([], dtype = np.float64),
        'MiceID':np.array([]), 
        'Training Day':np.array([]),
        'Stage': np.array([]),
        'Maze Type':np.array([]), 
        'Cell Type':np.array([]), 
        'pvalue': np.array([], dtype=np.float64),
        'KS Statics': np.array([], dtype=np.float64)
    }
    for i in tqdm(range(len(f1))):
        # Get Trace
        if os.path.exists(f1['Trace File'][i]) == False or f1['Stage'][i] == 'PRE':
            continue
        
        with open(f1['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
        
        # Get Place Field Number Collection of This Session
        field_number_pc = field_number_session(trace, is_placecell = True)

        x_max = int(np.nanmax(field_number_pc)) + 1

        # To make number of bins a integer.
        # place cell subfigure
        fig = plt.figure(figsize = (4,3))
        ax = Clear_Axes(plt.axes(), close_spines = ['top', 'right'], ifxticks = True, ifyticks = True)
        density = ax.hist(field_number_pc, bins = x_max, range = (0.5, x_max+0.5), rwidth = 0.9, color = 'gray')[0]
        if np.nansum(density) <= 100:
         
            continue

        prob = density / np.nansum(density)
        lam = EqualPoissonFit(np.arange(1, x_max+1), prob)
        y = EqualPoisson(np.arange(1, x_max+1), l = lam) * np.nansum(density)
        
        sta, pvalue = scipy.stats.kstest(field_number_pc, poisson.rvs(lam, size=int(np.nansum(density))), alternative='two-sided')
        #ax.plot(np.arange(1, x_max+1), y, color = 'red', label = 'Equal Poisson\n'+'lambda:'+str(round(lam,3)))
        #ax.legend(title = 'Fit Type', facecolor = 'white', edgecolor = 'white', loc = 'upper right')
        #ax.set_xticks(np.linspace(1, x_max, x_max))
        #ax.set_xlabel("Field Number")
        #ax.set_ylabel("Cell Counts")
        
        y_max = np.max([np.nanmax(density), np.nanmax(y)])+5        
        #ax.set_ylim([0, y_max])
        #ax.set_yticks(ColorBarsTicks(peak_rate=y_max, is_auto=True, tick_number=4))

        Data['lamda'] = np.concatenate([Data['lamda'], [lam]])
        Data['Cell Type'] = np.concatenate([Data['Cell Type'], ['Place Cells']])
        Data['MiceID'] = np.concatenate([Data['MiceID'], [str(int(f1['MiceID'][i]))]])
        Data['Training Day'] = np.concatenate([Data['Training Day'], [f1['training_day'][i]]])
        Data['Maze Type'] = np.concatenate([Data['Maze Type'], [str(int(f1['maze_type'][i]))]])
        Data['pvalue'] = np.concatenate([Data['pvalue'], [pvalue]])
        Data['KS Statics'] = np.concatenate([Data['KS Statics'], [sta]])
        Data['Stage'] = np.concatenate([Data['Stage'], [f1['Stage'][i]]])

        #plt.tight_layout()

        # Save figure (2 copies, 1 is saved togather in figpath and the other is saved in each session's directory)
        p = os.path.join(loc, str(int(f1['MiceID'][i]))+' - session'+str(int(f1['session'][i]))+' - '+str(int(f1['date'][i])))
        #plt.savefig(p+'.png', dpi = 600)
        #plt.savefig(p+'.svg', dpi = 600)
        plt.close()

    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as f:
        pickle.dump(Data, f)
    
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)


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
sns.lineplot(
    x='Training Day',
    y='pvalue',
    data=SubData,
    hue='Maze Type',
    ax=ax1,
    legend=False,
    palette=colors
)

stage_indices = np.where(Data['Stage'] == 'Stage 2')[0]
SubData = SubDict(Data, Data.keys(), idx=stage_indices)
sns.lineplot(
    x='Training Day',
    y='pvalue',
    data=SubData,
    hue='Maze Type',
    ax=ax2,
    palette=colors
)
ax1.semilogy()
ax2.semilogy()
ax1.set_ylim([0.00001, 1])
ax2.set_ylim([0.00001, 1])
ax1.axhline(0.05, color='black', linestyle='--', linewidth=0.8)
ax1.axhline(0.01, color='black', linestyle='--', linewidth=0.8)
ax1.axhline(0.001, color='black', linestyle='--', linewidth=0.8)
ax1.axhline(0.0001, color='black', linestyle='--', linewidth=0.8)
ax2.axhline(0.05, color='black', linestyle='--', linewidth=0.8)
ax2.axhline(0.01, color='black', linestyle='--', linewidth=0.8)
ax2.axhline(0.001, color='black', linestyle='--', linewidth=0.8)
ax2.axhline(0.0001, color='black', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig(join(loc, "pvalue.png"), dpi = 600)
plt.savefig(join(loc, "pvalue.svg"), dpi = 600)
plt.close()


def field_lineplot(Data:dict, save_loc:str = None, mice:str = '11095', cell_type:str = 'Place Cells', yticks = None):
    ValueErrorCheck(cell_type, ['Place Cells', 'All Cells'])
    ValueErrorCheck(mice, ['11095', '11092'])
    print(Data.keys())
    KeyWordErrorCheck(Data, __file__, ['lamda', 'Cell Type', 'MiceID', 'Maze Type', 'Training Day'])

    idx = np.where((Data['MiceID'] == mice)&(Data['Cell Type'] == cell_type))[0]
    SubData = DivideData(Data, index = idx, keys = ['lamda', 'Cell Type', 'MiceID', 'Maze Type', 'Training Day'])
    fig = plt.figure(figsize = (4,3))
    ax = Clear_Axes(plt.axes(), close_spines = ['top', 'right'], ifxticks=True, ifyticks=True)
    sns.lineplot(x = 'Training Day', y = 'lamda', data = SubData, hue = 'Maze Type', err_style = 'bars', err_kws = {'elinewidth':1, 'capsize':3, 'capthick':1})
    ax.legend(facecolor = 'white', edgecolor = 'white', title = 'Maze Type', ncol = 3, loc = 'upper center', fontsize = 8, title_fontsize = 8)
    plt.xticks(np.arange(11), ['P1','P2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7','D8', 'D9'])
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.axis([-0.5,10.5,0,np.max(yticks)])
    ax.set_ylabel('Lambda')
    ax.set_title(mice+' & '+cell_type)
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, mice+' - '+cell_type+' [lineplot].svg'), dpi = 600)
    plt.savefig(os.path.join(save_loc, mice+' - '+cell_type+' [lineplot].png'), dpi = 600)
    plt.close()

#field_lineplot(Data, os.path.join(figpath, code_id), mice = '11095', cell_type = 'Place Cells', yticks = np.linspace(0,7,8))
#field_lineplot(Data, os.path.join(figpath, code_id), mice = '11095', cell_type = 'All Cells', yticks = np.linspace(0,7,8))
#field_lineplot(Data, os.path.join(figpath, code_id), mice = '11092', cell_type = 'Place Cells', yticks = np.linspace(0,10,6))
#field_lineplot(Data, os.path.join(figpath, code_id), mice = '11092', cell_type = 'All Cells', yticks = np.linspace(0,10,6))

# 11095, 11092, 10209, 10212

def plot_subfigure(place_field_num: np.ndarray, ax: Axes):
    x_max = int(np.nanmax(place_field_num)) + 1

    # To make number of bins a integer.
    # place cell subfigure
    ax = Clear_Axes(ax, close_spines = ['top', 'right'], ifxticks = True, ifyticks = True)
    count = ax.hist(place_field_num, bins = x_max, range = (0.5, x_max+0.5), rwidth = 0.8, color = 'gray')[0]
    prob = count / np.nansum(count)
    lam = EqualPoissonFit(np.arange(1, x_max+1), prob)
    y = EqualPoisson(np.arange(1, x_max+1), l = lam) * np.nansum(count)
    ax.plot(np.arange(1, x_max+1), y, color = 'red', linewidth = 0.8)
    ax.set_xticks([1, x_max])
    #ax.set_title(str(round(lam, 3)))
        
    y_max = np.max([np.nanmax(count), np.nanmax(y)])+5
    ax.set_ylim([0, y_max])
    ax.set_yticks([0, np.nanmax(count)])
    return ax

fig, axes = plt.subplots(nrows=13, ncols=6, figsize=(6*2.5, 13*2.1))

def add_plots_on_ax(axes: Axes, stage: str, maze_type: int, mouse: int, row=13):
    idx = np.where((f1['MiceID']==mouse)&(f1['Stage']==stage)&(f1['maze_type']==maze_type))[0]
    uniq_day = f1['training_day'][idx]
    for k in range(row):
        if k >= idx.shape[0]:
            axes[k] = Clear_Axes(axes[k])
            continue     
       
        i = idx[k]
        if exists(f1['Trace File'][i]) == False:
            axes[k]= Clear_Axes(axes[k])
            continue
    
        with open(f1['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
        
        field_num = field_number_session(trace, is_placecell = True)
    
        plot_subfigure(field_num, axes[k])
"""
# 10209, Maze 1
add_plots_on_ax(axes=axes[:, 0], stage = 'Stage 1', maze_type=1, mouse=10209)
print('Done - 10209, Maze 1 - Stage 1')

# 10212, Maze 1
add_plots_on_ax(axes=axes[:, 1], stage = 'Stage 1', maze_type=1, mouse=10212)
print('Done - 10212, Maze 1 - Stage 1')

# 10209, Maze 1
add_plots_on_ax(axes=axes[:, 2], stage = 'Stage 2', maze_type=1, mouse=10209)
print('Done - 10209, Maze 1')

# 10212, Maze 1
add_plots_on_ax(axes=axes[:, 3], stage = 'Stage 2', maze_type=1, mouse=10212)
print('Done - 10212, Maze 1')

# 11095, Maze 1
add_plots_on_ax(axes=axes[:, 4], stage = 'Stage 2', maze_type=1, mouse=11095)
print('Done - 11095, Maze 1')

# 11092, Maze 1
add_plots_on_ax(axes=axes[:, 5], stage = 'Stage 2', maze_type=1, mouse=11092)
print('Done - 11092, Maze 1')


plt.tight_layout()
plt.savefig(join(loc, 'A_Sheet [Maze 1].png'), dpi=600)
plt.savefig(join(loc, 'A_Sheet [Maze 1].svg'), dpi=600)
plt.close()


fig, axes = plt.subplots(nrows=13, ncols=4, figsize=(4*2.5, 13*2.1))
# 10209, Maze 2
add_plots_on_ax(axes=axes[:, 0], stage = 'Stage 2', maze_type=2, mouse=10209)
print('Done - 10209, Maze 2')

# 10212, Maze 2
add_plots_on_ax(axes=axes[:, 1], stage = 'Stage 2', maze_type=2, mouse=10212)
print('Done - 10212, Maze 2')

# 11095, Maze 2
add_plots_on_ax(axes=axes[:, 2], stage = 'Stage 2', maze_type=2, mouse=11095)
print('Done - 11095, Maze 2')

# 11092, Maze 2
add_plots_on_ax(axes=axes[:, 3], stage = 'Stage 2', maze_type=2, mouse=11092)
print('Done - 11092, Maze 2')
plt.tight_layout()
plt.savefig(join(loc, 'A_Sheet [Maze 2].png'), dpi=600)
plt.savefig(join(loc, 'A_Sheet [Maze 2].svg'), dpi=600)
plt.close()
"""