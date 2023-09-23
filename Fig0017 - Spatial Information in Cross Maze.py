# Fig0017-1 Spatial Information Line Graph, for cross maze paradigm.

# data ara saved in 0017data.pkl, data is a dict type with 5 members: 
#                                     'MiceID': numpy.ndarray, shape(33301,), dtype(str) {e.g. 10031}
#                                     'Training Day': numpy.ndarray, shape(33301,), dtype(str) {e.g. Day 1}
#                                     'SI': numpy.ndarray, shape(33301,), dtype(np.float64)
#                                     'Cell': numpy.ndarray, shape(33301,), dtype(np.int64) {e.g. Cell 1}
#                                     'Maze Type': numpy.ndarray, shape(33301,), dtype(str) {e.g. Maze 1}
# All Cells (33301 cells), spike number > 30

from mylib.statistic_test import *

code_id = '0017 - Spatial Information - Cross Maze'
loc = os.path.join(figpath,code_id)
mkdir(loc)

idx = np.where((f1['MiceID'] != 10224)&(f1['MiceID'] != 10227))[0]
if os.path.exists(os.path.join(figdata,code_id+'-pc.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Cell','SI'], 
                              file_idx=idx, func_kwgs={'is_placecell':True},
                              f = f1, function = SpatialInformation_Interface, 
                              file_name = code_id+'-pc', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata,code_id+'-pc.pkl'), 'rb') as handle:
        Data = pickle.load(handle)


from scipy.stats import zscore
index6s = np.where((Data['MiceID'] == '11095')|(Data['MiceID'] == '11092'))[0]
index6f = np.where((Data['MiceID'] == '10212')|(Data['MiceID'] == '10209'))[0]

Data['SI Z-score'] = np.zeros_like(Data['SI'], dtype=np.int64)
Data['SI Z-score'][index6f] = zscore(Data['SI'][index6f])
Data['SI Z-score'][index6s] = zscore(Data['SI'][index6s])
SubData = SubDict(Data, Data.keys(), np.concatenate([index6s, index6f]))

# SI
uniq_day = ['Day 1', 'Day 2', 'Day 3', 'Day 4',
            'Day 5', 'Day 6', 'Day 7', 'Day 8',
            'Day 9', '>=Day 10']

idx = np.concatenate([np.where(SubData['Training Day'] == day)[0] for day in uniq_day])
SubData = SubDict(SubData, SubData.keys(), idx)

colors = sns.color_palette("rocket", 3)
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3))
ax1, ax2 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True), Clear_Axes(axes[1], close_spines=['top', 'right', 'left'], ifxticks=True)
idx = np.where((SubData['Stage'] == 'Stage 1')&(SubData['Maze Type'] == 'Maze 1'))[0]
SubData1 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='SI',
    data=SubData1,
    hue='MiceID',
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=4,
    err_style='bars',
    err_kws={'capsize':3},
    linewidth=1,
    ax=ax1
)

idx = np.where((SubData['Stage'] == 'Stage 2')&(SubData['Maze Type'] == 'Maze 1'))[0]
SubData2 = SubDict(SubData, SubData.keys(), idx)
sns.lineplot(
    x='Training Day',
    y='SI',
    data=SubData2,
    hue='MiceID',
    palette=colors,
    marker='o',
    markeredgecolor=None,
    markersize=4,
    err_style='bars',
    err_kws={'capsize':3},
    linewidth=1,
    ax=ax2
)
plt.tight_layout()
plt.savefig(join(loc, 'SI [pc-mouse].png'), dpi=600)
plt.savefig(join(loc, 'SI [pc-mouse].svg'), dpi=600)
plt.close()


'''
# Fig0017-1: SI vs training day. ---------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette('rocket', 3)
sns.lineplot(x = 'Training Day',y = 'SI', data = Data, hue = 'Maze Type', ax = ax, palette=colors,
             err_style = 'bars', err_kws = {'elinewidth':1, 'capsize':3, 'capthick':1})
ax.set_yticks(np.linspace(0,1.2,7))
plt.tight_layout()
plt.savefig(os.path.join(p,'[lineplot] SI.svg'), dpi = 2400)
plt.savefig(os.path.join(p,'[lineplot] SI.png'), dpi = 2400)
plt.close()

fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette('rocket', 3)
sns.barplot(x = 'Training Day',y = 'SI', data = Data, hue = 'Maze Type', ax = ax, palette=colors,
            errcolor='black', errwidth=1, capsize=0.1)
plt.tight_layout()
plt.savefig(os.path.join(p,'[barplot] SI.svg'), dpi = 2400)
plt.savefig(os.path.join(p,'[barplot] SI.png'), dpi = 2400)
plt.close()

# cross day significance test.
op_d1_idx = np.where((Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Open Field'))[0]
m1_d1_idx = np.where((Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
m2_d1_idx = np.where((Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
op_d9_idx = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Open Field'))[0]
m1_d9_idx = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1'))[0]
m2_d9_idx = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 2'))[0]

print("Day 1 vs Day 9")
print(ttest_ind(Data['SI'][op_d1_idx], Data['SI'][op_d9_idx]))
print(ttest_ind(Data['SI'][m1_d1_idx], Data['SI'][m1_d9_idx]))
print(ttest_ind(Data['SI'][m2_d1_idx], Data['SI'][m2_d9_idx]))

# Spatial Information-------------------------------------------------------------------------------------------------------
def plot_SI_lineplot(MiceID:list[str] = ['11095'], is_placecell:bool = False):

    Author: YAO Shuyang
    Date: Jan 26, 2023
    Note: This function is to plot linegraph of SI changing tendency in maze.

    # Read In Data:
    file_path = '0017' if is_placecell == False else '0017-ac-'
    if os.path.exists(os.path.join(figdata, file_path+'data.pkl')) == False:
        Data = DataFrameEstablish(variable_names = ['Cell','SI'], f = f1, function = SpatialInformation_Interface, is_placecell = is_placecell,
                                  file_name = file_path, behavior_paradigm = 'CrossMaze')
    else:
        with open(os.path.join(figdata, file_path+'data.pkl'), 'rb') as handle:
            Data = pickle.load(handle)

    # Get SubData
    idx = np.array([], dtype = np.int64)
    mice = ''
    cell_type = 'Place Cells Only' if is_placecell else 'All Cells'
    for m in MiceID:
        idx = np.concatenate([idx, np.where(Data['MiceID'] == m)[0]])
        # Get the title of the figure and the name of the figure file.
        if m == MiceID[0]:
            mice = mice+m
        else: mice = mice+'&'+m            
    
    SubData = SubDict(Data, keys = ['MiceID', 'Training Day', 'Maze Type', 'Cell', 'SI'], idx = idx)

    # plot figure
    fs = 14

    fig = plt.figure(figsize = (5,3.75))
    ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
    sns.lineplot(x = 'Training Day',y = 'SI', data = SubData, hue = 'Maze Type', ax = ax, legend = True, err_style = 'bars', errorbar = ('ci', 95), 
                 err_kws = {'elinewidth':2, 'capsize':3, 'capthick':2})
    ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze Type', loc = 'upper left')
    #ax.xaxis.set_major_locator(x)
    ax.set_xlabel('Training Day', fontsize = fs)
    ax.set_ylabel('Spatial Information', fontsize = fs)
    ax.set_title(mice+'\n'+cell_type)
    
    ax.axis([-0.5,8.5,0,2])
    plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9,10], labels = ['P1','P2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
    plt.tight_layout()
    plt.savefig(os.path.join(p, mice+'-'+cell_type+'.svg'), dpi = 600)
    plt.savefig(os.path.join(p, mice+'-'+cell_type+'.png'), dpi = 600)
    plt.close()

    # T test for increasing
    day1_idx_maze1 = np.where((Data['Training Day'] == 'Pre 2')&(Data['Maze Type'] == 'Maze 1'))[0]
    day9_idx_maze1 = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1'))[0]
    _, pvalue1 = scipy.stats.ttest_ind(Data['SI'][day1_idx_maze1], Data['SI'][day9_idx_maze1], nan_policy = 'omit', alternative = 'less')

    day1_idx_maze2 = np.where((Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
    day9_idx_maze2 = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 2'))[0]
    _, pvalue2 = scipy.stats.ttest_ind(Data['SI'][day1_idx_maze2], Data['SI'][day9_idx_maze2], nan_policy = 'omit', alternative = 'less')

    # open field maintain t test
    day1_idx_op = np.where((Data['Training Day'] == 'Pre 1')&(Data['Maze Type'] == 'Open Field'))[0]
    day9_idx_op = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Open Field'))[0]
    _, pvalue0 = scipy.stats.ttest_ind(Data['SI'][day1_idx_op], Data['SI'][day9_idx_op], nan_policy = 'omit', alternative = 'less')
    print('Open Field:',pvalue0)
    print('Maze 1:',pvalue1)
    print('Maze 2',pvalue2)


plot_SI_lineplot(MiceID=['11095'],is_placecell=True)
plot_SI_lineplot(MiceID=['11092'],is_placecell=True)

plot_SI_lineplot(MiceID=['11095'],is_placecell=False)
plot_SI_lineplot(MiceID=['11092'],is_placecell=False)
'''
#plot_SI_lineplot(MiceID=['11092','11095'],is_placecell=True)
#plot_SI_lineplot(MiceID=['11092','11095'],is_placecell=False)

