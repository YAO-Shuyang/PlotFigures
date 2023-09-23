from mylib.statistic_test import *

code_id = '0044 - In-session Population Vector'
loc = join(figpath, code_id)
mkdir(loc)

lines = np.where((f1['date'] >= 20220813)&(f1['date']!=20220814))[0]
if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Std OEC', 'Mean OEC', 'CP Std OEC', 'CP Mean OEC', 'IP Std OEC', 'IP Mean OEC',
                                                 'Std FSC', 'Mean FSC', 'CP Std FSC', 'CP Mean FSC', 'IP Std FSC', 'IP Mean FSC'], 
                              f = f1, function = PVCorrelations_Interface, file_idx=lines,
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        

def get_vector(map1, map2):
    SpatialPVector = np.zeros(map1.shape[1], np.float64)
    for i in range(map1.shape[1]):
        SpatialPVector[i], _ = pearsonr(map1[:, i], map2[:, i])
        
    return SpatialPVector

def plot_figure(map1, map2, maze_type, save_loc = loc, file_name = None):

    SpatialPVector = get_vector(map1, map2)
    
    fig = plt.figure(figsize=(8,6))
    ax = Clear_Axes(plt.axes())
    DrawMazeProfile(maze_type=maze_type, axes=ax, color='Black', linewidth=1, nx=48)
    im = ax.imshow(np.reshape(SpatialPVector, [48,48]), vmin=-1, vmax = 1, cmap='jet')
    plt.colorbar(im, ax=ax)
    ax.axis([-0.6,47.6,-0.6, 47.6])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    plt.savefig(join(save_loc, file_name+'.png'), dpi = 600)
    plt.savefig(join(save_loc, file_name+'.svg'), dpi = 600)
    plt.close()
    
    return 

"""
maze1_corr = np.zeros((9, 2304))
maze2_corr = np.zeros((9, 2304))

idx = np.where((f1['date'] >= 20220813)&(f1['date'] != 20220814)&(f1['maze_type']==1)&(f1['MiceID']==11095))[0]

for n, i in enumerate(idx):
    
    if exists(f1['Trace File'][i]) == False:
        continue
    
    trace = read_trace(f1['Trace File'][i])
    
    if 'laps' not in trace.keys():
        trace = CrossLapsCorrelation(trace, behavior_paradigm = trace['paradigm'])
        trace = OldMapSplit(trace)
    if trace['laps'] == 1:
        continue
    if 'fir_sec_corr' not in trace.keys():
        trace = half_half_correlation(trace)
    if 'odd_even_corr' not in trace.keys():
        trace = odd_even_correlation(trace)
        
    id = np.where(trace['is_placecell'] == 1)[0]
        
    maze1_corr[n, :] = get_vector(trace['smooth_map_odd'][id, :], trace['smooth_map_evn'][id, :])
        
Mat = np.zeros((9,9), dtype = np.float64)
for i in range(9):
    for j in range(9):
        non_nan_idx = np.where((np.isnan(maze1_corr[i, :])==False)&(np.isnan(maze1_corr[j, :])==False))[0]
        Mat[i, j], _ = pearsonr(maze1_corr[i, non_nan_idx], maze1_corr[j, non_nan_idx])

fig = plt.figure(figsize=(8,6))
ax = Clear_Axes(plt.axes())
ax.set_aspect('equal')
im = ax.imshow(Mat, vmin = -0.3, vmax = 1, cmap='jet')
plt.colorbar(im, ax=ax)
plt.savefig(join(loc, 'Maze 1-11095-CrossDayPVCorrelation.png'), dpi = 2400)
plt.savefig(join(loc, 'Maze 1-11095-CrossDayPVCorrelation.svg'), dpi = 2400)
plt.close()


idx = np.where((f1['date'] >= 20220813)&(f1['date'] != 20220814)&(f1['maze_type']==2)&(f1['MiceID']==11095))[0]

for n, i in enumerate(idx):
    
    if exists(f1['Trace File'][i]) == False:
        continue
    
    trace = read_trace(f1['Trace File'][i])
    
    if 'laps' not in trace.keys():
        trace = CrossLapsCorrelation(trace, behavior_paradigm = trace['paradigm'])
        trace = OldMapSplit(trace)
    if trace['laps'] == 1:
        continue
    if 'fir_sec_corr' not in trace.keys():
        trace = half_half_correlation(trace)
    if 'odd_even_corr' not in trace.keys():
        trace = odd_even_correlation(trace)
        
    id = np.where(trace['is_placecell'] == 1)[0]
        
    maze1_corr[n, :] = get_vector(trace['smooth_map_odd'][id, :], trace['smooth_map_evn'][id, :])
        
Mat = np.zeros((9,9), dtype = np.float64)
for i in range(9):
    for j in range(9):
        non_nan_idx = np.where((np.isnan(maze1_corr[i, :])==False)&(np.isnan(maze1_corr[j, :])==False))[0]
        Mat[i, j], _ = pearsonr(maze1_corr[i, non_nan_idx], maze1_corr[j, non_nan_idx])

fig = plt.figure(figsize=(8,6))
ax = Clear_Axes(plt.axes())
ax.set_aspect('equal')
im = ax.imshow(Mat, vmin = -0.3, vmax = 1, cmap='jet')
plt.colorbar(im, ax=ax)
plt.savefig(join(loc, 'Maze 2-11095-CrossDayPVCorrelation.png'), dpi = 2400)
plt.savefig(join(loc, 'Maze 2-11095-CrossDayPVCorrelation.svg'), dpi = 2400)
plt.close()
"""

"""    
FSC_loc = join(loc, 'FSC')
OEC_loc = join(loc, 'OEC')
mkdir(FSC_loc)
mkdir(OEC_loc)


lines = np.where((f1['date'] >= 20220813)&(f1['date']!=20220814))
for i in tqdm(range(len(f1))):
    if exists(f1['Trace File'][i]) == False:
        continue
    
    if f1['MiceID'][i] == 11094:
        continue
    
    trace = read_trace(f1['Trace File'][i])
    
    if 'laps' not in trace.keys():
        trace = CrossLapsCorrelation(trace, behavior_paradigm = trace['paradigm'])
        trace = OldMapSplit(trace)
    if trace['laps'] == 1:
        continue
    if 'fir_sec_corr' not in trace.keys():
        trace = half_half_correlation(trace)
    if 'odd_even_corr' not in trace.keys():
        trace = odd_even_correlation(trace)
    
    idx = np.where(trace['is_placecell'] == 1)[0]
    
    maze = 'Maze '+str(int(f1['maze_type'][i])) if f1['maze_type'][i] != 0 else 'Open Field'
    plot_figure(trace['smooth_map_fir'][idx, :], trace['smooth_map_sec'][idx, :], f1['maze_type'][i], FSC_loc, 
                str(int(f1['MiceID'][i]))+'-'+maze+'-session '+str(int(f1['session'][i]))+'-'+str(f1['date'][i]))
    
    plot_figure(trace['smooth_map_odd'][idx, :], trace['smooth_map_evn'][idx, :], f1['maze_type'][i], OEC_loc, 
                str(int(f1['MiceID'][i]))+'-'+maze+'-session '+str(int(f1['session'][i]))+'-'+str(f1['date'][i]))

idx = np.where(Data['Maze Type'] != '')[0]
SubData = {'Training Day': np.concatenate([Data['Training Day'][idx], Data['Training Day'][idx], Data['Training Day'][idx]]),
           'Maze Type': np.concatenate([Data['Maze Type'][idx], Data['Maze Type'][idx], Data['Maze Type'][idx]]), 
           'Mean': np.concatenate([Data['Mean OEC'][idx], Data['CP Mean OEC'][idx], Data['IP Mean OEC'][idx]]),
           'Path Type': np.concatenate([np.repeat('Total', idx.shape[0]), 
                                       np.repeat('CP', idx.shape[0]), 
                                       np.repeat('IP', idx.shape[0])])}
"""
fig = plt.figure(figsize=(4,3))
colors = sns.color_palette("rocket", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(x='Training Day', y='Mean OEC', data=Data, hue = 'Maze Type', palette=colors)
ax.set_yticks(np.linspace(0,0.7,8))
plt.savefig(join(loc, "Change of mean of Population vector.png"), dpi = 2400)
plt.savefig(join(loc, "Change of mean of Population vector.svg"), dpi = 2400)
plt.close()

Dates = ['Day '+str(i) for i in range(1,10)]
for d in Dates:
    idx1 = np.where((Data['Maze Type'] == 'Maze 1')&(Data['Training Day'] == d))[0]
    idx2 = np.where((Data['Maze Type'] == 'Maze 2')&(Data['Training Day'] == d))[0]
    idx0 = np.where((Data['Maze Type'] == 'Open Field')&(Data['Training Day'] == d))[0]

    print(d)
    print("Open - Maze 2", ttest_ind(Data['Mean OEC'][idx0], Data['Mean OEC'][idx2]))
    print("Maze 1 - Maze 2", ttest_ind(Data['Mean OEC'][idx1], Data['Mean OEC'][idx2]))
    print("Open - Maze 1", ttest_ind(Data['Mean OEC'][idx0], Data['Mean OEC'][idx1]))
    print("Done.", end='\n\n')