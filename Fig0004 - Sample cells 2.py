from mylib.statistic_test import *
from mylib import LocTimeCurveAxes, RateMapAxes, TraceMapAxes, LinearizedRateMapAxes
from matplotlib import gridspec


code_id = '0004 - Sample Cells'
loc = os.path.join(figpath, code_id)
mkdir(loc)

def plot_sample_cell(trace, i, save_loc: str, file_name: str):
    fig = plt.figure(figsize=(5.5, 4))
    grid = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[7, 4])

    ax1 = fig.add_subplot(grid[0:2, 0])   # First two rows, first column
    ax2 = fig.add_subplot(grid[2:4, 0])   # Last two rows, first column
    ax3 = fig.add_subplot(grid[0, 1])     # First row, second column
    ax4 = fig.add_subplot(grid[1:, 1])    # Second row onwards, second column
    
    maze_type = trace['maze_type']
    smooth_map = cp.deepcopy(trace['smooth_map_all'][i, :])
    old_map = cp.deepcopy(trace['old_map_clear'][i, :])
    behav_pos = cp.deepcopy(trace['correct_pos'])
    behav_time = cp.deepcopy(trace['correct_time'])
    behav_nodes = spike_nodes_transform(trace['correct_nodes'], nx=12)
    ms_time = cp.deepcopy(trace['ms_time_behav'])
    spike_nodes = spike_nodes_transform(trace['spike_nodes'], nx=12)
    spikes = cp.deepcopy(trace['Spikes'][i, :])
    place_fields = cp.deepcopy(trace['place_field_all_multiday'][i])
    
    _, im, cbar = RateMapAxes(
        ax=ax1,
        content=smooth_map,
        maze_type=maze_type,
        title=trace['SI_all'][i],
        maze_args={'color':'white', 'linewidth': 0.5}
    )
    cbar.outline.set_visible(False)
    ax1.set_aspect("equal")
    ax1.axis([-1, 48, 48, -1])   
     
    TraceMapAxes(
        ax=ax2,
        trajectory=behav_pos,
        behav_time=behav_time,
        spikes=spikes,
        spike_time=ms_time,
        maze_type=maze_type,
        maze_kwargs={'color':'brown', 'linewidth': 0.5},
        traj_kwargs={'linewidth': 0.5},
        markersize=2
    )
    ax2.set_aspect("equal")
    ax2.axis([-1, 48, 48, -1])


    MTOP = gaussian_smooth_matrix1d(1000, window = 20, sigma=3, folder=0.1)
    CP = cp.deepcopy(correct_paths[int(maze_type)])
    LinearizedRateMapAxes(
        ax=ax3,
        content=old_map,
        maze_type=maze_type,
        M=MTOP,
        linewidth=0.5
    )
    ax3.set_xlim([0, len(CP)+1])
    y_max = np.nanmax(old_map)

    lef, rig = np.zeros(len(place_fields.keys()), dtype=np.float64), np.zeros(len(place_fields.keys()), dtype=np.float64)
    colors = sns.color_palette("rainbow", 10) if lef.shape[0] < 8 else sns.color_palette("rainbow", lef.shape[0]+2)
    for j, k in enumerate(place_fields.keys()):
        father_field = SF2FF(place_fields[k])
        lef[j], rig[j] = set_range(maze_type=maze_type, field=father_field)
    lef = np.sort(lef + 0.5)
    rig = np.sort(rig + 1.5)
        
    for k in range(lef.shape[0]):
        ax3.plot([lef[k], rig[k]], [-y_max*0.09, -y_max*0.09], color = colors[k+2], linewidth=0.5)
        ax4.fill_betweenx(y=[0, np.nanmax(behav_time)/1000], x1=lef[k], x2=rig[k], color=colors[k+2], alpha=0.5, edgecolor=None)
    
    for k, center in enumerate(place_fields.keys()):
        for fd in place_fields[center]:
            if fd in incorrect_paths[(maze_type, 48)]:
                continue
            y, x = (fd-1)//48-0.5, (fd-1)%48-0.5
            ax2.fill_betweenx([y, y+1], x1=x, x2=x+1, color = colors[k+2], alpha=0.5, edgecolor=None)
    
    frame_labels = get_spike_frame_label(
        ms_time=cp.deepcopy(trace['correct_time']), 
        spike_nodes=cp.deepcopy(trace['correct_nodes']),
        trace=trace, 
        behavior_paradigm='CrossMaze',
        window_length = 1
    )
    
    #indices = np.where(frame_labels==1)[0]
    #print(indices.shape[0], frame_labels.shape[0])
    LocTimeCurveAxes(
        ax=ax4,
        behav_time=behav_time,#[indices],
        spikes=spikes,
        spike_time=ms_time,
        maze_type=maze_type,
        behav_nodes=behav_nodes,#[indices],
        line_kwargs={'markeredgewidth': 0, 'markersize': 0.6, 'color': 'black'},
        bar_kwargs={'markeredgewidth': 0.2, 'markersize':3}
    )
    ax4.set_xlim([0, len(CP)+0.5])
    plt.tight_layout()
    plt.savefig(join(save_loc, file_name+'.png'), dpi=600)
    plt.savefig(join(save_loc, file_name+'.svg'), dpi=600)
    plt.close()

def plot_cells(f: pd.DataFrame, i: int, cells: list, save_loc: str):
    if exists(f['Trace File'][i]) == False:
        return
    
    with open(f['Trace File'][i], 'rb') as handle:
        trace = pickle.load(handle)
    
    file_name = 'L'+str(i)+'-'+str(int(f['MiceID'][i]))+'-'+'Maze '+str(int(f['maze_type'][i]))+'-'
    for cell in cells:
        plot_sample_cell(trace=trace, i=cell-1, save_loc=save_loc, file_name=file_name+str(cell))

loc = os.path.join(figpath, "0004 - Sample Cells to CC")
mkdir(loc)
i = np.where((f1['MiceID'] == 10224)&(f1['date'] == 20230930)&(f1['session'] == 3))[0][0]
plot_cells(f=f1, i=i, cells=[261], save_loc=loc)

i = np.where((f1['MiceID'] == 10224)&(f1['date'] == 20230928)&(f1['session'] == 2))[0][0]
plot_cells(f=f1, i=i, cells=[5, 162], save_loc=loc)

i = np.where((f1['MiceID'] == 10224)&(f1['date'] == 20230928)&(f1['session'] == 3))[0][0]
plot_cells(f=f1, i=i, cells=[3, 111], save_loc=loc)

i = np.where((f1['MiceID'] == 10227)&(f1['date'] == 20230930)&(f1['session'] == 2))[0][0]
plot_cells(f=f1, i=i, cells=[280, 140], save_loc=loc)

i = np.where((f1['MiceID'] == 10227)&(f1['date'] == 20230930)&(f1['session'] == 3))[0][0]
plot_cells(f=f1, i=i, cells=[85, 165], save_loc=loc)

i = np.where((f1['MiceID'] == 10209)&(f1['date'] == 20230728)&(f1['session'] == 2))[0][0]
plot_cells(f=f1, i=i, cells=[126, 76], save_loc=loc)

i = np.where((f1['MiceID'] == 10209)&(f1['date'] == 20230728)&(f1['session'] == 3))[0][0]
plot_cells(f=f1, i=i, cells=[107, 36], save_loc=loc)

i = np.where((f1['MiceID'] == 10212)&(f1['date'] == 20230724)&(f1['session'] == 2))[0][0]
plot_cells(f=f1, i=i, cells=[101, 7], save_loc=loc)

i = np.where((f1['MiceID'] == 10212)&(f1['date'] == 20230724)&(f1['session'] == 3))[0][0]
plot_cells(f=f1, i=i, cells=[145, 33], save_loc=loc)

i = np.where((f1['MiceID'] == 10212)&(f1['date'] == 20230728)&(f1['session'] == 3))[0][0]
plot_cells(f=f1, i=i, cells=[9, 64, 65], save_loc=loc)
"""
new_emerge_loc = join(loc, 'newly emerge fields sample')
mkdir(new_emerge_loc)
dissapear_loc = join(loc, 'dissapear fields sample')
mkdir(dissapear_loc)
gap_loc = join(loc, 'gap sample')
mkdir(gap_loc)
switch_loc = join(loc, 'switch fields')
mkdir(switch_loc)

# 10227 Maze 1, 0930, i = 543
plot_cells(f=f1, i=543, cells=[4, 9, 27, 79, 252, 249, 306], save_loc=loc)

# 10224 Maze 1, 0930, i = 539
plot_cells(f=f1, i=539, cells=[1, 2, 5, 17, 20, 31, 77, 152, 153, 189, 215, 247, 362], save_loc=loc)

# 11095, Maze 1, 0830, i = 85
plot_cells(f=f1, i=85, cells=[2], save_loc=loc)

# 11095, Maze 1, 0828, i = 77
plot_cells(f=f1, i=77, cells=[2, 1, 13,3], save_loc=loc)

# 11092, Maze 1, 0828, i = 76
plot_cells(save_loc=loc, f=f1, i=76, cells=[4])

# 11092, Maze 2, 0826, i = 70
plot_cells(save_loc=loc, f=f1, i=70, cells=[10])

# 10209, Maze 2, 0726, i = 344
plot_cells(save_loc=loc, f=f1, i=344, cells=[50, 30, 26])
# 10209, Maze 1, 0726, i = 343
plot_cells(save_loc=loc, f=f1, i=343, cells=[1, 9, 44])
# 10212, Maze 2, 0724, i = 340
plot_cells(save_loc=loc, f=f1, i=340, cells=[5, 6, 13, 14])
# 10212, Maze 1, 0724, i = 339
plot_cells(save_loc=loc, f=f1, i=339, cells=[13, 21, 88])
# 10212, Maze 1, 0728, i = 361
plot_cells(save_loc=loc, f=f1, i=361, cells=[22, 35, 96, 223])

# 11095, Maze 1, 0826, i = 69
plot_cells(f=f1, i=69, cells=[16, 33], save_loc=dissapear_loc)
plot_cells(save_loc=switch_loc, f=f1, i=69, cells=[56, 127])
plot_cells(save_loc=gap_loc, f=f1, i=69, cells=[50, 53])
# 11095, Maze 2, 0830, i = 87
plot_cells(f=f1, i=87, cells=[39, 115], save_loc=dissapear_loc)
"""