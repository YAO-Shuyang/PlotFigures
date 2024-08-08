from mylib.statistic_test import *
from mylib import LocTimeCurveAxes, RateMapAxes, TraceMapAxes, LinearizedRateMapAxes
from matplotlib import gridspec


code_id = '0004 - Sample Cells'
loc = os.path.join(figpath, code_id, 'reverse')
mkdir(loc)

def plot_sample_cell(trace, i, save_loc: str, file_name: str):
    fig = plt.figure(figsize=(9, 4))
    grid = gridspec.GridSpec(nrows=4, ncols=3, width_ratios=[3, 3, 3])

    ax1 = fig.add_subplot(grid[0:2, 0])   # First two rows, first column
    ax3 = fig.add_subplot(grid[0:2, 1])   # Last two rows, first column
    ax2 = fig.add_subplot(grid[2:, 0])     # First row, second column
    ax4 = fig.add_subplot(grid[2:, 1])    # Second row onwards, second column
    ax5 = fig.add_subplot(grid[0, 2])    # Second row onwards, second column
    ax6 = fig.add_subplot(grid[1:, 2])
    
    maze_type = trace['maze_type']
    smooth_map_cis, smooth_map_trs = cp.deepcopy(trace['cis']['smooth_map_all'][i, :]), cp.deepcopy(trace['trs']['smooth_map_all'][i, :])
    old_map_cis, old_map_trs = cp.deepcopy(trace['cis']['old_map_clear'][i, :]), cp.deepcopy(trace['trs']['old_map_clear'][i, :])
    behav_pos = cp.deepcopy(trace['correct_pos'])
    behav_time = cp.deepcopy(trace['correct_time'])
    behav_nodes = spike_nodes_transform(trace['correct_nodes'], nx=12)
    ms_time = cp.deepcopy(trace['tot']['ms_time_behav'])
    spike_nodes = spike_nodes_transform(trace['tot']['spike_nodes'], nx=12)
    spikes = cp.deepcopy(trace['tot']['Spikes'][i, :])
    place_fields_cis, place_fields_trs = trace['cis']['place_field_all_multiday'][i], trace['trs']['place_field_all_multiday'][i]
    
    _, im, cbar = RateMapAxes(
        ax=ax1,
        content=smooth_map_cis,
        maze_type=maze_type,
        title=trace['cis']['SI_all'][i],
        maze_args={'color':'white', 'linewidth': 0.5}
    )
    cbar.outline.set_visible(False)
    _, im, cbar = RateMapAxes(
        ax=ax2,
        content=smooth_map_trs,
        maze_type=maze_type,
        title=trace['trs']['SI_all'][i],
        maze_args={'color':'white', 'linewidth': 0.5}
    )
    cbar.outline.set_visible(False)
    ax1.set_aspect("equal")
    ax1.axis([-1, 48, 48, -1])
    ax2.set_aspect("equal")
    ax2.axis([-1, 48, 48, -1])
    
    
    if trace['paradigm'] == 'ReverseMaze': 
        beg, end = LapSplit(trace, "ReverseMaze")
        if trace['MiceID'] in [10212, 10209]:
            cis_idx = np.concatenate([np.arange(beg[i], end[i]) for i in range(1, beg.shape[0], 2)])
            trs_idx = np.concatenate([np.arange(beg[i], end[i]) for i in range(0, beg.shape[0], 2)])
        else:
            cis_idx = np.concatenate([np.arange(beg[i], end[i]) for i in range(0, beg.shape[0], 2)])
            trs_idx = np.concatenate([np.arange(beg[i], end[i]) for i in range(1, beg.shape[0], 2)])
    else:
        beg, end = LapSplit(trace, "HairpinMaze")
        cis_idx = np.concatenate([np.arange(beg[i], end[i]) for i in range(0, beg.shape[0], 2)])
        trs_idx = np.concatenate([np.arange(beg[i], end[i]) for i in range(1, beg.shape[0], 2)])
        
    TraceMapAxes(
        ax=ax3,
        trajectory=behav_pos[cis_idx, :],
        behav_time=behav_time[cis_idx],
        spikes=cp.deepcopy(trace['cis']['Spikes'][i, :]),
        spike_time=cp.deepcopy(trace['cis']['ms_time_behav']),
        maze_type=maze_type,
        maze_kwargs={'color':'brown', 'linewidth': 0.5},
        traj_kwargs={'linewidth': 0.5},
        markersize=2
    )
    ax3.set_aspect("equal")
    ax3.axis([-1, 48, 48, -1])
    TraceMapAxes(
        ax=ax4,
        trajectory=behav_pos[trs_idx, :],
        behav_time=behav_time[trs_idx],
        spikes=cp.deepcopy(trace['trs']['Spikes'][i, :]),
        spike_time=cp.deepcopy(trace['trs']['ms_time_behav']),
        maze_type=maze_type,
        maze_kwargs={'color':'brown', 'linewidth': 0.5},
        traj_kwargs={'linewidth': 0.5},
        markersize=2
    )
    ax4.set_aspect("equal")
    ax4.axis([-1, 48, 48, -1])

    MTOP = gaussian_smooth_matrix1d(1000, window = 20, sigma=3, folder=0.1)
    CP = cp.deepcopy(correct_paths[int(maze_type)])
    LinearizedRateMapAxes(
        ax=ax5,
        content=old_map_cis,
        maze_type=maze_type,
        M=MTOP,
        linewidth=0.5,
        color = sns.color_palette("Blues", 5)[1],
        alpha = 0.9
    )
    LinearizedRateMapAxes(
        ax=ax5,
        content=old_map_trs,
        maze_type=maze_type,
        M=MTOP,
        linewidth=0.5,
        color = sns.color_palette("YlOrRd", 5)[1],
        alpha = 0.9
    )
    y_max = np.nanmax([np.nanmax(old_map_cis), np.nanmax(old_map_trs)])
    ax5.set_xlim([0, len(CP)+1])
    ax5.set_ylim([-y_max*0.16, y_max*1.02])
    ax5.set_yticks([0, y_max])

    lef, rig = np.zeros(len(place_fields_cis.keys()), dtype=np.float64), np.zeros(len(place_fields_cis.keys()), dtype=np.float64)
    for j, k in enumerate(place_fields_cis.keys()):
        father_field = SF2FF(place_fields_cis[k])
        lef[j], rig[j] = set_range(maze_type=maze_type, field=father_field)
    lef = np.sort(lef + 0.5)
    rig = np.sort(rig + 1.5)
        
    for k in range(lef.shape[0]):
        ax5.plot([lef[k], rig[k]], [-y_max*0.09, -y_max*0.09], color = sns.color_palette("Blues", 9)[1], linewidth=0.5)
        ax6.fill_betweenx(y=[0, np.nanmax(behav_time)/1000], x1=lef[k], x2=rig[k], color=sns.color_palette("Blues", 5)[1], alpha=0.5, edgecolor=None)
    
    for k, center in enumerate(place_fields_cis.keys()):
        for fd in place_fields_cis[center]:
            if fd in incorrect_paths[(maze_type, 48)]:
                continue
            y, x = (fd-1)//48-0.5, (fd-1)%48-0.5
            ax3.fill_betweenx([y, y+1], x1=x, x2=x+1, color = sns.color_palette("Blues", 5)[1], alpha=0.5, edgecolor=None)

    lef, rig = np.zeros(len(place_fields_trs.keys()), dtype=np.float64), np.zeros(len(place_fields_trs.keys()), dtype=np.float64)
    for j, k in enumerate(place_fields_trs.keys()):
        father_field = SF2FF(place_fields_trs[k])
        lef[j], rig[j] = set_range(maze_type=maze_type, field=father_field)
    lef = np.sort(lef + 0.5)
    rig = np.sort(rig + 1.5)
        
    for k in range(lef.shape[0]):
        ax5.plot([lef[k], rig[k]], [-y_max*0.15, -y_max*0.15], color = sns.color_palette("YlOrRd", 9)[1], linewidth=0.5)
        ax6.fill_betweenx(y=[0, np.nanmax(behav_time)/1000], x1=lef[k], x2=rig[k], color=sns.color_palette("YlOrRd", 9)[1], alpha=0.5, edgecolor=None)
    
    for k, center in enumerate(place_fields_trs.keys()):
        for fd in place_fields_trs[center]:
            if fd in incorrect_paths[(maze_type, 48)]:
                continue
            y, x = (fd-1)//48-0.5, (fd-1)%48-0.5
            ax4.fill_betweenx([y, y+1], x1=x, x2=x+1, color = sns.color_palette("YlOrRd", 9)[1], alpha=0.5, edgecolor=None)
    
    #indices = np.where(frame_labels==1)[0]
    #print(indices.shape[0], frame_labels.shape[0])
    LocTimeCurveAxes(
        ax=ax6,
        behav_time=behav_time,#[indices],
        spikes=spikes,
        spike_time=ms_time,
        maze_type=maze_type,
        behav_nodes=behav_nodes,#[indices],
        line_kwargs={'linewidth': 0.1, 'color': 'gray'},
        bar_kwargs={'markeredgewidth': 0.2, 'markersize':3}
    )
    ax6.set_xlim([0, len(CP)+0.5])
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


hairpin_loc = join(figpath, code_id, 'hairpin')
mkdir(hairpin_loc)

i = np.where((f4['MiceID'] == 10209)&(f4['date'] == 20230617)&(f4['session'] == 1))[0][0]
plot_cells(f=f4, i=i, cells=[75, 80, 81, 32, 15, 20], save_loc=hairpin_loc)

i = np.where((f4['MiceID'] == 10209)&(f4['date'] == 20230618)&(f4['session'] == 1))[0][0]
plot_cells(f=f4, i=i, cells=[57, 50], save_loc=hairpin_loc)

i = np.where((f4['MiceID'] == 10227)&(f4['date'] == 20231022)&(f4['session'] == 1))[0][0]
plot_cells(f=f4, i=i, cells=[33, 34, 35, 39, 40, 43], save_loc=hairpin_loc)

i = np.where((f4['MiceID'] == 10209)&(f4['date'] == 20230619)&(f4['session'] == 1))[0][0]
plot_cells(f=f4, i=i, cells=[6, 8, 9, 11, 85, 39, 11, 17], save_loc=hairpin_loc)

i = np.where((f4['MiceID'] == 10227)&(f4['date'] == 20231023)&(f4['session'] == 1))[0][0]
plot_cells(f=f4, i=i, cells=[1, 25, 28, 35, 37, 43, 54], save_loc=hairpin_loc)

i = np.where((f4['MiceID'] == 10224)&(f4['date'] == 20231023)&(f4['session'] == 1))[0][0]
plot_cells(f=f4, i=i, cells=[7, 15, 23], save_loc=hairpin_loc)

i = np.where((f4['MiceID'] == 10212)&(f4['date'] == 20230619)&(f4['session'] == 1))[0][0]
plot_cells(f=f4, i=i, cells=[13], save_loc=hairpin_loc)


i = np.where((f3['MiceID'] == 10209)&(f3['date'] == 20230619)&(f3['session'] == 1))[0][0]
plot_cells(f=f3, i=i, cells=[16, 24, 34], save_loc=loc)

i = np.where((f3['MiceID'] == 10212)&(f3['date'] == 20230619)&(f3['session'] == 1))[0][0]
plot_cells(f=f3, i=i, cells=[41, 53, 15, 18, 19, 23, 24, 37], save_loc=loc)

i = np.where((f3['MiceID'] == 10227)&(f3['date'] == 20231023)&(f3['session'] == 1))[0][0]
plot_cells(f=f3, i=i, cells=[44, 1, 2, 4, 5, 7, 11, 20], save_loc=loc)

i = np.where((f3['MiceID'] == 10224)&(f3['date'] == 20231023)&(f3['session'] == 1))[0][0]
plot_cells(f=f3, i=i, cells=[23, 21, 15, 8, 3, 2, 1], save_loc=loc)



