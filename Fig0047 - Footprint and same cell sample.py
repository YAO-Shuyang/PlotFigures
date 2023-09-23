from mylib.statistic_test import *

code_id = "0047 - Footprint same cell sample"
loc = join(figpath, code_id)
mkdir(loc)


dates = ['20220820', '20220822', '20220824', '20220826',
         '20220828', '20220830']
foot_print_maze1 = [join(r"E:\Data\FigData\0046 - Footprint & New recruit cells\Maze1-footprint", 'SFP'+d+'.mat') for d in dates]
foot_print_maze2 = [join(r"E:\Data\FigData\0046 - Footprint & New recruit cells\Maze2-footprint", 'SFP'+d+'.mat') for d in dates]


def add_loc_time_cp(trace, i, ax: Axes):
    nodes = spike_nodes_transform(trace['behav_nodes'], nx = 12)
    behav_time = trace['behav_time']/1000
    
    ms_time_behav = trace['ms_time_behav']/1000
    spikes = trace['Spikes'][i-1]
    spike_nodes = spike_nodes_transform(trace['spike_nodes'], nx = 12)
    
    idx = np.where(spikes == 1)[0]
    spike_time = ms_time_behav[idx]
    spike_nodes = spike_nodes[idx]
    
    CP = CorrectPath_maze_1 if trace['maze_type'] == 1 else CorrectPath_maze_2
    DP = DecisionPoint1_Linear if trace['maze_type'] == 1 else DecisionPoint2_Linear
    XO = xorder1-1 if trace['maze_type'] == 1 else xorder2-1
    LE = len(DP)
    
    x = NodesReorder(nodes, maze_type=trace['maze_type']) + np.random.rand(nodes.shape[0]) - 0.5
    xs = NodesReorder(spike_nodes, maze_type=trace['maze_type']) + np.random.rand(spike_nodes.shape[0]) - 0.5
    
    ax = Clear_Axes(ax, close_spines=['right', 'top'], ifyticks=True)
    
    # Edges of decision area band.
    MAX = int(behav_time[-1])+1
    DPO = NodesReorder(DP, maze_type=trace['maze_type'])
    x_l = DPO-0.5
    x_r = DPO+0.5
        
    # Decision area text labelself._xl1
    ax_shadow(ax=ax, x1_list=x_l, x2_list=x_r, y = np.linspace(0, 10000, 2), 
              colors = np.repeat('gray', LE), edgecolor = None)

    ax.set_xticks(DPO, labels = DP, fontsize = 6, rotation=90)
    
    # Plot top band
    cmap = matplotlib.colormaps['rainbow']
    maze_seg = MazeSegments(maze_type=trace['maze_type'], path_type = 'cp')
    seg_num = len(maze_seg.keys())
    colors = cmap(np.linspace(0, 1, seg_num))
    k = 0
    for j in range(len(x_l)-1):
        if x_l[j+1] == x_r[j]:
            continue
        ax.fill_between(x = np.linspace(x_r[j], x_l[j+1], 2), 
                        y1 = MAX*1.05, y2 = MAX*1.1, ec = None, 
                        color = colors[k])
        k += 1
    
    ax.plot(x, behav_time, '.', markersize = 2, color = 'k', markeredgewidth = 0)
    ax.plot(xs, spike_time, '|', markersize = 3, color = 'red', markeredgewidth = 0.5)
    ax.set_yticks(ColorBarsTicks(peak_rate=MAX, is_auto=True, tick_number=4))
    ax.axis([0, CP.shape[0], 0, MAX*1.1])
    
    return ax

def get_footprint(SFP_loc:str = None):
    if os.path.exists(SFP_loc):
        with h5py.File(SFP_loc, 'r') as f:
            sfp = np.array(f['SFP'])
            
    return sfp

def plot_figure(trace_set: list[dict], 
                SFP_list: list[str], 
                save_loc: str = None, 
                file_name: str = None,
                cells: list[int] = None):
    assert len(trace_set) == 6 and len(SFP_list) == 6 and len(cells) == 6
    
    fig, axes = plt.subplots(ncols=6, nrows=3, figsize=(6*2, 6.5), gridspec_kw={'height_ratios':[2, 3.5, 1]})
    for i in range(6):
        cell = cells[i]
        sfp = get_footprint(SFP_list[i])
        footprint = np.nanmax(sfp,axis = 2)
            
        ax = Clear_Axes(axes[0, i])
        ax.imshow(footprint, cmap = 'hot')
        x, y = np.where(sfp[:, :, cell-1] == np.nanmax(sfp[:, :, cell-1]))
        ax.plot(y, x, 'o', color = 'yellow', markersize = 3)
        ax.set_aspect('equal')
        ax.axis([y-15.5, y+15.5, x-15.5, x+15.5])
        
        add_loc_time_cp(trace_set[i], cell, ax=axes[1,i])
        
        ax = Clear_Axes(axes[2,i])
        DrawMazeProfile(maze_type=trace_set[i]['maze_type'], nx = 48, axes=ax)
        im = ax.imshow(np.reshape(trace_set[i]['smooth_map_all'][cell-1], [48, 48]), cmap='jet')
        cbar = plt.colorbar(im, ax = ax)
        cbar.set_ticks([0, np.nanmax(trace_set[i]['smooth_map_all'][cell-1])])
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(join(save_loc, file_name+'.png'), dpi=1200)
    plt.savefig(join(save_loc, file_name+'.svg'), dpi=1200)
        
    plt.close()


idx = np.where((f1['date'] >= 20220820)&(f1['MiceID'] == 11095)&(f1['maze_type'] == 1))[0]
trace_set = TraceFileSet(idx, file=f1, tp=r'E:\Data\Cross_maze')
plot_figure(trace_set, foot_print_maze1, cells=[40, 91, 130, 114, 90, 128], 
            save_loc=loc, file_name='40-90-130-114-90-128')