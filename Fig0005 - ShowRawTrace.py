from mylib.statistic_test import *
import matplotlib.gridspec as gridspec

code_id = '0005 - ShowRawTrace'
p = os.path.join(figpath, code_id)
mkdir(p)

def plot_rawtrace(trace:dict, cell_idx:list|np.ndarray, t_beg:int = 0, t_end:int = None, save_loc:str = None, time_bar:float = 20000, ampli_bar:float = 3):
    '''
    Author: Shuyang YAO
    Date: Feb 21, 2023
    
    Parameters
    ----------
    trace: dict, data struct
    cell_idx: list or numpy array (1d), the idx of cells that you select to show on the figures.
    t_beg: index of start time, int
    t_end: index of end time, int
    '''
    assert np.nanmax(cell_idx) < trace['n_neuron']

    n = len(cell_idx)
    ROWHEIGHT = 1.5
    ROWWIDTH = 12
    dw = 100
    
    RawTraces = cp.deepcopy(trace['RawTraces'])
    spike_nodes_original = cp.deepcopy(trace['spike_nodes_original'])
    idx = np.where(np.isnan(spike_nodes_original) == False)[0]
    RawTraces_behav = RawTraces[:,idx]
    Spikes = cp.deepcopy(trace['Spikes'])
    Spikes_original = cp.deepcopy(trace['Spikes_original'])
    spike_nodes = cp.deepcopy(trace['spike_nodes'])
    ms_time_behav = cp.deepcopy(trace['ms_time_behav'])
    ms_time = cp.deepcopy(trace['ms_time'])
    place_field_all = cp.deepcopy(trace['place_field_all'])

    fig = plt.figure(figsize = (12,9))
    gs = gridspec.GridSpec(n+3,1)
    ax = Clear_Axes(plt.subplot(gs[0:n]))
    loc_ax = Clear_Axes(plt.subplot(gs[n:n+3]), close_spines=['top','right','bottom'], ifyticks=True)
    
    # Plot loc-time curve
    spike_nodes_original[np.where(np.isnan(spike_nodes_original))[0]] = 1
    old_node = spike_nodes_transform(spike_nodes_original, nx = 12)[t_beg:t_end]
    #old_node = spike_nodes_transform(spike_nodes, nx = 12)[t_beg:t_end]
    reorder_node = np.zeros(old_node.shape[0], dtype = np.float64)
    order = xorder1 if trace['maze_type'] == 1 else xorder2
    for i in tqdm(range(old_node.shape[0])):
        reorder_node[i] = np.where(order == old_node[i])[0][0]+1 + np.random.rand()-0.5
        
    colors = sns.color_palette("Paired", n)
    # plot raw trace.
    t = ms_time[t_beg:t_end]
    for i in range(n):
        ax.plot(t, RawTraces[cell_idx[i], t_beg:t_end] + i*ROWHEIGHT, color = colors[i], linewidth = 0.5)
        idx = np.where(Spikes_original[cell_idx[i], t_beg:t_end] == 1)[0]
        ax.plot(t[idx], np.repeat(i*ROWHEIGHT-0.5,len(idx)),'.', color = 'black', markersize = 2, markeredgewidth=0)
        ax.text(t[0]-dw/2, i*ROWHEIGHT, str(cell_idx[i]+1))
        
        '''
        for k in place_field_all[cell_idx[i]].keys():
            father_field = spike_nodes_transform(place_field_all[cell_idx[i]][k], 12)
            idx = np.concatenate([np.where(old_node == j)[0] for j in father_field])
            rig += 1.5
            loc_ax.fill_betweenx(y=[0.5, 144.5], x1=t[j]-50, x2 = t[j]+50, alpha=0.3, edgecolor=None, linewidth=0, color = colors[i])
            ax.fill_betweenx(y=[0, i*ROWHEIGHT], x1=t[j]-50, x2 = t[j]+50, alpha=0.3, edgecolor=None, linewidth=0, color = colors[i])
        '''
    YMIN = np.nanmin(RawTraces[cell_idx[0], t_beg:t_end])-1
    YMAX = np.nanmax(RawTraces[cell_idx[-1], t_beg:t_end])+n*ROWHEIGHT + ampli_bar
    

    # time bar
    ax.plot([t[0], t[0]+time_bar], [n * ROWHEIGHT, n * ROWHEIGHT], color = 'black', linewidth = 0.5)
    ax.text(x = t[0]+time_bar/2, y = n * ROWHEIGHT+0.3, ha = 'center', s = str(int(time_bar/1000))+' s', fontsize = 12)
    ax.plot([t[0], t[0]], [n * ROWHEIGHT, n * ROWHEIGHT + ampli_bar], color = 'black', linewidth = 0.5)
    ax.text(x = t[0]+1000, y = n * ROWHEIGHT + ampli_bar, s = 'dF/F = 1.5', fontsize = 12)
    ax.axis([t[0]-dw,t[-1]+dw,YMIN, YMAX])
    loc_ax.set_xlabel("Time")

    
    divi_line = len(CorrectPath_maze_1)+0.5 if trace['maze_type'] == 1 else len(CorrectPath_maze_2)+0.5
    loc_ax.plot(t, reorder_node, '.', color = 'gray', markersize = 2, markeredgewidth=0, linewidth = 0.5)
    loc_ax.set_yticks([1,36,72,108,144])
    loc_ax.axis([t[0]-dw, t[-1]+dw, 0.5,144.5])
    loc_ax.axhline(divi_line, ls = '--', color = 'black', linewidth = 0.5)
    loc_ax.set_ylabel('Linearized Location (bin)')

    if save_loc is None:
        plt.show()
    else:
        plt.savefig(save_loc+'.svg', dpi = 600)
        plt.savefig(save_loc+'.png', dpi = 600)


def plot_rawtraces(
    trace:dict,
    cell_idx:list|np.ndarray,
    laps:list|np.ndarray,
    save_loc:str = None
):  
    trace['lap beg index'], trace['lap end index'] = LapSplit(trace)
    ROWHEIGHT = 1.5
    n = len(cell_idx)
    behav_nodes = spike_nodes_transform(trace['correct_nodes'][trace['lap beg index'][laps[0]]:trace['lap end index'][laps[-1]]+1], 12)
    behav_time = cp.deepcopy(trace['correct_time'][trace['lap beg index'][laps[0]]:trace['lap end index'][laps[-1]]+1])
    Graph = NRG[int(trace['maze_type'])]
    linearized_x = np.zeros_like(behav_nodes, np.float64)

    for i in range(behav_nodes.shape[0]):
        linearized_x[i] = Graph[int(behav_nodes[i])]
    
    linearized_x = linearized_x + np.random.rand(behav_nodes.shape[0]) - 0.5
    
    fig, axes = plt.subplots(figsize = (12,9), nrows = 2, ncols=1, gridspec_kw={'height_ratios': [n, 3]})
    ax1, ax2 = Clear_Axes(axes[0]), Clear_Axes(axes[1], close_spines=['top', 'right', 'bottom'], ifyticks=True)
    
    # Set lim of time axis
    t0, t1 = behav_time[0], behav_time[-1]
    print(t1-t0)
    ax1.set_xlim(t0, t1)
    ax2.set_xlim(t0, t1)
    ax2.set_ylim(0, 145)
    ax2.set_yticks([1, len(correct_paths[int(trace['maze_type'])])], labels = ['Entry', 'Exit'])
    
    # Plot trajectory
    ax2.plot(behav_time, linearized_x, 'o', color = 'black', linewidth = 0.5, markersize = 2, markeredgewidth = 0)
    
    # Plot raw traces for the cells.
    RawTraces = cp.deepcopy(trace['RawTraces'])
    ms_time = cp.deepcopy(trace['ms_time'])
    place_field_all = trace['place_field_all']
    Spikes = cp.deepcopy(trace['Spikes_original'])
    
    strength_bases = np.linspace(0, (n-1)*ROWHEIGHT, n)
    colors = sns.color_palette("muted", n)
    gray = sns.color_palette("Greys", 9)[1]
    for i, l in enumerate(laps):
        idx = np.where((ms_time >= trace['correct_time'][trace['lap beg index'][l]]) & (ms_time <= trace['correct_time'][trace['lap end index'][l]]))[0]
        
        for j in range(n):
            spike_idx = np.where(Spikes[cell_idx[j], idx] == 1)[0]
            ax1.plot(ms_time[idx], RawTraces[cell_idx[j], idx] + strength_bases[j], color = colors[j], linewidth = 0.5)
            ax1.plot(ms_time[idx[spike_idx]], np.repeat(strength_bases[j], len(spike_idx)), 'o', color = 'black', markersize = 2, markeredgewidth = 0)

            """
            for k in place_field_all[cell_idx[j]].keys():
                father_field = spike_nodes_transform(place_field_all[cell_idx[j]][k], 12)
                behav_idx = np.concatenate([np.where((behav_nodes == b)&(behav_time <= trace['correct_time'][trace['lap end index'][l]])&(behav_time >= trace['correct_time'][trace['lap beg index'][l]]))[0] for b in father_field])
                earliest_time, latest_time = behav_time[behav_idx][0], behav_time[behav_idx][-1]
                ax1.fill_betweenx(y=[0, strength_bases[j]], x1=earliest_time, x2 = latest_time, alpha=0.3, edgecolor=None, linewidth=0, color = colors[j])
                ax2.fill_betweenx(y=[0, len(correct_paths[int(trace['maze_type'])])+1], x1=earliest_time, x2 = latest_time, alpha=0.3, edgecolor=None, linewidth=0, color = colors[j])
            """
        # Plot raw traces that are within the gap of two laps
        if i != 0:
            idx = np.where((ms_time < trace['correct_time'][trace['lap beg index'][l]]) & (ms_time > trace['correct_time'][trace['lap end index'][laps[i-1]]]))[0]
            for j in range(n):
                ax1.plot(ms_time[idx], RawTraces[cell_idx[j], idx] + strength_bases[j], color = gray, linewidth = 0.5)
    
    if save_loc is None:
        plt.show()
    else:
        plt.savefig(save_loc+'.svg', dpi = 600)
        plt.savefig(save_loc+'.png', dpi = 600)

                

# 2,4,12,17,22,24,28,31,34,40
with open(r"E:\Data\Cross_maze\10227\20230928\session 2\trace.pkl", 'rb') as f:
    trace = pickle.load(f)

mkdir(os.path.join(p, str(trace['MiceID']), str(trace['date'])))
#plot_rawtrace(trace=trace, cell_idx=np.array([4, 6, 9, 27, 36, 43, 44, 72, 85, 86])-1, t_beg = t_beg, t_end=t_end, save_loc = os.path.join(p, str(trace['MiceID']), str(trace['date']), 'Maze '+str(trace['maze_type'])))
plot_rawtraces(trace, np.array([4, 6, 9, 27, 36, 43, 44, 72, 85, 86])-1, np.arange(14,21), save_loc = os.path.join(p, str(trace['MiceID']), str(trace['date']), 'Maze '+str(trace['maze_type'])))