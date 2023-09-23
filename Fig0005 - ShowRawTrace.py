from mylib.statistic_test import *
import matplotlib.gridspec as gridspec

code_id = 'Fig0005 - ShowRawTrace'
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
    ROWHEIGHT = 3
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

    fig = plt.figure(figsize = (12,9))
    gs = gridspec.GridSpec(n+3,1)
    ax = Clear_Axes(plt.subplot(gs[0:n]))
    loc_ax = Clear_Axes(plt.subplot(gs[n:n+3]), close_spines=['top','right','bottom'], ifyticks=True)

    # plot raw trace.
    t = ms_time[t_beg:t_end]
    for i in range(n):
        ax.plot(t, RawTraces[cell_idx[i], t_beg:t_end] + i*ROWHEIGHT)
        idx = np.where(Spikes_original[cell_idx[i], t_beg:t_end] == 1)[0]
        ax.plot(t[idx], np.repeat(i*ROWHEIGHT-0.5,len(idx)),'.',color = 'black', markersize = 2)
        ax.text(t[0]-dw/2, i*ROWHEIGHT, str(cell_idx[i]+1))

    YMIN = np.nanmin(RawTraces[cell_idx[0], t_beg:t_end])-1
    YMAX = np.nanmax(RawTraces[cell_idx[-1], t_beg:t_end])+n*ROWHEIGHT + ampli_bar
    

    # time bar
    ax.plot([t[0], t[0]+time_bar], [n * ROWHEIGHT, n * ROWHEIGHT], color = 'black')
    ax.text(x = t[0]+time_bar/2, y = n * ROWHEIGHT+0.3, ha = 'center', s = str(int(time_bar/1000))+' s', fontsize = 12)
    ax.plot([t[0], t[0]], [n * ROWHEIGHT, n * ROWHEIGHT + ampli_bar], color = 'black')
    ax.text(x = t[0]+1000, y = n * ROWHEIGHT + ampli_bar, s = 'dF/F = 3', fontsize = 12)
    ax.axis([t[0]-dw,t[-1]+dw,YMIN, YMAX])
    loc_ax.set_xlabel("Time")

    # Plot loc-time curve
    spike_nodes_original[np.where(np.isnan(spike_nodes_original))[0]] = 1
    old_node = spike_nodes_transform(spike_nodes_original, nx = 12)[t_beg:t_end]
    #old_node = spike_nodes_transform(spike_nodes, nx = 12)[t_beg:t_end]
    reorder_node = np.zeros(old_node.shape[0], dtype = np.float64)
    order = xorder1 if trace['maze_type'] == 1 else xorder2
    for i in tqdm(range(old_node.shape[0])):
        reorder_node[i] = np.where(order == old_node[i])[0][0]+1 + np.random.rand()-0.5
    
    
    divi_line = len(CorrectPath_maze_1)+0.5 if trace['maze_type'] == 1 else len(CorrectPath_maze_2)+0.5
    loc_ax.plot(t, reorder_node, '.', color = 'gray')
    loc_ax.set_yticks([1,36,72,108,144])
    loc_ax.axis([t[0]-dw, t[-1]+dw, 0.5,144.5])
    loc_ax.axhline(divi_line, ls = '--', color = 'black')
    loc_ax.set_ylabel('Linearized Location (bin)')

    if save_loc is None:
        plt.show()
    else:
        plt.savefig(save_loc+'.svg', dpi = 600)
        plt.savefig(save_loc+'.png', dpi = 600)

# 2,4,12,17,22,24,28,31,34,40
with open(f1['Trace File'][83], 'rb') as handle:
    trace = pickle.load(handle)
    
print(trace['ms_time'][10000]-trace['ms_time'][1000])
mkdir(os.path.join(p, str(trace['MiceID']), str(trace['date'])))
plot_rawtrace(trace=trace, cell_idx=np.array([4,12,17,22,24,28,31,34,40])-1, t_beg = 1000, t_end=10000, save_loc = os.path.join(p, str(trace['MiceID']), str(trace['date']), 'Maze '+str(trace['maze_type'])))
