import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib
from mylib.statistic_test import *

code_id = "0042 - Test Directional Firing at Decision Points"
loc = os.path.join(figpath, code_id)
mkdir(loc)

i = 88
n = 2
with open(f1['Trace File'][i], 'rb') as handle:
    trace = pickle.load(handle)

def plot_figure(trace, cell_idx, time_ons, time_end, ax):
    x, y = trace['processed_pos_new'][:, 0]/80 - 0.5, trace['processed_pos_new'][:, 1]/80 - 0.5
    behav_time = trace['correct_time']/1000
    idx_ons = np.where(behav_time <= time_ons)[0][-1]
    idx_end = np.where(behav_time <= time_end)[0][-1]
    
    ax = Clear_Axes(ax)
    DrawMazeProfile(axes = ax, color = 'brown', nx = 12, linewidth = 1, maze_type = trace['maze_type'])
    ax.plot(x[idx_ons:idx_end+1], y[idx_ons:idx_end+1], color = 'gray', linewidth = 0.5)
    ax.set_aspect('equal')
    ax.plot([x[idx_ons]], [y[idx_ons]], 'o', color = 'red')
    ax.plot([x[idx_end]], [y[idx_end]], '^', color = 'k')
    
    spikes = trace['Spikes'][cell_idx]
    ms_time = trace['ms_time_behav']/1000
    idx = np.where((spikes == 1)&(ms_time > behav_time[idx_ons])&(ms_time <= behav_time[idx_end]))[0]
    spike_time = ms_time[idx]
    pos_idx = np.zeros(spike_time.shape[0], dtype = np.int64)
    for i in range(spike_time.shape[0]):
        pos_idx = np.where(behav_time < spike_time[i])[0][-1]
    
    spike_x, spike_y = x[pos_idx], y[pos_idx]
    
    ax.plot(spike_x, spike_y, 'o', color = 'black', markeredgewidth = 0, markersize = 3)
    ax.invert_yaxis()
    ax.axis([1.5, 7.5, -0.5, 1.5])
    return ax
    
    
plt.imshow(np.reshape(trace['smooth_map_all'][n], [48,48]), cmap='jet')

obj = BehaviorEvents(trace['maze_type'], spike_nodes_transform(trace['behav_nodes'], 12), 
                     trace['behav_time'])

behav_nodes = obj.abbr_node
print(behav_nodes)
direc_vec = obj.direc_vec
behav_time = obj.abbr_time
idx = np.where(behav_nodes == 6)[0]
print(idx)
direc = direc_vec[idx]
print(direc)
start_time = behav_time[idx]/1000
end_time = behav_time[idx+1]/1000
            
fig, axes = plt.subplots(nrows = len(idx), ncols = 1, figsize = (6, 2*len(idx)))

for i in range(len(idx)):
    plot_figure(trace, n, start_time[i], end_time[i], ax = axes[i])
    
plt.tight_layout()
plt.show()