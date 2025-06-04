import numpy as np
from mylib.dsp.chmm_actions import CHMM
import matplotlib.pyplot as plt
from mylib.statistic_test import *

code_id = '0870 - CSCG Models'
loc = join(figpath, 'Dsp', code_id)
mkdir(loc)

def generate_data(behav_nodes: np.ndarray, maze_type: int = 1, nx: int = 12):
    assert maze_type in [1, 2], "Invalid maze type"
    
    G = maze1_graph if maze_type == 1 else maze2_graph

    is_deleted = np.zeros(behav_nodes.shape[0])
    behav_nodes = spike_nodes_transform(behav_nodes, 12).astype(np.int64)
    x, y = (behav_nodes-1) % nx, (behav_nodes-1) // nx
    
    for i in range(1, behav_nodes.shape[0]):
        if behav_nodes[i] == behav_nodes[i - 1]:
            is_deleted[i] = 1
        
    behav_nodes = behav_nodes[is_deleted == 0]
    action = np.zeros(behav_nodes.shape[0], dtype=np.int64)
    
    for i in range(1, behav_nodes.shape[0]):
        # If move South:
        if y[i] > y[i - 1] and x[i] == x[i - 1]:
            action[i-1] = 0
        # If move North:
        elif y[i] < y[i - 1] and x[i] == x[i - 1]:
            action[i-1] = 1
        # If move East:
        elif x[i] > x[i - 1] and y[i] == y[i - 1]:
            action[i-1] = 2
        # If move West:
        elif x[i] < x[i - 1] and y[i] == y[i - 1]:
            action[i-1] = 3

    return behav_nodes-1, action

with open(f2['Trace File'][34], 'rb') as handle:
    trace = pickle.load(handle)
    
x, a = generate_data(trace['correct_nodes'], maze_type=1, nx=12)
n_emissions = 144
n_clone_per_obs = 2
n_clones = np.ones(n_emissions, dtype=np.int64) * n_clone_per_obs
chmm = CHMM(n_clones=n_clones, pseudocount=1e-10, x=x, a=a) # Initialize the model
progression = chmm.learn_em_T(x, a, n_iter=100, term_early=False) # Training
print(chmm.C.shape)
for r in range(7):
    plt.figure(figsize=(4, 4))
    ax = Clear_Axes(plt.axes())
    bins = np.concatenate([np.arange(i*n_clone_per_obs, (i+1)*n_clone_per_obs) for i in CP_DSP[r]-1])
    ax.imshow(chmm.C[0, :][np.ix_(bins, bins)], cmap='Blues', vmin=0, vmax=1)
    for i in range(CP_DSP[r].shape[0]):
        ax.plot([i*n_clone_per_obs-0.5, (i+1)*n_clone_per_obs-0.5], [i*n_clone_per_obs-0.5, i*n_clone_per_obs-0.5], color='k', lw=0.5)
        ax.plot([i*n_clone_per_obs-0.5, (i+1)*n_clone_per_obs-0.5], [(i+1)*n_clone_per_obs-0.5, (i+1)*n_clone_per_obs-0.5], color='k', lw=0.5)
        ax.plot([i*n_clone_per_obs-0.5, i*n_clone_per_obs-0.5], [i*n_clone_per_obs-0.5, (i+1)*n_clone_per_obs-0.5], color='k', lw=0.5)
        ax.plot([(i+1)*n_clone_per_obs-0.5, (i+1)*n_clone_per_obs-0.5], [i*n_clone_per_obs-0.5, (i+1)*n_clone_per_obs-0.5], color='k', lw=0.5)
    plt.show()