import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mylib.maze_utils3 import Clear_Axes, GetDMatrices, mkdir
import copy as cp
from mylib.local_path import f2, figpath

code_id = "0860 - Decoding Retrieval With GNB"
loc = os.path.join(figpath, "Dsp", code_id)
mkdir(loc)

with open(f2['Trace File'][34], 'rb') as handle:
    trace = pickle.load(handle)

pos_traj = cp.deepcopy(trace['spike_nodes_original'])
in_maze_idx = np.where(np.isnan(pos_traj) == False)[0]
in_box_idx = np.where(np.isnan(pos_traj))[0]

with open(os.path.join(loc, "reduced_data_ntj_nb15.pkl"), 'rb') as handle:
    reduced_data = pickle.load(handle)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20), subplot_kw={'projection': '3d'})
    
D = GetDMatrices(1, 48)
dist_to_sp = D[pos_traj[in_maze_idx].astype(np.int64)-1, 0]
pos_colors = sns.color_palette("rainbow", as_cmap=True)(dist_to_sp / (np.max(dist_to_sp)+1e-8))
    
ax0 = axes
    
ax0.scatter(
    reduced_data[in_maze_idx, 0],
    reduced_data[in_maze_idx, 1],
    reduced_data[in_maze_idx, 2],
    s=1,
    edgecolor=None,
    c=pos_colors
)
"""
ax0.scatter(
    reduced_data[in_box_idx, 0],
    reduced_data[in_box_idx, 1],
    reduced_data[in_box_idx, 2],
    s=1,
    edgecolor=None,
    c='k'
)
"""
plt.show()