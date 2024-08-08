from mylib.statistic_test import *

code_id = '0814 - PCV Spatial Layout Examples'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

with open(r"E:\Data\Dsp_maze\10224\20231015\trace.pkl", 'rb') as handle:
    trace = pickle.load(handle)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    PVC = np.zeros((9, 2304))

    for i in range(1, 4):
        for j in range(2304):
            PVC[i-1, j], _ = pearsonr(
                trace['node 0']['smooth_map_all'][:, j], 
                trace['node '+str(i)]['smooth_map_all'][:, j]
            )
        
        nanbins = np.setdiff1d(np.arange(1, 2305), get_son_area(CP_DSP[trace[f'node {i}']['Route']])) - 1
        PVC[i-1, nanbins] = np.nan

    for i in range(6, 9):
        for j in range(2304):
            PVC[i-1, j], _ = pearsonr(
                trace['node 5']['smooth_map_all'][:, j], 
                trace['node '+str(i)]['smooth_map_all'][:, j]
            )
        
        nanbins = np.setdiff1d(np.arange(1, 2305), get_son_area(CP_DSP[trace[f'node {i}']['Route']])) - 1
        PVC[i-1, nanbins] = np.nan
            
    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(PVC, handle)
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        PVC = pickle.load(handle)

v_max, v_min = np.nanmax(PVC), np.nanmin(PVC)
print(v_max, v_min)
v_max = round(v_max, 2) + 0.01
v_min = round(v_min, 2) - 0.01
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i in range(3):
    for j in range(3):
        im = axes[i, j].imshow(PVC[i*3+j].reshape(48, 48), cmap='jet', vmin=v_min, vmax = v_max)
        axes[i, j].set_aspect('equal')
        axes[i, j] = Clear_Axes(axes[i, j])
        axes[i, j] = DrawMazeProfile(maze_type=1, axes=axes[i, j], nx=48, linewidth=0.5, color = 'black')
        axes[i, j].axis([-0.8, 47.8, 47.8, -0.8])

cbar = plt.colorbar(im)
cbar.set_ticks([v_min, 0, 0.2, 0.4, 0.6, 0.8, v_max])
plt.savefig(os.path.join(loc, 'PVC.svg'), dpi=600)
plt.savefig(os.path.join(loc, 'PVC.png'), dpi=600)
plt.close()