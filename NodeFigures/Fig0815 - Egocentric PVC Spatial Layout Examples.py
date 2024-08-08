from mylib.statistic_test import *

code_id = '0815 - Egocentric PVC Spatial Layout Examples'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)


if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    with open(r"E:\Data\Dsp_maze\10227\20231012\trace.pkl", 'rb') as handle:
        trace = pickle.load(handle)
    
    from mylib.preprocessing_ms import RateMap
    """
    trace['node 1']['MiceID'] = trace['MiceID']
    nan_idx = np.setdiff1d(np.arange(1, 2305), get_son_area(CP_DSP[1]))-1
    trace['node 1']['smooth_map_all'][:, nan_idx] = np.nan
    RateMap(trace['node 1'], color = 'black', linewidth=0.3)
    """
    DM = GetDMatrices(1, 48)
    PVC = np.zeros((10, 2304)) * np.nan
    PVC_CTRL = np.zeros((10, 2304)) * np.nan
    
    for i in range(1, 4):
        son_bins1 = get_son_area(CP_DSP[trace[f'node {i}']['Route']])
        son_bins2 = get_son_area(CP_DSP[0])
        D1 = DM[son_bins1-1, SP_DSP[trace[f'node {i}']['Route']]-1]
        D2 = DM[son_bins2-1, SP_DSP[0]-1]
        
        idx1 = np.argsort(D1)
        idx2 = np.argsort(D2)[:idx1.shape[0]]
        
        for j in range(idx1.shape[0]):
            PVC[i, son_bins1[idx1[j]]-1], _ = pearsonr(
                trace[f'node {i}']['smooth_map_all'][:, son_bins1[idx1[j]]-1],
                trace['node 0']['smooth_map_all'][:, son_bins2[idx2[j]]-1]
            )
            
            PVC_CTRL[i, son_bins1[idx1[j]]-1], _ = pearsonr(
                trace[f'node 4']['smooth_map_all'][:, son_bins1[idx1[j]]-1],
                trace['node 0']['smooth_map_all'][:, son_bins2[idx2[j]]-1]
            )
    
    for i in range(6, 9):
        son_bins1 = get_son_area(CP_DSP[trace[f'node {i}']['Route']])
        son_bins2 = get_son_area(CP_DSP[0])
        D1 = DM[son_bins1-1, SP_DSP[trace[f'node {i}']['Route']]-1]
        D2 = DM[son_bins2-1, SP_DSP[0]-1]
        
        idx1 = np.argsort(D1)
        idx2 = np.argsort(D2)[:idx1.shape[0]]
        
        for j in range(idx1.shape[0]):
            PVC[i, son_bins1[idx1[j]]-1], _ = pearsonr(
                trace[f'node {i}']['smooth_map_all'][:, son_bins1[idx1[j]]-1],
                trace['node 5']['smooth_map_all'][:, son_bins2[idx2[j]]-1]
            )
            
            PVC_CTRL[i, son_bins1[idx1[j]]-1], _ = pearsonr(
                trace[f'node 9']['smooth_map_all'][:, son_bins1[idx1[j]]-1],
                trace['node 5']['smooth_map_all'][:, son_bins2[idx2[j]]-1]
            )
            
    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump([PVC, PVC_CTRL], handle)
        
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        PVC, PVC_CTRL = pickle.load(handle)
            
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
cbar.set_ticks([v_min, 0, 0.2, v_max])
plt.savefig(os.path.join(loc, 'PVC.svg'), dpi=600)
plt.savefig(os.path.join(loc, 'PVC.png'), dpi=600)
plt.close()

v_max, v_min = np.nanmax(PVC_CTRL), np.nanmin(PVC_CTRL)
print(v_max, v_min)
v_max = round(v_max, 2) + 0.01
v_min = round(v_min, 2) - 0.01
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i in range(3):
    for j in range(3):
        im = axes[i, j].imshow(PVC_CTRL[i*3+j].reshape(48, 48), cmap='jet', vmin=v_min, vmax = v_max)
        axes[i, j].set_aspect('equal')
        axes[i, j] = Clear_Axes(axes[i, j])
        axes[i, j] = DrawMazeProfile(maze_type=1, axes=axes[i, j], nx=48, linewidth=0.5, color = 'black')
        axes[i, j].axis([-0.8, 47.8, 47.8, -0.8])

cbar = plt.colorbar(im)
cbar.set_ticks([v_min, 0, 0.2, v_max])
plt.savefig(os.path.join(loc, 'PVC [control].svg'), dpi=600)
plt.savefig(os.path.join(loc, 'PVC [control].png'), dpi=600)
plt.close()