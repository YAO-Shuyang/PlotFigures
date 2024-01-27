from mylib.statistic_test import *

code_id = '0059 - Naive - PVC Half-Half Correlation'
loc = join(figpath, code_id)
mkdir(loc)


""" 
with open(r"E:\Data\Cross_maze\10209\20230728\session 2\trace.pkl", "rb") as f:
    trace = pickle.load(f)
  
for i in range(trace['n_neuron']):
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize = (12,4))
    ax1, ax2 = Clear_Axes(axes[0]), Clear_Axes(axes[1])
    DrawMazeProfile(maze_type = 1, axes = ax1, nx = 48, color='black')
    DrawMazeProfile(maze_type = 1, axes = ax2, nx = 48, color='black')
    smooth_map_fir = trace['smooth_map_fir']
    smooth_map_sec = trace['smooth_map_sec']
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax1.imshow(np.reshape(smooth_map_fir[i],[48,48]), cmap = 'jet')
    ax2.imshow(np.reshape(smooth_map_sec[i],[48,48]), cmap = 'jet')
    plt.show()
"""
fig, axes = plt.subplots(ncols=13, nrows=12, figsize=(5*13,3*12))

to_d = {
    10209: [20230426, 20230428, 20230430, 20230502, 20230504, 20230506, 20230508, 20230510, 20230512, 20230515, 20230517, 20230519, 20230521],
    10212: [20230426, 20230428, 20230430, 20230502, 20230504, 20230506, 20230508, 20230510, 20230512, 20230515, 20230517, 20230519, 20230521],
    10224: [20230806, 20230808, 20230810, 20230812, 20230814, 20230816, 20230818, 20230820, 20230822, 20230824, 20230827, 20230829, 20230901],
    10227: [20230806, 20230808, 20230810, 20230812, 20230814, 20230816, 20230818, 20230820, 20230822, 20230824, 20230827, 20230829, 20230901]
}

for i, m in enumerate([10209, 10212, 10224, 10227]):
    for j, d in tqdm(enumerate(to_d[m])):
        axes[i*3, j] = Clear_Axes(axes[i*3, j])
        axes[i*3+1, j] = Clear_Axes(axes[i*3+1, j])
        axes[i*3+2, j] = Clear_Axes(axes[i*3+2, j])
        DrawMazeProfile(maze_type = 0, axes = axes[i*3, j], nx = 48, color='black')
        DrawMazeProfile(maze_type = 1, axes = axes[i*3+1, j], nx = 48, color='black')
        DrawMazeProfile(maze_type = 0, axes = axes[i*3+2, j], nx = 48, color='black')
        
        idx = np.where((f1['MiceID'] == m)&(f1['date'] == d)&(f1['session'] == 2))[0]
        if exists(f1['Trace File'][idx[0]]):
            with open(f1['Trace File'][idx[0]], 'rb') as handle:
                trace = pickle.load(handle)

            pc = np.where(trace['is_placecell'] == 1)[0]
            
            if 'LA' not in trace.keys() or trace['maze_type'] == 0:
                smooth_map_fir = trace['smooth_map_fir'][pc, :]
                smooth_map_sec = trace['smooth_map_sec'][pc, :]
            else:
                smooth_map_fir = trace['LA']['smooth_map_fir'][pc, :]
                smooth_map_sec = trace['LA']['smooth_map_sec'][pc, :]
            
            #PVC = np.corrcoef(smooth_map_fir, smooth_map_sec, rowvar=False)[(np.arange(smooth_map_fir.shape[1]), np.arange(smooth_map_sec.shape[1]))]
            
            PVC = np.zeros(smooth_map_fir.shape[1])
            for k in range(smooth_map_fir.shape[1]):
                PVC[k] = pearsonr(smooth_map_fir[:, k], smooth_map_sec[:, k])[0]
            
            im = axes[i*3+1, j].imshow(np.reshape(PVC, [48,48]), cmap='jet',vmax=1, vmin=-1)
            cbar = plt.colorbar(im, ax=axes[i*3+1, j])
            cbar.set_ticks([-1, 0, 1])
        
        idx = np.where((f1['MiceID'] == m)&(f1['date'] == d)&(f1['session'] == 1))[0]
        if exists(f1['Trace File'][idx[0]]):
            with open(f1['Trace File'][idx[0]], 'rb') as handle:
                trace = pickle.load(handle)
            
            pc = np.where(trace['is_placecell'] == 1)[0]
            smooth_map_fir = trace['smooth_map_fir'][pc, :]
            smooth_map_sec = trace['smooth_map_sec'][pc, :]
                
            #PVC = np.corrcoef(smooth_map_fir, smooth_map_sec, rowvar=False)[(np.arange(smooth_map_fir.shape[1]), np.arange(smooth_map_sec.shape[1]))]
            
            PVC = np.zeros(smooth_map_fir.shape[1])
            for k in range(smooth_map_fir.shape[1]):
                PVC[k] = pearsonr(smooth_map_fir[:, k], smooth_map_sec[:, k])[0]   
            
            im = axes[i*3, j].imshow(np.reshape(PVC, [48,48]), cmap='jet',vmax=1, vmin=-1)
            cbar = plt.colorbar(im, ax=axes[i*3, j])
            cbar.set_ticks([-1, 0, 1])
            
        idx = np.where((f1['MiceID'] == m)&(f1['date'] == d)&(f1['session'] == 3))[0]
        if exists(f1['Trace File'][idx[0]]):
            with open(f1['Trace File'][idx[0]], 'rb') as handle:
                trace = pickle.load(handle)

            pc = np.where(trace['is_placecell'] == 1)[0]
            smooth_map_fir = trace['smooth_map_fir'][pc, :]
            smooth_map_sec = trace['smooth_map_sec'][pc, :]
            

            PVC = np.zeros(smooth_map_fir.shape[1])
            for k in range(smooth_map_fir.shape[1]):
                PVC[k] = pearsonr(smooth_map_fir[:, k], smooth_map_sec[:, k])[0]

            # PVC = np.corrcoef(smooth_map_fir, smooth_map_sec, rowvar=False)[(np.arange(smooth_map_fir.shape[1]), np.arange(smooth_map_sec.shape[1]))]
            im = axes[i*3+2, j].imshow(np.reshape(PVC, [48,48]), cmap='jet',vmax=1, vmin=-1)
            cbar = plt.colorbar(im, ax=axes[i*3+2, j])
            cbar.set_ticks([-1, 0, 1])
        
        axes[i*3, j].axis([-0.7, 47.7, 47.7, -0.7])
        axes[i*3+1, j].axis([-0.7, 47.7, 47.7, -0.7])
        axes[i*3+2, j].axis([-0.7, 47.7, 47.7, -0.7])
        axes[i*3, j].set_aspect('equal')
        axes[i*3+1, j].set_aspect('equal')
        axes[i*3+2, j].set_aspect('equal')
# show the figure
plt.tight_layout()
plt.savefig(os.path.join(loc, 'Stage 1.png'), dpi = 600)
plt.savefig(os.path.join(loc, 'Stage 1.svg'), dpi = 600)
plt.show()

fig, axes = plt.subplots(ncols=13, nrows=8, figsize=(5*13,3*8))
to_d = {
    10209: [20230703, 20230705, 20230707, 20230709, 20230711, 20230713, 20230715, 20230717, 20230719, 20230721, 20230724, 20230726, 20230728],
    10212: [20230703, 20230705, 20230707, 20230709, 20230711, 20230713, 20230715, 20230717, 20230719, 20230721, 20230724, 20230726, 20230728],
}
for i, m in enumerate([10209, 10212]):
    for j, d in enumerate(to_d[m]):
        
        axes[i*4, j] = Clear_Axes(axes[i*4, j])
        axes[i*4+1, j] = Clear_Axes(axes[i*4+1, j])
        axes[i*4+2, j] = Clear_Axes(axes[i*4+2, j])
        axes[i*4+3, j] = Clear_Axes(axes[i*4+3, j])
        DrawMazeProfile(maze_type = 0, axes = axes[i*4, j], nx = 48, color='black')
        DrawMazeProfile(maze_type = 1, axes = axes[i*4+1, j], nx = 48, color='black')
        DrawMazeProfile(maze_type = 2, axes = axes[i*4+2, j], nx = 48, color='black')
        DrawMazeProfile(maze_type = 0, axes = axes[i*4+3, j], nx = 48, color='black')
        
        idx = np.where((f1['MiceID'] == m)&(f1['date'] == d)&(f1['session'] == 2))[0]
        if exists(f1['Trace File'][idx[0]]):
            with open(f1['Trace File'][idx[0]], 'rb') as handle:
                trace = pickle.load(handle)

            pc = np.where(trace['is_placecell'] == 1)[0]
            
            if 'LA' not in trace.keys() or trace['maze_type'] == 0:
                smooth_map_fir = trace['smooth_map_fir'][pc, :]
                smooth_map_sec = trace['smooth_map_sec'][pc, :]
            else:
                smooth_map_fir = trace['LA']['smooth_map_fir'][pc, :]
                smooth_map_sec = trace['LA']['smooth_map_sec'][pc, :]
            
            #PVC = np.corrcoef(smooth_map_fir, smooth_map_sec, rowvar=False)[(np.arange(smooth_map_fir.shape[1]), np.arange(smooth_map_sec.shape[1]))]
            
            PVC = np.zeros(smooth_map_fir.shape[1])
            for k in range(smooth_map_fir.shape[1]):
                PVC[k] = pearsonr(smooth_map_fir[:, k], smooth_map_sec[:, k])[0]
            
            im = axes[i*4+1, j].imshow(np.reshape(PVC, [48,48]), cmap='jet',vmax=1, vmin=-1)
            cbar = plt.colorbar(im, ax=axes[i*4+1, j])
            cbar.set_ticks([-1, 0, 1])
        
        idx = np.where((f1['MiceID'] == m)&(f1['date'] == d)&(f1['session'] == 3))[0]
        if exists(f1['Trace File'][idx[0]]):
            with open(f1['Trace File'][idx[0]], 'rb') as handle:
                trace = pickle.load(handle)

            pc = np.where(trace['is_placecell'] == 1)[0]
            
            if 'LA' not in trace.keys() or trace['maze_type'] == 0:
                smooth_map_fir = trace['smooth_map_fir'][pc, :]
                smooth_map_sec = trace['smooth_map_sec'][pc, :]
            else:
                smooth_map_fir = trace['LA']['smooth_map_fir'][pc, :]
                smooth_map_sec = trace['LA']['smooth_map_sec'][pc, :]
            
            #PVC = np.corrcoef(smooth_map_fir, smooth_map_sec, rowvar=False)[(np.arange(smooth_map_fir.shape[1]), np.arange(smooth_map_sec.shape[1]))]
            
            PVC = np.zeros(smooth_map_fir.shape[1])
            for k in range(smooth_map_fir.shape[1]):
                PVC[k] = pearsonr(smooth_map_fir[:, k], smooth_map_sec[:, k])[0]
            
            im = axes[i*4+2, j].imshow(np.reshape(PVC, [48,48]), cmap='jet',vmax=1, vmin=-1)
            cbar = plt.colorbar(im, ax=axes[i*4+2, j])
            cbar.set_ticks([-1, 0, 1])
        
        
        idx = np.where((f1['MiceID'] == m)&(f1['date'] == d)&(f1['session'] == 1))[0]
        if exists(f1['Trace File'][idx[0]]):
            with open(f1['Trace File'][idx[0]], 'rb') as handle:
                trace = pickle.load(handle)

            pc = np.where(trace['is_placecell'] == 1)[0]
            smooth_map_fir = trace['smooth_map_fir'][pc, :]
            smooth_map_sec = trace['smooth_map_sec'][pc, :]
            

            PVC = np.zeros(smooth_map_fir.shape[1])
            for k in range(smooth_map_fir.shape[1]):
                PVC[k] = pearsonr(smooth_map_fir[:, k], smooth_map_sec[:, k])[0]

            # PVC = np.corrcoef(smooth_map_fir, smooth_map_sec, rowvar=False)[(np.arange(smooth_map_fir.shape[1]), np.arange(smooth_map_sec.shape[1]))]
            im = axes[i*4, j].imshow(np.reshape(PVC, [48,48]), cmap='jet',vmax=1, vmin=-1)
            cbar = plt.colorbar(im, ax=axes[i*4, j])
            cbar.set_ticks([-1, 0, 1])
            
        idx = np.where((f1['MiceID'] == m)&(f1['date'] == d)&(f1['session'] == 4))[0]
        if exists(f1['Trace File'][idx[0]]):
            with open(f1['Trace File'][idx[0]], 'rb') as handle:
                trace = pickle.load(handle)

            pc = np.where(trace['is_placecell'] == 1)[0]
            smooth_map_fir = trace['smooth_map_fir'][pc, :]
            smooth_map_sec = trace['smooth_map_sec'][pc, :]
            

            PVC = np.zeros(smooth_map_fir.shape[1])
            for k in range(smooth_map_fir.shape[1]):
                PVC[k] = pearsonr(smooth_map_fir[:, k], smooth_map_sec[:, k])[0]

            # PVC = np.corrcoef(smooth_map_fir, smooth_map_sec, rowvar=False)[(np.arange(smooth_map_fir.shape[1]), np.arange(smooth_map_sec.shape[1]))]
            im = axes[i*4+3, j].imshow(np.reshape(PVC, [48,48]), cmap='jet',vmax=1, vmin=-1)
            cbar = plt.colorbar(im, ax=axes[i*4+3, j])
            cbar.set_ticks([-1, 0, 1])
    
plt.tight_layout()
plt.savefig(os.path.join(loc, 'Stage 2.png'), dpi = 600)
plt.savefig(os.path.join(loc, 'Stage 2.svg'), dpi = 600)
plt.show()  
