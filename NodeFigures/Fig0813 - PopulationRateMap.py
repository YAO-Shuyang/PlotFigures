from mylib.statistic_test import *

code_id = '0813 - Population Rate Map'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    data = []
    
    for day in tqdm([f"Day {i}" for i in range(1, 8)]):
        idx = np.where(f2['training_day'] == day)[0]
        
        smooth_maps = {}
        for i in range(10):
            smooth_maps[i] = []
            
        for i in idx:
            with open(f2['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
                
            for j in range(10):
                smooth_maps[j].append(trace[f'node {j}']['old_map_clear'])
                
        for k in smooth_maps.keys():
            smooth_maps[k] = np.vstack(smooth_maps[k])
            
        data.append(smooth_maps)
        
    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(data, handle)
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        data = pickle.load(handle)

width_ratios = np.array([
    CP_DSP[0].shape[0], CP_DSP[1].shape[0], CP_DSP[2].shape[0], CP_DSP[3].shape[0], CP_DSP[0].shape[0],
    CP_DSP[0].shape[0], CP_DSP[4].shape[0], CP_DSP[5].shape[0], CP_DSP[6].shape[0], CP_DSP[0].shape[0]
])
width = width_ratios/np.sum(width_ratios)*24
print(width_ratios, width)
routes = [0, 1, 2, 3, 0, 0, 4, 5, 6, 0]

for i in range(7):
    print(data[i][0].shape[0])

for day in range(1, 8):
    fig, axes = plt.subplots(ncols=10,nrows=1, gridspec_kw={"width_ratios": width_ratios}, figsize=(16, 2))
    for i in range(10):
        
        ratemap = data[day - 1][i][:, CP_DSP[routes[i]]-1]
        ratemap = ratemap/np.max(ratemap, axis=1)[:, np.newaxis]
        ratemap[np.isnan(ratemap)] = 0
    
        if i == 0:
            peak_pos = np.argmax(ratemap, axis=1)
            sort_idx = np.argsort(peak_pos)
    
        ratemap = ratemap[sort_idx, :]
    
        sns.heatmap(
            ratemap,
            cmap=sns.color_palette("rocket", as_cmap=True),
            ax=axes[i],
            cbar=False,
            rasterized=True
        )
        axes[i] = Clear_Axes(axes[i])
        axes[i].set_yticks([0, ratemap.shape[0]-1])
        axes[i].set_aspect('auto')
    
    plt.savefig(os.path.join(loc, f'population_day{day}.svg'), dpi=600)
    plt.savefig(os.path.join(loc, f'population_day{day}.png'), dpi=600)
    plt.close()
