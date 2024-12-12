from mylib.statistic_test import *

code_id = '0014 - Mean Rate Regress Out Speed'
loc = os.path.join(figpath, code_id)
mkdir(loc)

from mazepy.basic.convert import coordinate_recording_time

def separate_speed_bins(trace):
    dx = np.ediff1d(trace['processed_pos_new'][:, 0])/10
    dy = np.ediff1d(trace['processed_pos_new'][:, 1])/10
        
    dt = np.ediff1d(trace['behav_time'])
    try:
        Speed = np.sqrt(dx**2 + dy**2) * 1000 / dt
    except:
        Speed = np.sqrt(dx[:-1]**2 + dy[:-1]**2) * 1000 / dt
        
    Speed = np.convolve(Speed, np.ones(3)/3, mode='same')

    speed_labels = np.clip(((Speed+10) // 20).astype(np.int64), 0, 3)
    
    coordinate_indices = coordinate_recording_time(trace['ms_time'], trace['behav_time'][:-1])
    ms_speed_labels = speed_labels[coordinate_indices]
    
    return ms_speed_labels

def expand_bins(bins, maze_type: int):
    G = maze_graphs[(maze_type, 48)]
    
    expand_bins = [bins+1]
    for b in bins:
        expand_bins.append(np.array(G[b]))
        
    expand_bins = np.concatenate(expand_bins)
    expand_bins = np.unique(expand_bins)-1
    
    return expand_bins

idx = np.where(
    (np.isin(f1['MiceID'], [10209, 10212, 10224, 10227])) &
    (f1['maze_type'] != 0)
)
if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = {
        "MiceID": [],
        "Maze Type": [],
        "Training Day": [],
        "Stage": [],
        "Mean Rate": [],
        "Speed": []
    }
    for mouse in [10209, 10212, 10224, 10227]:
        for maze_type in [1, 2]:
            CP = cp.deepcopy(correct_paths[(maze_type, 48)])-1
            idx = np.where((f1['MiceID'] == mouse) & (f1['maze_type'] == maze_type))[0]
            
            print(f"{mouse}, Maze {maze_type}")    
            
            for j in tqdm(idx):
                if f1['include'][j] == 0:
                    continue
                with open(f1['Trace File'][j], 'rb') as handle:
                    trace = pickle.load(handle)
                
                ms_speed_labels = separate_speed_bins(trace)
                
                beg, end = trace['lap beg time'], trace['lap end time']
                
                spike_idx = np.concatenate([
                    np.where((trace['ms_time'] >= beg[i]) & (trace['ms_time'] <= end[i]) & (np.isnan(trace['spike_nodes_original']) == False))[0] 
                    for i in range(beg.shape[0])
                ])
                ms_speed_labels = ms_speed_labels[spike_idx]
                dt = np.ediff1d(trace['ms_time'][spike_idx])
                spike_nodes = trace['spike_nodes_original'][spike_idx]
                Spikes = trace['Spikes_original'][:, spike_idx]
                
                remain_idx = np.where(dt <= 10000)[0]
                ms_speed_labels = ms_speed_labels[remain_idx]
                
                for s in range(4):
                    d = np.where((ms_speed_labels == s))[0]
                    d = remain_idx[d]
                    occu_time = scipy.stats.binned_statistic(
                        spike_nodes[d],
                        dt[d],
                        bins=2304,
                        statistic="sum",
                        range=[0, 2304 + 0.0001]
                    )[0]
                    t_total = np.nansum(occu_time)/1000
                    
                    mean_rate = np.sum(Spikes[:, d], axis = 1) / t_total
                
                    mazet = f"Maze {f1['maze_type'][j]}"
                    Data['MiceID'].append(mouse)
                    Data['Maze Type'].append(mazet)
                    Data['Training Day'].append(f1['training_day'][j])
                    Data['Stage'].append(f1['Stage'][j])
                    Data['Mean Rate'].append(np.nanmean(mean_rate))
                    Data['Speed'].append(s)
    
    for key in Data.keys():
        Data[key] = np.array(Data[key])
    
    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    D = pd.DataFrame(Data)
    D.to_excel(os.path.join(figdata, code_id+'.xlsx'), index=False)
    print(Data['Mean Rate'].shape)
                
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

colors = sns.color_palette("rocket", 3)

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8, 3))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

idx = np.where((Data['Stage'] == "Stage 1") & (Data['Maze Type'] == "Maze 1") & (Data['Speed'] <= 2))[0]
SubData = SubDict(Data, Data.keys(), idx)
sns.lineplot(
    x = "Training Day",
    y = "Mean Rate",
    data = SubData,    
    hue='Speed',
    palette = sns.color_palette("rocket", 3),
    ax = ax1,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
idx = np.where((Data['Stage'] == "Stage 2") & (Data['Maze Type'] == "Maze 1") & (Data['Speed'] <= 2))[0]
SubData = SubDict(Data, Data.keys(), idx)
sns.lineplot(
    x = "Training Day",
    y = "Mean Rate",
    data = SubData,
    hue='Speed',
    palette = sns.color_palette("rocket", 3),
    ax = ax2,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
ax1.set_ylim(0, 1)
ax1.set_yticks(np.linspace(0, 1, 6))
ax2.set_ylim(0, 1)
ax2.set_yticks(np.linspace(0, 1, 6))
plt.savefig(os.path.join(loc, 'SI [Maze 1].png'), dpi=600)
plt.savefig(os.path.join(loc, 'SI [Maze 1].svg'), dpi=600)
plt.show()

fig = plt.figure(figsize=(4, 3))
ax1 = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

idx = np.where((Data['Stage'] == "Stage 2") & (Data['Maze Type'] == "Maze 2") & (Data['Speed'] <= 2))[0]
SubData = SubDict(Data, Data.keys(), idx)
sns.lineplot(
    x = "Training Day",
    y = "Mean Rate",
    data = SubData,
    hue='Speed',
    palette = sns.color_palette("rocket", 3),
    ax = ax1,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
ax1.set_ylim(0, 1)
ax1.set_yticks(np.linspace(0, 1, 6))
plt.savefig(os.path.join(loc, 'SI [Maze 2].png'), dpi=600)
plt.savefig(os.path.join(loc, 'SI [Maze 2].svg'), dpi=600)
plt.show()