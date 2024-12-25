from mylib.statistic_test import *
from mazepy.datastruc.neuact import SpikeTrain, TuningCurve, NeuralTrajectory
from mazepy.datastruc.variables import VariableBin
from mylib.calcium.dsp_ms import classify_lap

code_id = "0838 - Neural Trajectory Remapping"
loc = join(figpath, "Dsp", code_id)
mkdir(loc)

def get_neural_traj(trace: dict):    
    beg_time, end_time = trace['lap beg time'], trace['lap end time']
    beg_idx = np.array([np.where(trace['correct_time'] >= beg_time[i])[0][0] for i in range(beg_time.shape[0])])
    routes = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg_idx)
    
    neural_trajs = []
    pos_trajs = []
    route_trajs = []
    lap_trajs = []
    map_trajs = []
    
    for i in tqdm(range(beg_idx.shape[0])):
        if trace['is_perfect'][i] == 0:
            continue
        
        spike_idx = np.where(
            (trace['ms_time'] >= beg_time[i]) & (trace['ms_time'] <= end_time[i]) &
            (np.isnan(trace['spike_nodes_original']) == False)
        )[0]
        
        spike_nodes = trace['spike_nodes_original'][spike_idx].astype(np.int64)-1
        Spikes = trace['Spikes_original'][:, spike_idx]
        
        spike_train = SpikeTrain(
            activity=Spikes,
            time=trace['ms_time'][spike_idx],
            variable=VariableBin(spike_nodes),
        )
        
        neural_traj = spike_train.calc_neural_trajectory(500, 100)
        neural_traj_vec = neural_traj.to_array()
        pos_traj = neural_traj.variable.to_array()
        
        neural_trajs.append(neural_traj_vec)
        pos_trajs.append(pos_traj)
        route_trajs.append(np.repeat(routes[i], neural_traj_vec.shape[1]).astype(np.int64))
        lap_trajs.append(np.repeat(i, neural_traj_vec.shape[1]).astype(np.int64))
        map_trajs.append(np.repeat(trace['map_cluster'][i], neural_traj_vec.shape[1]))
    
    return np.concatenate(neural_trajs, axis=1), np.concatenate(pos_trajs), np.concatenate(route_trajs), np.concatenate(lap_trajs), np.concatenate(map_trajs)

def hex_to_rgba(hex_color):
    """
    Convert a hex color (#RRGGBB or #RRGGBBAA) to RGBA format (0-255).
    """
    hex_color = hex_color.lstrip('#')  # Remove '#' if present
    if len(hex_color) == 6:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        a = 255  # Default alpha
    elif len(hex_color) == 8:
        r, g, b, a = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), int(hex_color[6:8], 16)
    else:
        raise ValueError("Invalid hex color format. Use #RRGGBB or #RRGGBBAA.")
    return r, g, b, a

def hex_to_rgba_normalized(hex_color):
    """
    Convert a hex color (#RRGGBB or #RRGGBBAA) to RGBA format (0-1).
    """
    r, g, b, a = hex_to_rgba(hex_color)
    return np.array([r / 255, g / 255, b / 255, a / 255])

DSPPaletteRGBA = np.vstack([hex_to_rgba_normalized(c) for c in DSPPalette])
MAPPaletteRGBA = np.vstack([hex_to_rgba_normalized(c) for c in ['#333766', '#A4C096']])

from umap.umap_ import UMAP
from matplotlib import cm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from umap.umap_ import UMAP
from matplotlib import cm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.svm import SVC

def plot(traces, index_map, elev = 50, azim = 160):
    count_map = np.where(index_map > 0, 1, 0)
    sums = np.sum(count_map, axis=0)
    idx = np.where(sums == 7)[0]
    
    neural_trajs, pos_trajs, route_trajs, lap_trajs, map_trajs = [], [], [], [], []
    session_trajs = []
    for i in range(len(traces)):
        neural_traj, pos_traj, route_traj, lap_traj, map_traj = get_neural_traj(traces[i])
        neural_trajs.append(neural_traj[index_map[i, idx]-1, :])
        pos_trajs.append(pos_traj)
        route_trajs.append(route_traj)
        lap_trajs.append(lap_traj)
        session_trajs.append(np.repeat(i, len(lap_traj)))
        map_trajs.append(map_traj)
        
    neural_traj = np.concatenate(neural_trajs, axis=1)
    pos_traj = np.concatenate(pos_trajs)
    route_traj = np.concatenate(route_trajs)
    lap_traj = np.concatenate(lap_trajs)
    session_traj = np.concatenate(session_trajs)
    map_traj = np.concatenate(map_trajs)
    map_traj[map_traj==2] = 1

    overlapping_bins = np.concatenate([Father2SonGraph[i] for i in CP_DSP[3]])-1

    print(route_traj)
    idx = np.where((np.isin(pos_traj, overlapping_bins)) & (np.isin(route_traj, [0, 1.,2., 3., 4., 5.,6.])))[0]

    neural_traj = neural_traj[:, idx]
    pos_traj = pos_traj[idx]

    oldbin_traj = spike_nodes_transform(pos_traj+1, 12)
    route_traj = route_traj[idx]
    lap_traj = lap_traj[idx]
    session_traj = session_traj[idx]
    map_traj = map_traj[idx]

    D = GetDMatrices(1, 48)
    dist_traj = D[pos_traj, 2303]
    dists = (dist_traj // 2).astype(np.int64)
    lda_label = dists + np.max(dists + 1) * route_traj

    print(neural_traj.shape, pos_traj.shape, route_traj.shape, lap_traj.shape)

    pca = PCA(n_components=30)
    denoised_data = pca.fit_transform(neural_traj.T)
    model = UMAP(n_components=3)
    reduced_data = model.fit_transform(denoised_data)
    PC1, PC2, PC3 = reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2]

    svm = SVC()
    svm.fit(reduced_data[:, :3], map_traj)

    map_clusters = svm.predict(reduced_data[:, :3])
    manifold_colors = MAPPaletteRGBA[map_clusters, :]

    dist_colors = plt.get_cmap("rainbow")(dist_traj/np.max(dist_traj))
    session_colors = plt.get_cmap("rainbow")(session_traj/np.max(session_traj))
    route_colors = DSPPaletteRGBA[route_traj.astype(np.int64), :]

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), subplot_kw={'projection': '3d'})
    
    dlap = np.ediff1d(lap_traj)
    beg = np.concatenate([[0], np.where(dlap != 0)[0]+1])
    end = np.concatenate([np.where(dlap != 0)[0]+1, [lap_traj.shape[0]]])
    
    # Each route select 10 laps
    selected_beg, selected_end = [], []
    for route in range(7):
        for day in range(7):
            if i < 6:
                continue
            idx = np.where((route_traj[beg] == route)&(session_traj[beg] == day))[0]
            try:
                selected_idx = np.random.choice(idx, 5, replace = False)
            except:
                selected_idx = idx
            selected_beg.append(beg[selected_idx])
            selected_end.append(end[selected_idx])
        
    selected_beg, selected_end = np.concatenate(selected_beg), np.concatenate(selected_end)
    
    print(f"{beg.shape[0]} Laps total")
    session_colors2 = sns.color_palette("rainbow", 7)
    exclude_idx = []
    for i in range(selected_beg.shape[0]):
        beg_idx, end_idx = selected_beg[i], selected_end[i]
        if np.unique(map_clusters[beg_idx:end_idx]).shape[0] == 2:
            exclude_idx.append(i)
            continue
        
        axes[0].plot(PC1[beg_idx:end_idx], PC2[beg_idx:end_idx], PC3[beg_idx:end_idx], linewidth=1, color = DSPPalette[route_traj[beg_idx]])
        axes[2].plot(PC1[beg_idx:end_idx], PC2[beg_idx:end_idx], PC3[beg_idx:end_idx], linewidth=1, color = ['#333766', '#A4C096'][map_clusters[beg_idx]])
        axes[1].scatter( 
            PC1[beg_idx:end_idx], 
            PC2[beg_idx:end_idx], 
            PC3[beg_idx:end_idx],
            color=dist_colors[beg_idx:end_idx, :],
            s=5,
            alpha=0.8,
            linewidth = 0
        )
        
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    exclude_idx = np.array(exclude_idx)
    axes[0].view_init(elev=elev, azim=azim)
    axes[1].view_init(elev=elev, azim=azim)
    axes[2].view_init(elev=elev, azim=azim)
    plt.savefig(join(loc, f"{traces[0]['MiceID']} [example].png"), dpi=600)
    plt.savefig(join(loc, f"{traces[0]['MiceID']} [example].svg"), dpi=600)
    plt.show()
    
    fig = plt.figure(figsize=(4, 4))
    ax = Clear_Axes(plt.axes())
    DrawMazeProfile(axes=ax, color='k', linewidth = 0.5)
    x, y = (pos_traj) % 48 + np.random.rand(pos_traj.shape[0])-0.5, pos_traj // 48 + np.random.rand(pos_traj.shape[0])-0.5
    idx = np.concatenate([np.arange(selected_beg[i], selected_end[i]) for i in np.setdiff1d(np.arange(selected_beg.shape[0]), exclude_idx)])
    ax.scatter(
        x[idx], y[idx], color = dist_colors[idx], 
        s=2, alpha=0.8, linewidth = 0
    )
    plt.savefig(join(loc, f"{traces[0]['MiceID']} [example] position.png"), dpi=600)
    plt.savefig(join(loc, f"{traces[0]['MiceID']} [example] position.svg"), dpi=600)
    plt.show()

traces = []
mouse = 10212
for i in np.where(f2['MiceID'] == mouse)[0]:
    with open(f2['Trace File'][i], 'rb') as handle:
        trace = pickle.load(handle)
    traces.append(trace)

with open(f_CellReg_dsp['cellreg_folder'][np.where(f_CellReg_dsp['MiceID'] == mouse)[0][0]], 'rb') as handle:
    index_map = pickle.load(handle)
    
    if mouse != 10232:
        index_map = index_map[1:, :]
        
    index_map = index_map.astype(np.int64)
plot(traces, index_map.astype(np.int64), 33, -128) #10224
#plot(traces, index_map.astype(np.int64), 50, 20) #10224
#plot(traces, index_map.astype(np.int64), 70, -178) #10227