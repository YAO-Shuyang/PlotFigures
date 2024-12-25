from mylib.statistic_test import *
from mazepy.datastruc.neuact import SpikeTrain, TuningCurve, NeuralTrajectory
from mazepy.datastruc.variables import VariableBin
from mylib.calcium.dsp_ms import classify_lap
from mazepy.basic._time_sync import _coordinate_recording_time

code_id = "0844 - Manifold of Initialization"
loc = join(figpath, "Dsp", code_id)
mkdir(loc)

# Segment of Interests
SIG1 = np.array([4, 5, 8, 7, 6, 18, 17, 29, 30, 31, 19, 20, 21, 9, 10, 11, 12, 24])
SIG2 = np.array([23, 22, 34, 33, 32, 44, 45, 46, 47, 48, 60, 59, 58, 57, 56])
SIG3 = np.array([84, 83, 95, 93, 105, 106, 94, 82, 81, 80, 92, 104, 103, 91, 90, 78, 79, 67, 55, 54, 66, 65, 64, 63, 75, 74, 62, 
                 50, 51, 39, 38, 37, 49])
SIG4 = np.array([61, 73, 85, 97, 135, 134, 133, 121, 109, 110, 122, 123, 111, 112, 100])

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


def get_neural_traj(trace: dict):    
    beg_time, end_time = trace['lap beg time'], trace['lap end time']
    beg_idx = np.array([np.where(trace['correct_time'] >= beg_time[i])[0][0] for i in range(beg_time.shape[0])])
    routes = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg_idx)
    
    neural_trajs = []
    pos_trajs = []
    route_trajs = []
    lap_trajs = []
    map_trajs = []
    speed_trajs = []

    for i in range(beg_idx.shape[0]):
        if trace['is_perfect'][i] != 1:
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
        time_traj = neural_traj.time
        
        neural_trajs.append(neural_traj_vec)
        pos_trajs.append(pos_traj)
        dx, dy = np.ediff1d(trace['correct_pos'][:, 0]), np.ediff1d(trace['correct_pos'][:, 1])
        dt = np.ediff1d(trace['correct_time'])
        speed = np.sqrt(dx**2+dy**2) / dt * 100
        speed = np.convolve(speed, np.ones(3)/3, mode='same')
        idx = _coordinate_recording_time(time_traj.astype(np.float64), trace['correct_time'].astype(np.float64))
        speed_trajs.append(speed[idx])
        route_trajs.append(np.repeat(routes[i], neural_traj_vec.shape[1]).astype(np.int64))
        lap_trajs.append(np.repeat(i, neural_traj_vec.shape[1]).astype(np.int64))
        map_trajs.append(np.repeat(trace['map_cluster'][i], neural_traj_vec.shape[1]))
    
    return np.concatenate(neural_trajs, axis=1), np.concatenate(pos_trajs), np.concatenate(route_trajs), np.concatenate(lap_trajs), np.concatenate(map_trajs), np.concatenate(speed_trajs)

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
from sklearn.cluster import DBSCAN

def compute_centroids(
    reduced_data: np.ndarray,
    route_traj: np.ndarray,
    pos_traj: np.ndarray
):
    idx = np.where(route_traj==0)[0]

    old_traj = spike_nodes_transform(pos_traj[idx]+1, 12)-1
    radii = np.ones(144)
    centroids = np.zeros((144, 3))
    for i in CP_DSP[0]-1:
        subids = np.where(old_traj == i)[0]
        centroids[i, :] = np.mean(reduced_data[idx[subids], :], axis=0)[:3]
        res = reduced_data[idx[subids], :][:, :3] - centroids[i, :]
        radii[i] = np.percentile(
            np.sqrt(np.sum(res**2, axis=1)), 95
        )
    return centroids, radii
        
def find_min_normalized_distance(points, pos_traj, centroids, radii):
    """
    points   : shape (N, 3)
    centroids: shape (C, 3)
    radii    : shape (C,)  (the 95% range for each centroid)
    
    Returns
    -------
    min_norm_dist : shape (N,)  (the min normalized distance for each point)
    best_centroid : shape (N,)  (index of centroid with min normalized distance)
    """
    old_traj = spike_nodes_transform(pos_traj+1, 12)-1
    # Compute pairwise distance (N, C)
    # diffs[i, j] = points[i] - centroids[j]
    diffs = points - centroids[old_traj, :]  # shape (N, 3)
    sq_dists = np.sum(diffs**2, axis=1)                             # shape (N, )
    dists = np.sqrt(sq_dists)                                       # shape (N, )
    
    # Divide each distance by that centroid's radius
    # radii is (C,), so we broadcast to (N, C)
    norm_dists = dists / radii[old_traj]
    return norm_dists

def counts(traces, index_map):
    count_map = np.where(index_map > 0, 1, 0)
    sums = np.sum(count_map, axis=0)
    idx = np.where(sums == 7)[0]
    
    neural_trajs, pos_trajs, route_trajs, lap_trajs, map_trajs, speed_trajs = [], [], [], [], [], []
    session_trajs = []
    for i in range(len(traces)):
        neural_traj, pos_traj, route_traj, lap_traj, map_traj, speed_traj = get_neural_traj(traces[i])
        neural_trajs.append(neural_traj[index_map[i, idx]-1, :])
        pos_trajs.append(pos_traj)
        route_trajs.append(route_traj)
        lap_trajs.append(lap_traj)
        session_trajs.append(np.repeat(i, len(lap_traj)))
        map_trajs.append(map_traj)
        speed_trajs.append(speed_traj)
        
    neural_traj = np.concatenate(neural_trajs, axis=1)
    pos_traj = np.concatenate(pos_trajs)
    route_traj = np.concatenate(route_trajs)
    lap_traj = np.concatenate(lap_trajs)
    session_traj = np.concatenate(session_trajs)
    map_traj = np.concatenate(map_trajs)
    speed_traj = np.concatenate(speed_trajs)
    map_traj[map_traj==2] = 1

    #overlapping_bins = np.concatenate([Father2SonGraph[i] for i in [None, SIG1, SIG2, SIG3, SIG4][sig_type]])-1
    #overlapping_bins = np.concatenate([Father2SonGraph[i] for i in CP_DSP[3]])-1
    #idx = np.where((np.isin(pos_traj, overlapping_bins) == False))[0]
    """
    idx = np.concatenate([
        np.where(
            (np.isin(pos_traj, np.concatenate([Father2SonGraph[j] for j in CP_DSP[i][:6]]))) &
            (np.isin(route_traj, [i]))
        )[0] for i in [1, 2, 3, 4, 5, 6]
    ])
    
    neural_traj = neural_traj[:, idx]
    pos_traj = pos_traj[idx]
    route_traj = route_traj[idx]
    lap_traj = lap_traj[idx]
    session_traj = session_traj[idx]
    map_traj = map_traj[idx]
    speed_traj = speed_traj[idx]
    """
    print(neural_traj.shape, pos_traj.shape, route_traj.shape, lap_traj.shape)

    D = GetDMatrices(1, 48)
    dist_traj = D[pos_traj, 2303]
    dists = (dist_traj // 2).astype(np.int64)

    pca = PCA(n_components=30, random_state=42)
    denoised_data = pca.fit_transform(neural_traj.T)
    model = UMAP(n_components=3)
    reduced_data = model.fit_transform(denoised_data)

    # Use DBSCAN to further denoise
    dbscan = DBSCAN(eps=1, min_samples=50)
    res = dbscan.fit_predict(reduced_data[:, :3])
    idx = np.where(res != -1)[0]
    
    neural_traj = neural_traj[:, idx]
    pos_traj = pos_traj[idx]
    route_traj = route_traj[idx]
    lap_traj = lap_traj[idx]
    session_traj = session_traj[idx]
    map_traj = map_traj[idx]
    speed_traj = speed_traj[idx]
    reduced_data = reduced_data[idx, :]

    centroid, raddi = compute_centroids(
        reduced_data=reduced_data,
        route_traj=route_traj,
        pos_traj=pos_traj
    )
    dist_clusters = find_min_normalized_distance(reduced_data[:, :3], pos_traj, centroid, raddi)
    map_clusters = np.where(dist_clusters <= 1, 0, -1)

    """
    # Manually Label
    for_cluster_idx0 = np.where(
        (route_traj == 0)
    )[0]
    dlap = np.ediff1d(lap_traj)
    beg = np.concatenate([[0], np.where(dlap != 0)[0]+1])
    for_cluster_idx1 = np.concatenate([
        np.arange(beg[i], beg[i]+10)
        for i in range(beg.shape[0])
    ])
    for_cluster_idx = np.concatenate([for_cluster_idx0, for_cluster_idx1])
    labels = np.concatenate([
        np.zeros(for_cluster_idx0.shape[0]),
        np.ones(for_cluster_idx1.shape[0])
    ])
    
    """
    dbscan = DBSCAN(eps=0.4, min_samples=50)
    idx = np.where(map_clusters == -1)[0]
    res = dbscan.fit_predict(reduced_data[idx, :3])
    max_label = np.argmax([np.where(res == i)[0].shape[0] for i in np.unique(res)[1:]])
    print("Second DBSCAN: ", np.unique(res))
    map_clusters[idx[res == max_label]] = 1
    
    # Label Cluster Using 
    svm = SVC(kernel="rbf", random_state=42)
    svm.fit(reduced_data[map_clusters >= 0, :3], map_clusters[map_clusters>=0])
    map_clusters = np.asarray(svm.predict(reduced_data[:, :3])).astype(np.int64)
    dist_clusters = np.asarray(svm.decision_function(reduced_data[:, :3])).astype(np.float64)
    print(f"SVM Final Cluster:", np.unique(map_clusters))
    
    dlap = np.ediff1d(lap_traj)
    beg = np.concatenate([[0], np.where(dlap != 0)[0]+1])
    end = np.concatenate([np.where(dlap != 0)[0]+1, [lap_traj.shape[0]]])
    
    return map_clusters, session_traj, dist_clusters, dists, route_traj, lap_traj, pos_traj, speed_traj, beg, end, reduced_data, centroid, raddi

def draw(traces, index_map, elev = 50, azim = 160):
    map_clusters, session_traj, dist_clusters, dists, route_traj, lap_traj, pos_traj, speed_traj, beg, end, reduced_data, centroid, raddi = counts(
        traces, index_map
    )
    PC1, PC2, PC3 = reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2]
    D = GetDMatrices(1, 48)
    dist_traj = D[pos_traj, 2303]
    dists = (dist_traj // 2).astype(np.int64)
    dist_colors = plt.get_cmap("rainbow")((dist_traj - np.min(dist_traj))/(np.max(dist_traj) - np.min(dist_traj)))
    manifold_colors = MAPPaletteRGBA[map_clusters, :]
    
    
    dlap = np.ediff1d(lap_traj)
    beg = np.concatenate([[0], np.where(dlap != 0)[0]+1])
    end = np.concatenate([np.where(dlap != 0)[0]+1, [lap_traj.shape[0]]])
    
    # Each route select 10 laps
    selected_beg, selected_end = [], []
    for route in range(7):
        for day in range(7):
            idx = np.where((route_traj[beg] == route)&(session_traj[beg] == day))[0]
            if idx.shape[0] == 0:
                continue
            
            selected_idx = np.random.choice(idx, min(idx.shape[0], 1), replace = False)
            selected_beg.append(beg[selected_idx])
            selected_end.append(end[selected_idx])
        
    selected_beg, selected_end = np.concatenate(selected_beg), np.concatenate(selected_end)
    
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 8), subplot_kw={'projection': '3d'})
    print(f"{beg.shape[0]} Laps total")
    session_colors2 = sns.color_palette("rainbow", 7)
    exclude_idx = []
    for i in np.random.permutation(np.arange(selected_beg.shape[0])):
        beg_idx, end_idx = selected_beg[i], selected_end[i]
        
        axes[0, 0].plot(PC1[beg_idx:end_idx], PC2[beg_idx:end_idx], PC3[beg_idx:end_idx], linewidth=1, color = DSPPalette[route_traj[beg_idx]])
        #axes[2].plot(PC1[beg_idx:end_idx], PC2[beg_idx:end_idx], PC3[beg_idx:end_idx], linewidth=1, color = ['#333766', '#A4C096'][map_clusters[beg_idx]])
        axes[0, 1].scatter( 
            PC1[beg_idx:end_idx], 
            PC2[beg_idx:end_idx], 
            PC3[beg_idx:end_idx],
            color=manifold_colors[beg_idx:end_idx, :],
            s=5,
            linewidth = 0
        )
        axes[1, 0].scatter( 
            PC1[beg_idx:end_idx], 
            PC2[beg_idx:end_idx], 
            PC3[beg_idx:end_idx],
            color=dist_colors[beg_idx:end_idx, :],
            s=5,
            linewidth = 0
        )
    
    axes[1, 1].plot(centroid[CP_DSP[0]-1, 0], centroid[CP_DSP[0]-1, 1], centroid[CP_DSP[0]-1, 2], linewidth=0.5, color='k')
    colors2 = plt.get_cmap("rainbow")(np.linspace(0, 0.9999, CP_DSP[0].shape[0]))
    axes[1, 1].scatter( 
        centroid[CP_DSP[0]-1, 0], centroid[CP_DSP[0]-1, 1], centroid[CP_DSP[0]-1, 2],
        color=colors2,
        s=10,
        alpha=0.8,
        linewidth = 0
    )
        
    axes[0, 0].set_xlabel("UMAP1")
    axes[0, 0].set_ylabel("UMAP2")
    exclude_idx = np.array(exclude_idx)
    axes[0, 0].view_init(elev=elev, azim=azim)
    axes[0, 1].view_init(elev=elev, azim=azim)
    axes[1, 0].view_init(elev=elev, azim=azim)
    axes[1, 1].view_init(elev=elev, azim=azim)
    plt.savefig(join(loc, f"{traces[0]['MiceID']} - [example].png"), dpi=600)
    plt.savefig(join(loc, f"{traces[0]['MiceID']} - [example].svg"), dpi=600)
    plt.show()
    
    fig = plt.figure(figsize=(4, 4))
    ax = Clear_Axes(plt.axes())
    DrawMazeProfile(axes=ax, color='k', linewidth = 0.5)
    x, y = (pos_traj) % 48 + np.random.rand(pos_traj.shape[0])-0.5, pos_traj // 48 + np.random.rand(pos_traj.shape[0])-0.5
    """
    ego_dist_traj = np.zeros_like(dist_traj)
    for i in range(7):
        idx = np.where(route_traj == i)[0]
        ego_dist_traj[idx] = D[pos_traj[idx], SP_DSP[i]-1]
    """
    idx = np.concatenate([np.arange(selected_beg[i], selected_end[i]) for i in np.setdiff1d(np.arange(selected_beg.shape[0]), exclude_idx)])
    #idx = idx[ego_dist_traj[idx] <= 30]
    #ego_colors = plt.get_cmap("rainbow")((ego_dist_traj[idx] - np.min(ego_dist_traj[idx]))/(np.max(ego_dist_traj[idx]) - np.min(ego_dist_traj[idx])))

    ax.scatter(
        x[idx], y[idx], color = dist_colors[idx], 
        s=2, alpha=0.8, linewidth = 0
    )
    ax.invert_yaxis()
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
# draw(traces, index_map, 33, -29) #10227
# draw(traces, index_map, 33, -29) #10224
#draw(traces, index_map, 33, -29) #10232
draw(traces, index_map, 17, -163) #10212