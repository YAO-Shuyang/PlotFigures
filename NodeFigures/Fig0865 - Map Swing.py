from mylib.statistic_test import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mazepy.basic._time_sync import _coordinate_recording_time
from sklearn.manifold import Isomap

from umap.umap_ import UMAP
from mazepy.basic._time_sync import _coordinate_recording_time
from mazepy.datastruc.neuact import SpikeTrain, NeuralTrajectory
from mazepy.datastruc.variables import VariableBin

code_id = "0865 - Map Swing"
loc = join(figpath, 'Dsp', code_id)
mkdir(loc)
dir_name0 = join(figpath, "Dsp", "0850 - Lisa Paper Revisits")


def fit_kmeans(X, R: int, kmeans_init=None, is_return_model: bool = False):
    """
    Cluster Maps with KMeans model.
    
    Parameters
    ----------
    X : np.ndarray
        The entire map of this cell within this session.
        shape: (I x J x K) tensor of normalized firing rates
        
        I: Trials
        J: Spatial bins
        K: Neurons
    
    Returns
    -------
    U, V
    """
    X_wrap = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
    
    if kmeans_init is not None:
        kmean = KMeans(n_clusters=R, init=kmeans_init)
    else:
        kmean = KMeans(n_clusters=R)
        
    kmean.fit(X_wrap)
    
    U = np.zeros((X.shape[0], R))
    for i in range(X.shape[0]):
        U[i, kmean.labels_[i]] = 1
        
    V = kmean.cluster_centers_
    
    if R == 2 and is_return_model == False:
        nclusters = np.sum(U, axis=0)
        if nclusters[0] < nclusters[1]:
            print(f"{nclusters[0]} < {nclusters[1]}")
            U = U[:, [1, 0]]
            V = V[[1, 0], :]
    
    if is_return_model:
        return U, V, kmean
    else:
        return U, V

pre_and_final_segment_bins = np.concatenate([CP_DSP[6][:20], [97, 98, 86]]) #pre_and_final_segment_bins = np.concatenate([CP_DSP[6], [97, 98, 86, 128, 140, 139, 143, 118, 119, 107, 108, 120]]) #np.concatenate([np.array([135, 134, 133, 121, 109, 110, 122, 123, 111, 112, 100]), CP_DSP[3]])
region_label = np.zeros(144)
region_label[pre_and_final_segment_bins[4:7]-1] = -1
region_label[pre_and_final_segment_bins[-8:]-1] = 1

def visualize_phase_plot(reduced_data:np.ndarray, lap_traj: np.ndarray):
    beg = np.concatenate([[0], np.where(np.diff(lap_traj) != 0)[0] + 1])
    end = np.concatenate([np.where(np.diff(lap_traj) != 0)[0]+1, [len(lap_traj)]])
    effect_idx = np.concatenate([np.arange(beg[i], end[i]-1) for i in range(len(beg))])
    
    x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
    y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
    
    nx = ny = 50
    
    x_norm = (reduced_data[:, 0] - x_min) / (x_max - x_min + 1e-10)
    y_norm = (reduced_data[:, 1] - y_min) / (y_max - y_min + 1e-10)
    x_binned = (x_norm // (1/nx)).astype(np.int64)
    y_binned = (y_norm // (1/ny)).astype(np.int64)
    dx = np.diff(x_norm)
    dy = np.diff(y_norm)
    
    angles_mat = np.zeros((nx, ny))
    strengths = np.zeros((nx, ny))
    for i in range(len(nx)):
        for j in range(len(ny)):
            idx = np.where((x_binned == i) & (y_binned == j))[0]
            if len(idx) == 0:
                angles_mat[i, j] = np.nan
            else:
                dx_mean = np.mean(dx[idx])
                dy_mean = np.mean(dy[idx])
                angles_mat[i, j] = np.arctan2(dy_mean, dx_mean)
                strengths[i, j] = np.sqrt(dx_mean**2 + dy_mean**2)
                
                
    # Gaussian smoothing
    from scipy.ndimage import gaussian_filter
    angles_mat = gaussian_filter(angles_mat, sigma=1)
    strengths = gaussian_filter(strengths, sigma=1)
    
    fig = plt.figure(figsize=(4, 4))
    ax = Clear_Axes(fig.add_subplot(111), close_spines=['top', 'right'])
    ax.axis([0, 1, 0, 1])
    ax.set_aspect('equal')
    #ax.imshow(strengths.T, cmap='coolwarm')
    x = np.linspace(0, 1, nx+1)
    y = np.linspace(0, 1, ny+1)
    x = (x[:-1] + x[1:]) / 2
    y = (y[:-1] + y[1:]) / 2
    X, Y = np.meshgrid(x, y)
    ax.quiver(
        X.flatten(),
        Y.flatten(),
        np.cos(angles_mat.flatten()),
        -np.sin(angles_mat.flatten()),
        strengths.flatten(),
        cmap = 'coolwarm',
        scale=5
    )
    plt.show()
    
def visualize(mouse: int, azim: int = 152, elev: int = 9):
    mouse_date = {
        10212: [3, 4, 5],
        10224: [0],
        10227: [2, 3, 4],
        10232: [0, 1]
    }
    if exists(join(loc, f"{mouse}.pkl")):
        with open(join(loc, f"{mouse}.pkl"), 'rb') as f:
            neural_traj, node_traj, pos_traj, speed_traj, lap_traj, session_traj, map_traj, route_traj, time_traj, len_traj = pickle.load(f)
        print(neural_traj.shape)
    else:
        file_idx = np.where(f2['MiceID'] == mouse)[0]
        
        neural_traj = []
        node_traj = []
        pos_traj = []
        speed_traj = []
        lap_traj = []
        session_traj = []
        map_traj = []
        route_traj = []
        time_traj = []
        len_traj = []
        
        with open(f_CellReg_dsp['cellreg_folder'][np.where(f_CellReg_dsp['MiceID'] == mouse)[0][0]], 'rb') as f:
            index_map = pickle.load(f).astype(np.int64)
            
        if mouse != 10232:
            index_map = index_map[1:, :]
            
        is_cell = np.where(index_map == 0, 0, 1)
        cell_count = np.sum(is_cell, axis=0)
        cell_idx = np.where(cell_count == 7)[0]
        
        for session in mouse_date[mouse]:
            with open(f2['Trace File'][file_idx[session]], 'rb') as f:
                trace = pickle.load(f)
                
            with open(join(dir_name0, f"{mouse}.pkl"), "rb") as f:
                X, session_label, route_label, kmeans_init = pickle.load(f)

            kmeans_init = np.vstack([np.mean(i, axis=0) for i in kmeans_init])
            U, V = fit_kmeans(X, 2, kmeans_init=kmeans_init)
            U = U[session_label == session, :]
            
            beg_time, end_time = trace['lap beg time'], trace['lap end time']
            beg, end = LapSplit(trace, trace['paradigm'])
            routes = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg)    
            
            for j in tqdm(np.where(np.isin(routes, [3, 6, 1, 4]))[0]):
                spike_idx = np.where(
                    (trace['ms_time'] >= beg_time[j]) & (trace['ms_time'] <= end_time[j]) &
                    (np.isnan(trace['spike_nodes_original']) == False)
                )[0]
                init_idx = spike_idx[0]
                in_seg_idx = np.where(np.isin(spike_nodes_transform(trace['spike_nodes_original'][spike_idx], 12), pre_and_final_segment_bins))[0]
                spike_idx = spike_idx[in_seg_idx]

                spike_nodes = trace['spike_nodes_original'][spike_idx].astype(np.int64)-1
                Spikes = trace['Spikes_original'][:, spike_idx]
                
                spike_train = SpikeTrain(
                    activity=Spikes,
                    time=trace['ms_time'][spike_idx],
                    variable=VariableBin(spike_nodes),
                )
                neural_trajs: NeuralTrajectory = spike_train.calc_neural_trajectory(500, 100)
                spike_t = np.asarray(neural_trajs.time) / 1000
                
                x = trace['correct_pos'][beg[j]:end[j], 0]/10
                y = trace['correct_pos'][beg[j]:end[j], 1]/10
                t = trace['correct_time'][beg[j]:end[j]]/1000 - trace['correct_time'][beg[j]]/1000
                dx = np.diff(x)
                dy = np.diff(y)
                dt = np.diff(t)
                dis = np.sqrt(dx**2+dy**2)
                cumdis = np.concatenate([[0], np.cumsum(dis)])
                v = np.convolve(np.sqrt(dx**2 + dy**2), np.ones(5), mode='same') / np.convolve(dt, np.ones(5), mode='same')
                
                #spike_t = (trace['ms_time'][spike_idx] - trace['ms_time'][init_idx])/1000
                idx = _coordinate_recording_time(spike_t, t[:-1])
                
                neural_traj.append(neural_trajs.to_array()[index_map[session, cell_idx]-1, :].astype(np.float64))
                _Length = neural_traj[-1].shape[1]
                node_traj.append(neural_trajs.variable.to_array().astype(np.int64))
                pos_traj.append(np.vstack([x, y])[:, idx].astype(np.float64).T)
                speed_traj.append(v[idx].astype(np.float64))
                lap_traj.append(np.repeat(j, _Length).astype(np.int64))
                session_traj.append(np.repeat(session, _Length).astype(np.int64))
                map_traj.append(np.repeat(U[j, 1], _Length).astype(np.int64))
                route_traj.append(np.repeat(routes[j], _Length).astype(np.int64))
                time_traj.append(spike_t.astype(np.float64))
                len_traj.append(cumdis[idx].astype(np.float64))
            
        neural_traj = np.concatenate(neural_traj, axis=1)
        node_traj = np.concatenate(node_traj)
        pos_traj = np.concatenate(pos_traj, axis=0)
        speed_traj = np.concatenate(speed_traj)
        lap_traj = np.concatenate(lap_traj)
        session_traj = np.concatenate(session_traj)
        map_traj = np.concatenate(map_traj)
        route_traj = np.concatenate(route_traj)
        time_traj = np.concatenate(time_traj)
        len_traj = np.concatenate(len_traj)
        
        with open(join(loc, f"{mouse}.pkl"), 'wb') as f:
            pickle.dump([
                neural_traj, node_traj, pos_traj, speed_traj, lap_traj, session_traj, map_traj, route_traj, time_traj, len_traj
            ], f)
        print(neural_traj.shape)
        print(join(loc, f"{mouse}.pkl"), "  is saved")
    
    D = GetDMatrices(1, 48)
    
    dist_arr = D[node_traj, 2303]
    colors = sns.color_palette('rainbow', as_cmap=True)(1-(dist_arr - dist_arr.min()) / (dist_arr.max() + 1e-10 - dist_arr.min()))
    map_colors = MAPPaletteRGBA[map_traj, :]
    session_colors = [sns.color_palette('rainbow', len(np.unique(session_traj)))[i] for i in session_traj - np.min(session_traj)]
    
    if exists(join(loc, f"{mouse}_isomap.pkl")):#exists(join(loc, f"{mouse}_umap.pkl")):#
        """ 
        with open(join(loc, f"{mouse}_umap.pkl"), 'rb') as f:
            reduced_data, umap_model = pickle.load(f)
        """
        with open(join(loc, f"{mouse}_isomap.pkl"), 'rb') as f:
            reduced_data, isomap_model = pickle.load(f)
        
    else:
        n_component_pca = 30
        n_component_isomap = 6
        n_neighbors = 100
        
        pca = PCA(n_components=n_component_pca)
        denoised_data = pca.fit_transform(neural_traj.T)
        """   
        umap_model = UMAP(
            n_components=n_component_isomap, 
            n_neighbors=n_neighbors,
            metric='cosine'
        )

        print("UMAP Fitting...  ", end='')
        t1 = time.time()
        reduced_data = umap_model.fit_transform(denoised_data)
        print(f"{time.time() - t1:.3f} seconds")
        with open(join(loc, f"{mouse}_umap.pkl"), 'wb') as f:
            pickle.dump([reduced_data, umap_model], f)
        """
        t1 = time.time()
        isomap_model = Isomap(n_components=n_component_isomap, n_neighbors=n_neighbors)
        print("ISOMAP Fitting...  ", end='')
        isomap_model.fit(denoised_data)
        print(f"{time.time() - t1:.3f} seconds")
        reduced_data = np.zeros((neural_traj.shape[1], n_component_isomap))
        print("ISOMAP Transforming...")
        for i in tqdm(range(0, neural_traj.shape[1], 1000)):
            sup_idx = min(i + 1000, neural_traj.shape[1])
            reduced_data[i:sup_idx] = isomap_model.transform(denoised_data[i:sup_idx])
        
        with open(join(loc, f"{mouse}_isomap.pkl"), 'wb') as f:
            pickle.dump([reduced_data, isomap_model], f)
         
    beg = np.concatenate([[0], np.where(np.diff(lap_traj) != 0)[0] + 1])
    end = np.concatenate([np.where(np.diff(lap_traj) != 0)[0], [len(lap_traj)]])
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 8), subplot_kw={'projection': '3d'})
    ax0, ax1, ax2, ax3 = axes.flatten()
    #ax0 = Clear_Axes(axes[0], close_spines=['top', 'right'])
    #ax1 = Clear_Axes(axes[1], close_spines=['top', 'right'])
    #ax2 = Clear_Axes(axes[2], close_spines=['top', 'right'])
    #ax3 = Clear_Axes(axes[3], close_spines=['top', 'right'])
    x, y, z = reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2]
    x2, y2, z2 = reduced_data[map_traj == 1, 3], reduced_data[map_traj == 1, 4], reduced_data[map_traj == 1, 5]
    
    centroid_m0 = np.zeros((len(pre_and_final_segment_bins)-7, 6))
    centroid_m1 = np.zeros((len(pre_and_final_segment_bins)-7, 6))
    
    for i, bin in enumerate(CP_DSP[6][4:20]):
        if i >= len(centroid_m0):
            break
        idx = np.where((S2F[node_traj] == bin)&(map_traj == 0)&(np.isin(route_traj, [1, 4])))[0]
        idx2 = np.where((S2F[node_traj] == bin)&(((map_traj == 1)&(route_traj == 6)&(session_traj == 1))|(route_traj == 3)))[0]
        if len(idx) > 0:
            centroid_m0[i] = np.mean(np.vstack([reduced_data[idx, d] for d in range(6)]), axis=1)
            centroid_m1[i] = np.mean(np.vstack([reduced_data[idx2, d] for d in range(6)]), axis=1)
            
    print(centroid_m0[:, 0])
    print(centroid_m1[:, 1])
    
    centroid_colors = sns.color_palette("rainbow", len(centroid_m0))
    ax0.scatter(
        x, y, z,
        c=colors, 
        s=1,
        alpha=0.8, 
        edgecolors=None
    )
    ax2.scatter(
        centroid_m0[:, 0],
        centroid_m0[:, 1],
        centroid_m0[:, 2],
        c=centroid_colors, 
        marker='s',
        edgecolors=None
    )
    ax2.plot(
        centroid_m0[:, 0],
        centroid_m0[:, 1],
        centroid_m0[:, 2],
        lw=0.5,
        color=RemappingPalette[0]
    )
    ax2.scatter(
        centroid_m1[:, 0],
        centroid_m1[:, 1],
        centroid_m1[:, 2],
        marker='s',
        c=centroid_colors, 
        edgecolors=None
    )
    ax2.plot(
        centroid_m1[:, 0],
        centroid_m1[:, 1],
        centroid_m1[:, 2],
        lw=0.5,
        color=RemappingPalette[1]
    )
    
    ax3.scatter(
        centroid_m0[:, 3],
        centroid_m0[:, 4],
        centroid_m0[:, 5],
        c=centroid_colors, 
        marker='s',
        edgecolors=None
    )
    ax3.plot(
        centroid_m0[:, 3],
        centroid_m0[:, 4],
        centroid_m0[:, 5],
        lw=0.5,
        color=RemappingPalette[0]
    )
    ax3.scatter(
        centroid_m1[:, 3],
        centroid_m1[:, 4],
        centroid_m1[:, 5],
        marker='s',
        c=centroid_colors, 
        edgecolors=None
    )
    ax3.plot(
        centroid_m1[:, 3],
        centroid_m1[:, 4],
        centroid_m1[:, 5],
        lw=0.5,
        color=RemappingPalette[1]
    )
    
    for i in range(len(beg)):
        
        ax1.plot(
                x[beg[i]:end[i]],
                y[beg[i]:end[i]],
                z[beg[i]:end[i]],
                color=DSPPalette[route_traj[beg[i]]], 
                linewidth=0.5, 
        )
        """
        ax0.plot(
            [reduced_data[beg[i], 0]],
            [reduced_data[beg[i], 1]],
            'o',
            c='k',
            markersize=5,
            markeredgewidth=0
        )
        ax0.plot(
            [reduced_data[end[i]-1, 0]],
            [reduced_data[end[i]-1, 1]],
            '^',
            c='k',
            markersize=5,
            markeredgewidth=0
        )
        """
    ax1.scatter(
        x, y, z,
        c=session_colors, 
        s=1, 
        edgecolors=None##
    )
    """
    ax3.scatter(
        x, y, z,
        c=map_colors, 
        s=1, 
        edgecolors=None##
    )
    """
    plt.show()
# 27 elev 9 azim 152
visualize(10227)