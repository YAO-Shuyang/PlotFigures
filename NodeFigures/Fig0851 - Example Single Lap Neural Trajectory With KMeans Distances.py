from mylib.statistic_test import *

code_id = "0851 - Example Single Lap Neural Trajectory With KMeans Distances"
loc = join(figpath, "Dsp", code_id)
mkdir(loc)

dir_name0 = join(figpath, "Dsp", "0844 - Manifold of Initialization")
dir_name1 = join(figpath, "Dsp", "0850 - Lisa Paper Revisits")

def get_transient_map(mouse: int):
    if exists(join(dir_name1, f"transient_{mouse}.pkl")):
        with open(join(dir_name1, f"transient_{mouse}.pkl"), "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(
            f"Please run Fig0850 - Lisa Paper Revisits.ipynb first "
            f"to generate transient_{mouse}.pkl"
        )
        
if __name__ == '__main__':
    from sklearn.decomposition import PCA
    from umap.umap_ import UMAP
    from sklearn.svm import SVC
    

def visualize_manifolds(mouse: 10232, elev = 48, azim = 152):
    (
        kmeans_dist_traj, 
        neural_traj, 
        session_traj, 
        old_pos_traj, 
        route_traj, 
        lap_traj, 
        pos_traj, 
        speed_traj
    ) = get_transient_map(mouse)
    
    """
    pca = PCA(n_components=30)
    denoised_data = pca.fit_transform(neural_traj.T)
    model = UMAP(n_components=3)
    reduced_data = model.fit_transform(denoised_data)
    PC1, PC2, PC3 = reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2]
    """
    
    beg = np.concatenate([[0], np.where(np.ediff1d(lap_traj)!=0)[0]+1])
    end = np.concatenate([np.where(np.ediff1d(lap_traj)!=0)[0]+1, [lap_traj.shape[0]]])
    
    pca = PCA(n_components=30)
    denoised_data = pca.fit_transform(neural_traj.T)
    model = UMAP(n_components=3)
    reduced_data = model.fit_transform(denoised_data)
    PC1, PC2, PC3 = reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2] 

    svm = SVC()
    svm.fit(reduced_data[:, :3], np.where(kmeans_dist_traj>0, 0, 1))

    map_clusters = svm.predict(reduced_data[:, :3])
    
    # Each route select 10 laps
    selected_beg, selected_end = [], []
    for route in range(7):
        for day in range(7):
            idx = np.where((route_traj[beg] == route)&(session_traj[beg]))[0]
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
    
    dist_signs = np.where(kmeans_dist_traj > 0, 1, 0)
    
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), subplot_kw={'projection': '3d'})
    ax = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]
    
    colors = palette = sns.diverging_palette(h_neg=120, h_pos=240, s=80, l=38, as_cmap=True, sep=20)(np.clip((kmeans_dist_traj + 2)/4, 0, 1-1e-8))
    
    D = GetDMatrices(1, 48)
    dist_traj = D[pos_traj, 2303]
    dist_colors = plt.get_cmap("rainbow")(dist_traj/np.max(dist_traj))
    
    for i in range(selected_beg.shape[0]):
        beg_idx, end_idx = selected_beg[i], selected_end[i]
        if np.unique(map_clusters[beg_idx:end_idx]).shape[0] == 2:
            exclude_idx.append(i)
            continue

        ax.scatter( 
            PC1[beg_idx:end_idx], 
            PC2[beg_idx:end_idx], 
            PC3[beg_idx:end_idx],
            color=colors[beg_idx:end_idx],
            s=5,
            alpha=0.8,
            linewidth = 0
        )
        ax1.plot(PC1[beg_idx:end_idx], PC2[beg_idx:end_idx], PC3[beg_idx:end_idx], linewidth=1, color = DSPPalette[route_traj[beg_idx]])
        ax2.scatter( 
            PC1[beg_idx:end_idx], 
            PC2[beg_idx:end_idx], 
            PC3[beg_idx:end_idx],
            color=dist_colors[beg_idx:end_idx, :],
            s=5,
            alpha=0.8,
            linewidth = 0
        )   
    ax.view_init(elev=elev, azim=azim)
    ax1.view_init(elev=elev, azim=azim)
    ax2.view_init(elev=elev, azim=azim)
    plt.savefig(join(loc, f"{mouse} [example].png"), dpi=600)
    plt.savefig(join(loc, f"{mouse} [example].svg"), dpi=600)
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
    ax.invert_yaxis()
    plt.savefig(join(loc, f"{mouse} [example] position.png"), dpi=600)
    plt.savefig(join(loc, f"{mouse} [example] position.svg"), dpi=600)
    plt.show()

#visualize_manifolds(10232)
visualize_manifolds(10212, 26, -123)
visualize_manifolds(10224, 61, -37)
#visualize_manifolds(10227, 70, -178)