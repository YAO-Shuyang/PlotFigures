from mylib.statistic_test import *
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, MDS

code_id = "0860 - Decoding Retrieval With GNB"
loc = os.path.join(figpath, "Dsp", code_id)
mkdir(loc)

def isomap_raw_traces(
    i: int, 
    f2: pd.DataFrame = f2, 
    n_fit: int = 15000, 
    n_neighbors: int = 30, 
    n_components: int = 3
) -> np.ndarray:
    """Perform ISOMAP Dimensionality Reduction on Calcium Raw Traces."""
    
    with open(f2['Trace File'][i], 'rb') as handle:
        trace = pickle.load(handle)
    
    print(f"{i} {f2['MiceID'][i]} {f2['date'][i]}")
    print(f"  {DateTime()}")
    pos_traj = cp.deepcopy(trace['spike_nodes_original'])
    in_maze_idx = np.where(np.isnan(pos_traj) == False)[0]
    in_box_idx = np.where(np.isnan(pos_traj))[0]

    pca = PCA(n_components=30)
    denoised_data = pca.fit_transform(trace['RawTraces'].T)

    selected_idx = (np.linspace(0, denois ed_data.shape[0]-1, n_fit)//1).astype(np.int64)
    #np.random.choice(denoised_data.shape[0], n_fit, replace=False)#range(0, n_fit)#
    isomap_model = Isomap(n_neighbors=n_neighbors, n_components=n_components, metric='cosine', n_jobs=8)
    #isomap_model.fit(denoised_data[selected_idx, :])
    t1 = time.time()
                                                                                                                                                                                                                                                                                                                                      
    isomap_model.fit(denoised_data[selected_idx, :])
    print(f"  Isomap Fitting Cost: {time.time() - t1:.3f}s")
    print(f"  Isomap Transformation Starts")
    reduced_data = np.zeros((denoised_data.shape[0], n_components))
    for j in tqdm(range(0, denoised_data.shape[0], 1000)):
        targ = j + 1000 if j + 1000 < denoised_data.shape[0] else denoised_data.shape[0]
        reduced_data[j:targ, :] = isomap_model.transform(denoised_data[j:targ, :])

    trace['isomap_reduced_data'] = reduced_data

    with open(f2['Trace File'][i], 'wb') as handle:
        pickle.dump(trace, handle)
        
    print("  File Saved.")
    print(f"  {DateTime()}", end="\n\n")
        
#for i in range(20, len(f2)):
#    isomap_raw_traces(i)

def visualize(trace):
    """Visualize the ISOMAP reduced data."""

    pos_traj = cp.deepcopy(trace['spike_nodes_original'])
    in_maze_idx = np.where(np.isnan(pos_traj) == False)[0]
    in_box_idx = np.where(np.isnan(pos_traj))[0]
    
    D = GetDMatrices(1, 48)
    in_maze_tail = np.where(D[pos_traj[in_maze_idx].astype(np.int64)-1, 0] > D[Father2SonGraph[99][3]-1, 0])[0]
    in_maze_color = sns.color_palette("rainbow", as_cmap=True)(D[pos_traj[in_maze_idx].astype(np.int64)-1, 0] / (D[0, 2303] + 1e-8))

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 8), subplot_kw={'projection': '3d'})
    show_inmaze_idx0 = np.random.permutation(in_maze_idx.shape[0])[:8000]
    show_inmaze_idx = in_maze_idx[show_inmaze_idx0]
    axes[0].scatter(
        trace['isomap_reduced_data'][show_inmaze_idx, 0], 
        trace['isomap_reduced_data'][show_inmaze_idx, 1], 
        trace['isomap_reduced_data'][show_inmaze_idx, 2], 
        c=in_maze_color[show_inmaze_idx0, :], 
        s=1, 
        edgecolors=None
    )
    
    show_inbox_idx = np.random.choice(in_box_idx, 8000, replace=False)
    axes[1].scatter(
        trace['isomap_reduced_data'][show_inbox_idx, 0], 
        trace['isomap_reduced_data'][show_inbox_idx, 1], 
        trace['isomap_reduced_data'][show_inbox_idx, 2], 
        c='k', 
        s=1, 
        edgecolors=None
    )
    axes[1].scatter(
        trace['isomap_reduced_data'][in_maze_idx[in_maze_tail], 0], 
        trace['isomap_reduced_data'][in_maze_idx[in_maze_tail], 1], 
        trace['isomap_reduced_data'][in_maze_idx[in_maze_tail], 2], 
        c=in_maze_color[in_maze_tail, :], 
        s=1, 
        edgecolors=None
    )
    
    plt.show()
    
with open(f2['Trace File'][34], 'rb') as handle:
    trace = pickle.load(handle)
    
visualize(trace)

with open(f2['Trace File'][27], 'rb') as handle:
    trace = pickle.load(handle)
    
visualize(trace)

with open(f2['Trace File'][26], 'rb') as handle:
    trace = pickle.load(handle)
    
visualize(trace)

with open(f2['Trace File'][11], 'rb') as handle:
    trace = pickle.load(handle)
    
visualize(trace)