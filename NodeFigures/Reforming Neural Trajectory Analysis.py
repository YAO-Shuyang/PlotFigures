from mylib.statistic_test import *
from mazepy.datastruc.neuact import SpikeTrain, TuningCurve, NeuralTrajectory
from mazepy.datastruc.variables import VariableBin
from mylib.calcium.dsp_ms import classify_lap
from mazepy.basic._time_sync import _coordinate_recording_time

from umap.umap_ import UMAP
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from pydiffmap import diffusion_map as dm
from sklearn.manifold import Isomap

# Create the diffusion map object


code_id = "Reforming Neural Trajectory Analysis"
loc = join(figpath, "Dsp", code_id)
mkdir(loc)

dir_name = join(figpath, "Dsp", "0850 - Lisa Paper Revisits")

mouse = 10212

def visualize_neural_traj(mouse):
    
    with open(join(loc, f"{mouse}.pkl"), 'rb') as handle:
        res = pickle.load(handle)
        
    reduced_traj = res['reduced_traj']
    pos_traj = res['pos_traj']
    manifold_traj = res['manifold_traj']
    neural_traj = res['neural_traj']

    t1 = time.time()
    print("Start Isomap")
    #dmap_model = dm.DiffusionMap.from_sklearn(n_evecs=10, epsilon='bgh', alpha=0.5)
    pca = PCA(n_components=30)
    denoised_data = pca.fit_transform(neural_traj.T)
    umap_model = UMAP(n_neighbors=15, n_components=3, metric='correlation')
    reduced_traj = umap_model.fit_transform(denoised_data)

    # Fit to your data
    #reduced_traj = dmap_model.fit_transform(neural_traj.T)    
    print(f"Time Used: {time.time() - t1:.3f}s")
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), subplot_kw={'projection': '3d'})
    
    D = GetDMatrices(1, 48)
    dist_to_sp = D[pos_traj, 0]
    pos_colors = sns.color_palette("rainbow", as_cmap=True)(dist_to_sp / (np.max(dist_to_sp)+1e-8))
        
    ax0, ax1 = axes[0], axes[1]
    
    ax0.scatter(
        reduced_traj[:, 0],
        reduced_traj[:, 1],
        reduced_traj[:, 2],
        s=1,
        edgecolor=None,
        c=pos_colors
    )
    
    ax1.scatter(
        reduced_traj[:, 0],
        reduced_traj[:, 1],
        reduced_traj[:, 2],
        s=1,
        edgecolor=None,
        c=MAPPaletteRGBA[manifold_traj, :]
    )
    plt.show()
    
visualize_neural_traj(mouse)
    
    
