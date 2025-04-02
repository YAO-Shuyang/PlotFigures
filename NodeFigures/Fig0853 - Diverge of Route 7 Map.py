from mylib.statistic_test import *

code_id = "0853 - Diverge of Route 7 Map"
loc = os.path.join(figpath, code_id)
mkdir(loc)

from mylib.statistic_test import *
from mazepy.datastruc.neuact import SpikeTrain, TuningCurve, NeuralTrajectory
from mazepy.datastruc.variables import VariableBin
from mylib.calcium.dsp_ms import classify_lap
from mazepy.basic._time_sync import _coordinate_recording_time

dir_name1 = join(figpath, "Dsp", "0850 - Lisa Paper Revisits")
dir_name0 = join(figpath, "Dsp", "0844 - Manifold of Initialization")

def get_transient_map(mouse: int):
    """
    Get Final Maps and Information
    """
    if exists(join(dir_name1, f"transient_{mouse}.pkl")):
        with open(join(dir_name1, f"transient_{mouse}.pkl"), "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(
            f"Please run Fig0850 - Lisa Paper Revisits.ipynb first "
            f"to generate transient_{mouse}.pkl"
        )

from umap.umap_ import UMAP
from matplotlib import cm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN, KMeans
import hdbscan

def counts(mouse: int):
    with open(join(dir_name0, f"{mouse}.pkl"), 'rb') as handle:
        _, session_traj, _, _, route_traj, lap_traj, pos_traj, speed_traj, _, _, _, _, _, neural_traj = pickle.load(handle)

    #bins = np.concatenate([Father2SonGraph[i] for i in np.setdiff1d(CP_DSP[6], CP_DSP[3])])
    idx = np.where(
        #(np.isin(pos_traj, bins-1)) &
        (np.isin(route_traj, [0, 1, 2, 3, 4, 5, 6]))
    )[0]
    
    D = GetDMatrices(1, 48)
    dist_traj = D[pos_traj, 2303]
    
    with open(join(loc, f"{mouse}.pkl"), "wb") as f:
        pickle.dump([
            neural_traj, 
            session_traj, 
            route_traj, 
            lap_traj, 
            pos_traj, 
            speed_traj
        ], f)

def get_data(mouse):
    """All Data"""
    if exists(join(loc, f"{mouse}.pkl")) == False:
        counts(mouse)
        
    with open(join(loc, f"{mouse}.pkl"), 'rb') as handle:
        return pickle.load(handle)
    
def get_fate(mouse: int):
    (
        kmeans_dist_traj2, 
        neural_traj2, 
        session_traj2, 
        old_pos_traj2, 
        route_traj2, 
        lap_traj2, 
        pos_traj2, 
        speed_traj2,
        dists
    ) = get_transient_map(mouse)
    
    beg2 = np.concatenate([[0], np.where(np.ediff1d(lap_traj2) != 0)[0]+1])
    end2 = np.concatenate([np.where(np.ediff1d(lap_traj2) != 0)[0]+1, [lap_traj2.shape[0]]])
    
    mat = np.corrcoef(neural_traj2[:, beg2].T)
    avg_mat = np.zeros((7, 7))
    for i in range(7):
        for j in range(i, 7):
            idxi = np.where(route_traj2[beg2] == i)[0]
            idxj = np.where(route_traj2[beg2] == j)[0]
            idx = np.ix_(idxi, idxj)
            avg_mat[i, j] = avg_mat[j, i] = np.nanmean(mat[idx])

    median_kmean_dists = np.zeros(beg2.shape[0])
    for i in range(beg2.shape[0]):
        median_kmean_dists[i] = np.nanmean(kmeans_dist_traj2[beg2[i]:end2[i]])

    labels = np.where(median_kmean_dists > 0, 0, 1)
    return median_kmean_dists, labels

from sklearn.decomposition import PCA
from umap.umap_ import UMAP

mouse = 10224
elev, azim = 25, 30
(            
    neural_traj,
    session_traj,  
    route_traj, 
    lap_traj, 
    pos_traj, 
    speed_traj
) = get_data(mouse)

beg = np.concatenate([[0], np.where(np.ediff1d(lap_traj) != 0)[0]+1])
end = np.concatenate([np.where(np.ediff1d(lap_traj) != 0)[0]+1, [lap_traj.shape[0]]])
mean_kmean_dists, lap_labels = get_fate(mouse)

label_traj = np.zeros(lap_traj.shape[0], np.int64)
for i in range(beg.shape[0]):
    label_traj[beg[i]:end[i]] = lap_labels[i]

idx = np.where((np.isin(S2F[pos_traj], CP_DSP[6])))[0]

neural_traj = neural_traj[:, idx]
lap_traj = lap_traj[idx]
session_traj = session_traj[idx]
pos_traj = pos_traj[idx]
route_traj = route_traj[idx]
speed_traj = speed_traj[idx]
label_traj = label_traj[idx]

pca = PCA(n_components=30)
denoised_data = pca.fit_transform(neural_traj.T)
umap_model = UMAP(n_components=3, n_neighbors=20)
red_data = umap_model.fit_transform(denoised_data)

idx = np.where(route_traj == 6)[0]
neural_traj = neural_traj[:, idx]
lap_traj = lap_traj[idx]
session_traj = session_traj[idx]
pos_traj = pos_traj[idx]
route_traj = route_traj[idx]
speed_traj = speed_traj[idx]
label_traj = label_traj[idx]
red_data = red_data[idx, :]

beg = np.concatenate([[0], np.where(np.ediff1d(lap_traj) != 0)[0]+1])
end = np.concatenate([np.where(np.ediff1d(lap_traj) != 0)[0]+1, [lap_traj.shape[0]]])
  
x, y = pos_traj % 48, pos_traj // 48
x = x.astype(np.float64) + np.random.rand(x.shape[0]) - 0.5
y = y.astype(np.float64) + np.random.rand(y.shape[0]) - 0.5
D = GetDMatrices(1, 48)
vmin, vmax = np.nanmin(D[pos_traj, 2303]), np.nanmax(D[pos_traj, 2303])

fig = plt.figure(figsize=(6, 6))
ax0 = fig.add_subplot(2, 2, 1, projection='3d')
ax1 = fig.add_subplot(2, 2, 2, projection='3d')
ax2 = fig.add_subplot(2, 2, 3, projection='3d')
ax3 = fig.add_subplot(2, 2, 4)
ax0.scatter(
    red_data[:, 0],
    red_data[:, 1],
    red_data[:, 2],
    s=1,
    edgecolors=None,
    c=sns.color_palette("rainbow", as_cmap=True)((D[pos_traj, 2303] - vmin) /(vmax - vmin + 1e-8)),
)
ax0.view_init(elev=elev, azim=azim)
ax1.scatter(
    red_data[:, 0],
    red_data[:, 1],
    red_data[:, 2],
    s=1,
    edgecolors=None,
    c=MAPPaletteRGBA[label_traj, :]
)
ax1.view_init(elev=elev, azim=azim)

ax2.scatter(
    red_data[:, 0],
    red_data[:, 1],
    red_data[:, 2],
    s=1,
    edgecolors=None,
    c=sns.color_palette("rainbow", as_cmap=True)(session_traj/(6 + 1e-8)),
)
ax2.view_init(elev=elev, azim=azim)

ax3 = DrawMazeProfile(axes=ax3, color='k', linewidth=0.5)
ax3.scatter(
    x,
    y,
    s=1,
    edgecolors=None,
    c=sns.color_palette("rainbow", as_cmap=True)((D[pos_traj, 2303] - vmin) /(vmax - vmin + 1e-8)),
    alpha=0.5
)
ax3.axis([0, 48, 48, 0])
plt.show()