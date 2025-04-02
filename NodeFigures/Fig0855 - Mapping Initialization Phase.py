from mylib.statistic_test import *
from mazepy.datastruc.neuact import NeuralTrajectory, SpikeTrain
from mazepy.datastruc.variables import VariableBin
from mylib.calcium.dsp_ms import classify_lap
from mazepy.basic._time_sync import _coordinate_recording_time

code_id = "0855 - Mapping Initialization Phase"
loc = os.path.join(figpath, "Dsp", code_id)
mkdir(loc)

import scipy.stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

dir_name = join(loc, "Dynamic Analysis")
mkdir(dir_name)

temp_bins = [
    np.array([]),
    np.array([33, 32, 44, 45, 46, 47, 48]), #24, 23, 22, 34, 
    np.array([75, 74, 62, 50, 51, 39, 38, 37, 49, 61, 73]), # 54, 66, 65,64,63, 
    np.array([99, 87, 88, 76, 77, 89, 101, 102, 114, 113, 125]),
    np.array([8, 7, 6, 18, 17, 29, 30, 31, 19, 20]),
    np.array([104, 103, 91, 90, 78, 79, 67, 55, 54, 66, 65, 64, 63, 75]), #93, 105, 106,94, 82, 81, 89, 92, 
    np.array([135, 134, 133, 121, 97, 109, 110, 122, 123, 111, 112, 100, 99, 87, 88, 76, 77])
]    

def get_init_data(mouse):
    if exists(join(dir_name, f"{mouse}.pkl")):
        with open(join(dir_name, f"{mouse}.pkl"), 'rb') as handle:
            return pickle.load(handle)
        
(
    neural_traj,
    pos_traj,
    time_traj,
    route_traj,
    session_traj,
    speed_traj,
    lap_traj,
    len_traj,
    retrieval_traj
) = get_init_data(mouse=10232)

s = 5
r = 2
idx0 = np.where((session_traj == s) & (np.isin(S2F[pos_traj], temp_bins[r])) & (retrieval_traj == 1))[0]
idx1 = np.where((session_traj == s) & (np.isin(route_traj, [r]) & (np.isin(S2F[pos_traj], temp_bins[r])) & (retrieval_traj == 0)))[0]
idx2 = np.where(session_traj[idx0] == s)[0]
 
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from umap.umap_ import UMAP

model = PCA(n_components=30)
denoised_data = model.fit_transform(neural_traj.T)
lda = LDA(n_components=6)
red_data = lda.fit_transform(denoised_data[idx0, :], pos_traj[idx0])
#umap_model = UMAP(n_components=10)
#red_data = umap_model.fit_transform(red_data)
D = GetDMatrices(1, 48)

vmax, vmin = np.max(D[pos_traj[idx0], 2303]), np.min(D[pos_traj[idx0], 2303])

beg = np.concatenate([[0], np.where(np.ediff1d(lap_traj[idx1])!=0)[0]+1])
end = np.concatenate([np.where(np.ediff1d(lap_traj[idx1])!=0)[0]+1, [len(lap_traj[idx1])]])

colors = sns.color_palette("Spectral", len(beg))
fig = plt.figure(figsize=(3, 3))
ax = Clear_Axes(fig.add_subplot(111), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
for i in range(len(beg)):
    res = lda.transform(denoised_data[idx1[beg[i]:end[i]], :])

    ax.scatter(
        red_data[idx2, 0],
        red_data[idx2, 1],
        s=3,
        edgecolors=None,
        c=sns.color_palette("rainbow", as_cmap=True)((D[pos_traj[idx0[idx2]], 2303] - vmin) /(vmax - vmin + 1e-8)*0.8),
        zorder=1
    )
    
    ax.plot(
        res[:, 0],
        res[:, 1],
        lw=1,
        c='k',
        zorder=2
    )
    ax.plot(
        [res[0, 0]],
        [res[0, 1]],
        'o',
        c='red',
        markersize=5,
        markeredgewidth=0,
        zorder=2
    )
    """"""
    ax.plot(
        [res[-1, 0]],
        [res[-1, 1]],
        '^',
        c='k',
        markersize=5,
        markeredgewidth=0,
        zorder=2
    )
"""
R1
ax.azim = -90
ax.elev = 90
"""
ax.azim = 0
ax.elev = 90
plt.show()