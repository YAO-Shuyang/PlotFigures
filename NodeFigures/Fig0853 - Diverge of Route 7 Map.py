from mylib.statistic_test import *

code_id = "0853 - Diverge of Route 7 Map"
loc = os.path.join(figpath, code_id)
mkdir(loc)

from mylib.statistic_test import *
from mazepy.datastruc.neuact import SpikeTrain, TuningCurve, NeuralTrajectory
from mazepy.datastruc.variables import VariableBin
from mylib.calcium.dsp_ms import classify_lap
from mazepy.basic._time_sync import _coordinate_recording_time

# Segment of Interests
SIG1 = np.array([4, 5, 8, 7, 6, 18, 17, 29, 30, 31, 19, 20, 21, 9, 10, 11, 12, 24])
SIG2 = np.array([23, 22, 34, 33, 32, 44, 45, 46, 47, 48, 60, 59, 58, 57, 56])
SIG3 = np.array([84, 83, 95, 93, 105, 106, 94, 82, 81, 80, 92, 104, 103, 91, 90, 78, 79, 67, 55, 54, 66, 65, 64, 63, 75, 74, 62, 
                 50, 51, 39, 38, 37, 49])
SIG4 = np.array([61, 73, 85, 97, 135, 134, 133, 121, 109, 110, 122, 123, 111, 112, 100])


def get_neural_traj(trace: dict, is_shuffle=False):    
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
        
        if is_shuffle:
            for j in range(Spikes.shape[0]):
                Spikes[j, :] = Spikes[j, np.random.permutation(Spikes.shape[1])]
        
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
import hdbscan

def counts(traces, index_map, is_shuffle=False):
    count_map = np.where(index_map > 0, 1, 0)
    sums = np.sum(count_map, axis=0)
    idx = np.where(sums == 7)[0]
    
    neural_trajs, pos_trajs, route_trajs, lap_trajs, map_trajs, speed_trajs = [], [], [], [], [], []
    session_trajs = []
    for i in range(len(traces)):
        neural_traj, pos_traj, route_traj, lap_traj, map_traj, speed_traj = get_neural_traj(traces[i], is_shuffle=is_shuffle)
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

    idx = np.concatenate([
        np.where(
            (np.isin(pos_traj, np.concatenate([Father2SonGraph[j] for j in np.setdiff1d(CP_DSP[6], CP_DSP[3])]))) &
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
    print(neural_traj.shape, pos_traj.shape, route_traj.shape, lap_traj.shape)

    D = GetDMatrices(1, 48)
    dist_traj = D[pos_traj, 2303]
    dists = (dist_traj // 2).astype(np.int64)

    pca = PCA(n_components=30, random_state=42)
    denoised_data = pca.fit_transform(neural_traj.T)
    model = UMAP(n_components=3)
    reduced_data = model.fit_transform(denoised_data)

    with open(join(loc, f"{traces[0]['MiceID']}.pkl"), "wb") as f:
        pickle.dump([
            neural_traj, pos_traj, route_traj, lap_traj, session_traj, 
            map_traj, speed_traj, reduced_data
        ], f)

def get_data(mouse):
    if exists(join(loc, f"{mouse}.pkl")) == False:
        traces = []
        for i in np.where(f2['MiceID'] == mouse)[0]:
            with open(f2['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
            traces.append(trace)

        with open(f_CellReg_dsp['cellreg_folder'][1+m], 'rb') as handle:
            index_map = pickle.load(handle)
            
            if mouse != 10232:
                index_map = index_map[1:, :]
            
        counts(traces, index_map.astype(np.int64), is_shuffle=True)
        
    with open(join(loc, f"{mouse}.pkl"), 'rb') as handle:
        return pickle.load(handle)
          
(            
    neural_traj, 
    pos_traj, 
    route_traj, 
    lap_traj, 
    session_traj, 
    map_traj, 
    speed_traj, 
    reduced_data
) = get_data(10232)

fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(projection='3d')
ax.scatter(
    reduced_data[:, 0],
    reduced_data[:, 1],
    reduced_data[:, 2],
    s=5,
    linewidth = 0,
    color=DSPPaletteRGBA[route_traj, :]
)
plt.show()