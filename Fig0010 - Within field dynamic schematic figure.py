from mylib.statistic_test import *


XMIN = 0
XMAX = 2
XCENTER = 1

DENSE_LENGTH = 1000
SPARSE_LENGTH = 21

SIGMA = 3
PROB_MAX = 0.7
FIELD_THRE = 0.05

x = np.linspace(XMIN, XMAX, DENSE_LENGTH)
prob = Gaussian(x-XCENTER, sigma=SIGMA, nx=2)
prob = prob/np.max(prob)*PROB_MAX

frames_x = np.linspace(XMIN, XMAX, SPARSE_LENGTH)
field_indices = np.where(prob>=FIELD_THRE)[0]

with open(r"E:\Data\Cross_maze\10209\20230728\session 2\trace.pkl", 'rb') as handle:
    trace = pickle.load(handle)

CP = cp.deepcopy(correct_paths[trace['maze_type']])  
father_field = [30, 31, 19, 20, 21, 9, 10, 11, 12, 24, 23]
LEF, RIG = np.where(CP == 30)[0]+0.5, np.where(CP == 23)[0]+1.5
cell = 7

frame_labels = get_spike_frame_label(
    ms_time=cp.deepcopy(trace['correct_time']), 
    spike_nodes=cp.deepcopy(trace['correct_nodes']),
    trace=trace, 
    behavior_paradigm='CrossMaze',
    window_length = 1
)
    
behav_indices = np.where(frame_labels==1)[0]
behav_nodes = spike_nodes_transform(trace['correct_nodes'], nx = 12)[behav_indices]
behav_time = cp.deepcopy(trace['correct_time'])[behav_indices]

in_field_indices = np.sort(np.concatenate([np.where(behav_nodes == i)[0] for i in father_field]))

in_field_nodes = behav_nodes[in_field_indices]
in_field_time = behav_time[in_field_indices]

raw_traces = []
ms_time = []
linearized_x = np.zeros_like(in_field_nodes.shape[0], dtype=np.float64)
for i in range(len(linearized_x)):
    linearized_x[i] = np.where(CP==in_field_nodes[i])[0]+1
    
linearized_x = linearized_x + np.random.rand(linearized_x.shape[0])-0.5

dt = np.ediff1d(in_field_time)
idx = np.concatenate([[0], np.where(dt > 200)[0], in_field_time.shape[0]])
for i in range(idx.shape[0]-1):
    ms_idx = np.where((trace['ms_time'] >= in_field_nodes[idx[i]])&(trace['ms_time'] <= in_field_nodes[idx[i+1]]))[0]
    ms_time.append(trace['ms_time'][ms_idx] - trace['ms_time'][ms_idx[0]])
    raw_traces.append(trace['RawTraces'][ms_idx] + in_field_time[idx[i]]/1000)
    
    spike_idx = np.where((trace['ms_time_behav'] >= in_field_nodes[idx[i]])&(trace['ms_time_behav'] <= in_field_nodes[idx[i+1]])&(trace['Spikes'][cell-1, :]==1))[0]
    spike_time = trace['ms_time_behav'][spike_idx]
    proj_to_behav_idx = np.zeros_like(spike_time)
    for j in range(len(spike_time)):
        a, b = np.where(in_field_time <= spike_time[j])[0][-1], np.where(in_field_time >= spike_time[j])[0][0]
        proj_to_behav_idx[j] = a if np.abs(in_field_time[a] - spike_time[j]) < np.abs(in_field_time[b] - spike_time[j]) else b
    
    
    