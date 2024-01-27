from mylib.statistic_test import *


with open(r"E:\Data\FigData\0050 Lap-wise Traveling Distance.pkl", 'rb') as handle:
    Data = pickle.load(handle)
    
Maze1Data = SubDict(Data, Data.keys(), np.where(Data['Maze Type'] == 'Maze 1')[0])
Maze2Data = SubDict(Data, Data.keys(), np.where(Data['Maze Type'] == 'Maze 2')[0])

# Maze 1 - 6.3m
behav_nodes = np.array([])
stay_time = np.array([])

for i in tqdm(np.argpartition(Maze1Data['Lap-wise Distance'], 50)[:50]):
    idx = np.where((f1['MiceID'] == Maze1Data['MiceID'][i])&(f1['date'] == Maze1Data['date'][i])&(f1['session'] == 2))[0][0]
    with open(f1['Trace File'][idx], 'rb') as handle:
        trace = pickle.load(handle)
        
    beg, end = LapSplit(trace, trace['paradigm'])
    behav_nodes = np.concatenate([behav_nodes, spike_nodes_transform(trace['correct_nodes'][beg[int(Maze1Data['Lap ID'][i])-1]:end[int(Maze1Data['Lap ID'][i])-1]], 12)])
    stay_time = np.concatenate([stay_time, np.ediff1d(trace['correct_time'])[beg[int(Maze1Data['Lap ID'][i])-1]:end[int(Maze1Data['Lap ID'][i])-1]]])
  
CP = correct_paths[1]
frac = np.zeros(CP.shape[0], np.float64)
for i in range(frac.shape[0]):
    idx = np.where(behav_nodes == CP[i])[0]
    frac[i] = np.nansum(stay_time[idx])

frac = frac / np.sum(frac)
cum_frac = np.cumsum(frac)*6.3
with open(r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\dismat\maze1_rev.pkl", "wb") as f:
    pickle.dump(cum_frac, f)

plt.bar(np.arange(111), cum_frac)
plt.show()

# Maze 1 - 5.7m
behav_nodes = np.array([])
stay_time = np.array([])

for i in tqdm(np.argpartition(Maze2Data['Lap-wise Distance'], 50)[:50]):
    idx = np.where((f1['MiceID'] == Maze2Data['MiceID'][i])&(f1['date'] == Maze2Data['date'][i])&(f1['session'] == 3))[0][0]
    with open(f1['Trace File'][idx], 'rb') as handle:
        trace = pickle.load(handle)
        
    beg, end = LapSplit(trace, trace['paradigm'])
    behav_nodes = np.concatenate([behav_nodes, spike_nodes_transform(trace['correct_nodes'][beg[int(Maze2Data['Lap ID'][i])-1]:end[int(Maze2Data['Lap ID'][i])-1]], 12)])
    stay_time = np.concatenate([stay_time, np.ediff1d(trace['correct_time'])[beg[int(Maze2Data['Lap ID'][i])-1]:end[int(Maze2Data['Lap ID'][i])-1]]])
  
CP = correct_paths[1]
frac = np.zeros(CP.shape[0], np.float64)
for i in range(frac.shape[0]):
    idx = np.where(behav_nodes == CP[i])[0]
    frac[i] = np.nansum(stay_time[idx])

frac = frac / np.sum(frac)
cum_frac = np.cumsum(frac)*5.7
with open(r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\dismat\maze2_rev.pkl", "wb") as f:
    pickle.dump(cum_frac, f)
    
plt.bar(np.arange(101), cum_frac)
plt.show()
