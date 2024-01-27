from mylib.statistic_test import *
from mylib.multiday.field_tracker import field_overlapping

code_id = "0315 - Field Overlap With Models"
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = {"MiceID": np.array([], np.int64), "Maze Type": np.array([]), 
            "Start Session": np.array([], np.int64), "Overlapping": np.array([], np.float64)}
    
    for i in range(len(f_CellReg_day)):
        if f_CellReg_day['include'][i] == 0 or f_CellReg_day['maze_type'][i] == 0:
            continue
        
        with open(f_CellReg_day['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
            
        overlapping = field_overlapping(trace)
        
        retained_dur = np.arange(overlapping.shape[0]-1)
        
        Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(int(f_CellReg_day['MiceID'][i]), overlapping.shape[0]-1)])
        Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat("Maze "+str(int(f_CellReg_day['maze_type'][i])), overlapping.shape[0]-1)])
        Data['Start Session'] = np.concatenate([Data['Start Session'], retained_dur])
        print(overlapping[(np.arange(1, overlapping.shape[0]), np.arange(overlapping.shape[0]-1))])
        Data['Overlapping'] = np.concatenate([Data['Overlapping'], overlapping[(np.arange(1, overlapping.shape[0]), np.arange(overlapping.shape[0]-1))]*100])
        
        del trace
        gc.collect()
        
    with open(join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
        
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)
    

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(x="Start Session", y="Overlapping", hue="Maze Type", data=Data, ax=ax)
plt.show()
