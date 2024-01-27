from mylib.statistic_test import *

code_id = '0052 - First Exposure'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Average Velocity', 'Time'], is_behav=True,
                              f = f_pure_behav, function = FirstExposure_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
    
    with open(join(figdata, code_id+'.pkl'), 'wb') as f:
        pickle.dump(Data, f)
    
def get_data(Data: dict, Stage: str, maze: str):
    v_op = []
    t_ms = []
    
    mice = np.unique(Data['MiceID'])
    print(mice)
    for m in mice:
        
        try:
            maze_idx = np.where((Data['Stage'] == Stage)&(Data['Maze Type'] == maze)&(Data['Training Day'] == 'Day 1')&(Data['MiceID'] == m))[0][0]
            #open_idx = np.where((Data['Stage'] == Stage)&(Data['Maze Type'] == 'Open Field')&(Data['Training Day'] == 'Day 1')&(Data['MiceID'] == m))[0][0]
        
            v_op.append(Data['Average Velocity'][maze_idx])
            t_ms.append(Data['Time'][maze_idx])
        except:
            print(m, maze)
        
    return v_op, t_ms

v_op, t_ms = get_data(Data, 'Stage 1', 'Maze 1')

print(t_ms)

fig = plt.figure(figsize=(4,4))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(v_op, t_ms, 'o')
plt.show()