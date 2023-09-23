from mylib.statistic_test import *
from matplotlib.axes import Axes
from mylib.maze_graph import maze1_graph as MG1
from mylib.maze_graph import maze2_graph as MG2
from mylib.maze_graph import CorrectPath_maze_1 as CP1
from mylib.maze_graph import CorrectPath_maze_2 as CP2
from mylib.maze_graph import IncorrectPath_maze_1 as IP1
from mylib.maze_graph import IncorrectPath_maze_2 as IP2
from mylib.maze_graph import xorder1, xorder2, S2F
from mylib.maze_graph import DecisionPoint1_Linear as DP1
from mylib.maze_graph import DecisionPoint2_Linear as DP2
from mylib.maze_graph import DecisionPoint2WrongGraph1 as WG1
from mylib.maze_graph import DecisionPoint2WrongGraph2 as WG2

code_id = '0034 - Place Field Distribution Lineplot'
loc = os.path.join(figpath, code_id)
data_loc = os.path.join(figdata, code_id)

def get_data(mice: int, maze_type: int, dates: list):
    if os.path.exists(os.path.join(data_loc, str(mice)+'-'+str(maze_type)+'.pkl')):
        with open(os.path.join(data_loc, str(mice)+'-'+str(maze_type)+'.pkl'), 'rb') as handle:
            Data = pickle.load(handle)
    else:
        Data = np.zeros((len(dates), 144), dtype=np.float64)
        
        for d in range(len(dates)):
            i = np.where((f1['date']==dates[d])&(f1['maze_type']==maze_type)&(f1['MiceID']==mice))[0][0]
        
            with open(f1['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)

            Data[d, :] = field_arange(trace)
        
        with open(os.path.join(data_loc, str(mice)+'-'+str(maze_type)+'.pkl'), 'wb') as f:
            pickle.dump(Data, f)
            
    return Data
'''
for i in range(len(f1)):
    if f1['maze_type'][i] == 0:
        continue
    
    with open(f1['Trace File'][i], 'rb') as handle:
        trace = pickle.load(handle)
    
    FieldDisImage.plot_figure_fr_trace(trace, save_loc=os.path.join(f1['Path'][i], 'PeakCurve'), file_name='place_field_arrangement')
    FieldDisImage.plot_figure_fr_trace(trace, save_loc=os.path.join(loc, 'each session'), 
                                       file_name = str(trace['MiceID'])+'-Maze '+str(trace['maze_type'])+'-'+str(trace['date']))

'''
dates = [
         20220820, 20220822, 20220824, 20220826, 20220828, 20220830]
Data = get_data(11095, 1, dates)
Data = (Data.T / np.nansum(Data, axis = 1)).T * 100
print("Plot 11095, Maze 1")
labels = ['Day 4', 'Day 5', 'Day 6', 
          'Day 7', 'Day 8', 'Day 9']
FieldDisImage.plot_figure(maze_type=1, data=Data, 
                    save_loc=loc, file_name='Maze 1-11095',
                    twin_data = None, 
                    cp_args = {'labels':labels},
                    ip_args = {})
# plot_field_arange_all(Data, maze_type=1, save_loc=loc, file_name='Maze 1-11095', labels=labels, title='Maze 1, Mouse #11095')
print("  Done.")
Mean = np.nanmean(Data, axis = 0)
print('95 open field - maze 1',np.nanmean(Mean) + np.nanstd(Mean)*3)


dates = [20220820, 20220822, 
         20220824, 20220826, 20220828, 20220830]
Data = get_data(11095, 2, dates)
Data = (Data.T / np.nansum(Data, axis = 1)).T * 100
print("Plot 11095, Maze 2")
labels = ['Day 4', 'Day 5', 'Day 6',
          'Day 7', 'Day 8', 'Day 9']
FieldDisImage.plot_figure(maze_type=2, data=Data, 
                    save_loc=loc, file_name='Maze 2-11095',
                    twin_data = None, 
                    cp_args = {'labels':labels},
                    ip_args = {})
# plot_field_arange_all(Data, maze_type=2, save_loc=loc, file_name='Maze 2-11095', labels=labels, title='Maze 2, Mouse #11095')
print("  Done.") 
Mean = np.nanmean(Data, axis = 0)
print('95 open field - maze 2',np.nanmean(Mean) + np.nanstd(Mean)*3)

dates = [
         20220820, 20220822, 20220824, 20220826, 20220828, 20220830]
Data = get_data(11092, 1, dates)
Data = (Data.T / np.nansum(Data, axis = 1)).T * 100
print("Plot 11092, Maze 1")
labels = ['Day 4', 'Day 5', 'Day 6',
          'Day 7', 'Day 8', 'Day 9']
FieldDisImage.plot_figure(maze_type=1, data=Data, 
                    save_loc=loc, file_name='Maze 1-11092',
                    twin_data = None, 
                    cp_args = {'labels':labels},
                    ip_args = {})
# plot_field_arange_all(Data, maze_type=1, save_loc=loc, file_name='Maze 1-11092', labels=labels, title='Maze 1, Mouse #11092')
print("  Done.")
Mean = np.nanmean(Data, axis = 0)
print('92 open field - maze 1',np.nanmean(Mean) + np.nanstd(Mean)*3)

dates = [20220820, 20220822, 
         20220824, 20220826, 20220828, 20220830]
Data = get_data(11092, 2, dates)  
Data = (Data.T / np.nansum(Data, axis = 1)).T * 100    
print("Plot 11092, Maze 2")
labels = ['Day 4', 'Day 5', 'Day 6',
          'Day 7', 'Day 8', 'Day 9']
FieldDisImage.plot_figure(maze_type=2, data=Data, 
                    save_loc=loc, file_name='Maze 2-11092',
                    twin_data = None, 
                    cp_args = {'labels':labels},
                    ip_args = {})
# plot_field_arange_all(Data, maze_type=2, save_loc=loc, file_name='Maze 2-11092', labels=labels, title='Maze 2, Mouse #11092')
print("  Done.")
Mean = np.nanmean(Data, axis = 0)
print('92 open field - maze 2',np.nanmean(Mean) + np.nanstd(Mean)*3)




    