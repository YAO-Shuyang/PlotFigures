""" 
Provide regression analysis and correlation analysis of data.
"""
from sklearn.linear_model import LinearRegression as LR 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing as fch
import pandas as pd
from mylib.statistic_test import *

code_id = '0035 - Field Arange with Other Parameters'
loc = os.path.join(figpath, code_id)
data_loc = os.path.join(figdata, code_id)
mkdir(data_loc)
mkdir(loc)

def get_data(mice: int, maze_type: int, dates: list):
    if os.path.exists(os.path.join(data_loc, str(mice)+'-'+str(maze_type)+'.pkl')):
        with open(os.path.join(data_loc, str(mice)+'-'+str(maze_type)+'.pkl'), 'rb') as handle:
            Data = pickle.load(handle)
    else:
        Data = np.zeros((2, len(dates), 144), dtype=np.float64)
        
        for d in range(len(dates)):
            i = np.where((f1['date']==dates[d])&(f1['maze_type']==maze_type)&(f1['MiceID']==mice))[0][0]
        
            with open(f1['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)

            Data[0, d, :] = field_arange(trace)
            Data[1, d, :] = BehaviorEvents.success_rate(trace['maze_type'], 
                                spike_nodes_transform(trace['correct_nodes'], nx = 12),
                                trace['correct_time'])
        
        with open(os.path.join(data_loc, str(mice)+'-'+str(maze_type)+'.pkl'), 'wb') as f:
            pickle.dump(Data, f)
            
    return Data

dates = [20220810, 20220811, 20220812, 20220813, 20220815, 
         20220820, 20220822, 20220824, 20220826, 20220828, 20220830]
Data = get_data(11095, 1, dates)
BehaviorEventsAnalyzer.analyze(Data, 1)

dates = [20220813, 20220815, 20220817, 20220820, 20220822, 
         20220824, 20220826, 20220828, 20220830]
Data = get_data(11095, 2, dates)

dates = [20220810, 20220811, 20220812, 20220813, 20220815, 
         20220820, 20220822, 20220824, 20220826, 20220828, 20220830]
Data = get_data(11092, 1, dates)

dates = [20220813, 20220815, 20220817, 20220820, 20220822, 
         20220824, 20220826, 20220828, 20220830]
Data = get_data(11092, 2, dates)