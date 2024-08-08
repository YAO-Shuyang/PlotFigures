from mylib.statistic_test import *

code_id = '0328 - illustration for place field tracking'
loc = join(figpath, code_id)
mkdir(loc)

with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10209\Maze1-footprint\trace_mdays_conc.pkl", 'rb') as handle:
    trace = pickle.load(handle)
    
print(trace.keys())
index_map = trace['index_map']
index_map[np.where(np.isnan(index_map))] = 0
index_map = index_map.astype(np.int64)

#cellnum = np.where(index_map == 0, 0, 1)
cellnum = np.where(index_map > 0, 1, 0)
cellnum = np.nansum(cellnum, axis=0)

dates = [ 20230806, 20230808, 20230810, 20230812, 
          20230814, 20230816, 20230818, 20230820, 
          20230822, 20230824, 20230827, 20230829,
          20230901,
          20230906, 20230908, 20230910, 20230912, 
          20230914, 20230916, 20230918, 20230920, 
          20230922, 20230924, 20230926, 20230928, 
          20230930]

idx = 