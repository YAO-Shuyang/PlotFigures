from mylib.statistic_test import *
from mylib.field.multiday_in_field import MultiDayInFieldRateChangeModel
from mylib.multiday.multiday import MultiDayLayout, MultiDayLayout2, MultiDayLayout3
from mylib.multiday.visualize.plot_fields import MultiDayFields
from mylib.multiday.core import MultiDayCore

code_id = "0326 - Long-term example cells with marked place fields"
loc = join(figpath, code_id)
mkdir(loc)

# index_map = GetMultidayIndexmap(mouse=10224, stage='Stage 1+2', session=2, occu_num=2)
with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10227\Maze1-footprint\trace_mdays_conc.pkl", 'rb') as handle:
    trace = pickle.load(handle)
    

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
          20230930] #[20230806, 20230808, 20230810, 20230812, 20230814, 20230816, 20230818, 20230820, 20230822, 20230824, 20230827, 20230829, 20230901],
        # 20230426, 20230428, 20230430, 20230502, 20230504, 20230506, 20230508,20230510, 20230512, 20230515, 20230517, 20230519, 20230521,
        # 20230703, 20230705, 20230707, 20230709, 20230711, 20230713,20230715, 20230717, 20230719, 20230721, 20230724, 20230726,20230728],

print(len(trace['place_field_all']), trace['index_map'].shape[1])

for n in range(26, 1, -1):
    
    idx = np.where(cellnum == n)[0]
    print(n, idx)
    mkdir(join(loc, '10227-Stage 1+2-Maze 1', str( n)+' Cells'))

    MultiDayFields.visualize_cells(
        index_map=index_map,
        cell_pairs=idx,
        field_info=trace['field_info'],
        field_reg=trace['field_reg'],
        place_field_all=trace['place_field_all'],
        f=f1,
        #footprint_dirs=[os.path.join(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1", 
        #                             "SFP"+str(date)+".mat") for date in dates],
        mouse=10227,
        maze_type=1,
        session=2,
        dates=dates,
        save_loc=join(loc, '10227-Stage 1+2-Maze 1', str(n)+' Cells'),
        is_show=False,
        loctimecurve_kwargs = {'bar_kwargs':{'markeredgewidth': 3, 'markersize': 5},
                               'line_kwargs':{'linewidth': 0.1, 'color': 'gray'}}
    )