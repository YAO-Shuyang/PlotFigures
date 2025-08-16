from mylib.statistic_test import *
from mylib.field.multiday_in_field import MultiDayInFieldRateChangeModel
from mylib.multiday.multiday import MultiDayLayout, MultiDayLayout2, MultiDayLayout3
from mylib.multiday.visualize.open_field import MultiDayLayoutOpenField
from mylib.multiday.core import MultiDayCore

code_id = "0310 - Test MultiDayModel"
loc = join(figpath, 'Field Analysis', code_id)
mkdir(loc)

# index_map = GetMultidayIndexmap(mouse=10224, stage='Stage 1+2', session=2, occu_num=2)
index_map = GetMultidayIndexmap(mouse=10227, stage='Stage 1+2', session=2, occu_num=2, f=f_CellReg_modi)
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
"""
for n in range(26, 1, -1):
    
    idx = np.where(cellnum == n)[0]
    print,(n, idx)
    mkdir(join(loc, '10227-Stage 1+2-Maze 1 [footprint]', str( n)+' Cells'))

    MultiDayLayout.visualize_cells(
        index_map=index_map,
        cell_pairs=idx,
        f=f1,
        #footprint_dirs=[os.path.join(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1", 
        #                             "SFP"+str(date)+".mat") for date in dates],
        mouse=10227,
        maze_type=1,
        session=2,
        dates=dates,
        save_loc=join(loc, '10227-Stage 1+2-Maze 1 [footprint]', str(n)+' Cells'),
        is_show=False,
        loctimecurve_kwargs = {'bar_kwargs':{'markeredgewidth': 3, 'markersize': 5},
                               'line_kwargs':{'linewidth': 0.1, 'color': 'gray'}},
        layout_kw={"width_ratios": [2, 2, 10,1]}
    )
"""

# Reverse dates
# dates = [20230605, 20230606, 20230607, 20230608, 20230609, 20230613, 20230614, 20230615, 20230616, 20230617, 20230618, 20230619]
# dates = [20230613, 20230614, 20230615, 20230616, 20230617, 20230618, 20230619]
dates = [
    20240829, 20240831, 20240902, 20240904, 20240906,
    20240908, 20240910, 20240912, 20240914,
    20240916, 20240918, 20240920, 20240922,
    20240924, 20240926, 20240928, 20240930,
    20241003, 20241004, 20241006, 20241008,
    20241011, 20241013, 20241015, 20241017,
    20241019
]#[20231017, 20231018, 20231019, 20231020, 20231021, 20231022, 20231023]
mouse = 10232
index_map = GetMultidayIndexmap(mouse=mouse, stage='Stage 1+2', session=1, occu_num=2, f=f_CellReg_modi)
index_map[np.where(np.isnan(index_map))] = 0
index_map = index_map.astype(np.int64)

#cellnum = np.where(index_map == 0, 0, 1)
cellnum = np.where(index_map > 0, 1, 0)
cellnum = np.nansum(cellnum, axis=0)
print(index_map.shape)


for n in range(26, 1, -1):
    
    idx = np.where(cellnum == n)[0]
    print(n, idx)
    p = join(loc, f'{mouse}-HMP', str(n)+' Cells')
    mkdir(p)


    """ 
    MultiDayLayout.visualize_cells(
        index_map=index_map,
        cell_pairs=idx,
        f=f4,
        #footprint_dirs=[os.path.join(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1", 
        #                             "SFP"+str(date)+".mat") for date in dates],
        mouse=mouse,
        maze_type=3,
        session=1,
        dates=dates,
        save_loc=p,
        is_show=False,
        loctimecurve_kwargs = {'bar_kwargs':{'markeredgewidth': 0.75, 'markersize': 5},
                               'line_kwargs':{'linewidth': 0.1, 'color': 'gray'}},
        layout_kw={"width_ratios": [2, 2, 10,1]},
        paradigm='HairpinMaze',
        direction='cis'
    )
    """
    MultiDayLayout.visualize_cells(
        index_map=index_map,
        cell_pairs=idx,
        f=f4,
        #footprint_dirs=[os.path.join(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1", 
        #                             "SFP"+str(date)+".mat") for date in dates],
        mouse=mouse,
        maze_type=3,
        session=1,
        dates=dates,
        save_loc=p,
        is_show=False,
        loctimecurve_kwargs = {'bar_kwargs':{'markeredgewidth': 0.75, 'markersize': 5},
                               'line_kwargs':{'linewidth': 0.1, 'color': 'gray'}},
        layout_kw={"width_ratios": [2, 2, 10,1]},
        paradigm='HairpinMaze',
        direction='trs'
    )
   