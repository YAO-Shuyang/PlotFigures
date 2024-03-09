from mylib.statistic_test import *
from mylib.field.multiday_in_field import MultiDayInFieldRateChangeModel
from mylib.multiday.multiday import MultiDayLayout, MultiDayLayout2, MultiDayLayout3
from mylib.multiday.core import MultiDayCore

code_id = "0310 - Test MultiDayModel"
loc = join(figpath, 'Field Analysis', code_id)
mkdir(loc)

def plot_figures(cellreg_path: str, mouse:int, maze_type: int, session: int, save_loc: str):

    index_map = Read_and_Sort_IndexMap(
        path = cellreg_path,
        occur_num=1,
        name_label='SFP2023',
        order=np.array(['20230703', '20230705', '20230707', '20230709', '20230711', '20230713',
                    '20230715', '20230717', '20230719', '20230721', '20230724', '20230726', 
                    '20230728'])
    )

    cellnum = np.where(index_map == 0, 0, 1)
    cellnum = np.nansum(cellnum, axis=0)
    mkdir(save_loc)
    dates = [
        20230703, 20230705, 20230707, 20230709, 20230711, 20230713,
        20230715, 20230717, 20230719, 20230721, 20230724, 20230726,
        20230728
    ]
    
    core = MultiDayCore()
    core.get_trace_set(
        f=f1,
        file_indices=np.concatenate([np.where((f1['MiceID'] == mouse)&(f1['session'] == session)&(f1['date'] == d))[0] for d in dates])
    )
    
    for n in range(2, 14):
        idx = np.where(cellnum == n)[0]
        mkdir(join(save_loc, str(n)+" Cells"))

        for k, i in enumerate(idx):
            print(f"{k}/{len(idx)}, cell num = {n}")
            MultiDayLayout.visualize_cells(
                index_map=index_map,
                i=i,
                f=f1,
                mouse=mouse,
                maze_type=maze_type,
                session=session,
                dates=[20230703, 20230705, 20230707, 20230709, 20230711, 20230713,
                       20230715, 20230717, 20230719, 20230721, 20230724, 20230726,
                       20230728],
                core=core,
                save_loc=join(save_loc, str(n)+" Cells"),
                file_name="Line "+str(i+1),
                is_show=False,
                loctimecurve_kwargs = {'bar_kwargs':{'markeredgewidth': 2}},
            )

#plot_figures(cellReg_12_maze1_2, 10212,maze_type=1, session=2, save_loc=join(loc, '10212-Stage 2-Maze 1'))

"""
index_map = Read_and_Sort_IndexMap(
    path = cellReg_12_maze1_sup,
    occur_num=15,
    name_label='SFP2023',
    order=np.array([
        '20230426', '20230428', '20230430', '20230502', '20230504', '20230508',
        '20230510', '20230512', '20230515', '20230517', '20230519', '20230521',
        '20230703', '20230705', '20230707', '20230709', '20230711', '20230713',
        '20230715', '20230717', '20230719', '20230721', '20230724', '20230726', 
        '20230728'])
)
"""
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

for n in range(26, 1, -1):
    
    idx = np.where(cellnum == n)[0]
    print(n, idx)
    mkdir(join(loc, '10227-Stage 1+2-Maze 1 [footprint]', str( n)+' Cells'))
    """
    MultiDayLayout.visualize_cells(
        index_map=index_map,
        cell_pairs=idx,
        f=f1,
        #footprint_dirs=[os.path.join(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1", "SFP"+str(date)+".mat") for date in dates],
        mouse=10224,
        maze_type=1,
        session=2,
        dates=dates,
        save_loc=join(loc, '10224-Stage 1+2-Maze 1', str(n)+' Cells'),
        is_show=False,
        loctimecurve_kwargs = {'bar_kwargs':{'markeredgewidth': 3, 'markersize': 5}},
        layout_kw={"width_ratios": [2, 2, 10, 2]}
    )
    """
    MultiDayLayout3.visualize_cells(
        index_map=index_map,
        cell_pairs=idx,
        f=f1,
        footprint_dirs=[os.path.join(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1", 
                                     "SFP"+str(date)+".mat") for date in dates],
        mouse=10227,
        maze_type=1,
        session=2,
        dates=dates,
        save_loc=join(loc, '10227-Stage 1+2-Maze 1 [footprint]', str(n)+' Cells'),
        is_show=False,
        loctimecurve_kwargs = {'bar_kwargs':{'markeredgewidth': 3, 'markersize': 5}},
        layout_kw={"width_ratios": [2, 2, 2, 10]}
    )
