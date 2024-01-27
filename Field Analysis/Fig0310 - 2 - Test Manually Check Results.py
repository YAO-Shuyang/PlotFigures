from mylib.statistic_test import *
from mylib.field.multiday_in_field import MultiDayInFieldRateChangeModel
from mylib.multiday.multiday import MultiDayLayout, MultiDayLayout2, MultiDayLayout3
from mylib.multiday.core import MultiDayCore

code_id = "0310 - Test MultiDayModel"
loc = join(figpath, 'Field Analysis', code_id, 'Manually Check Results')
mkdir(loc)

f = pd.read_excel(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\caiman10227.xlsx", sheet_name='cellreg-14')
index_map = np.zeros((26, len(f)))
dates = [ 20230806, 20230808, 20230810, 20230812, 
          20230814, 20230816, 20230818, 20230820, 
          20230822, 20230824, 20230827, 20230829,
          20230901,
          20230906, 20230908, 20230910, 20230912, 
          20230914, 20230916, 20230918, 20230920, 
          20230922, 20230924, 20230926, 20230928, 
          20230930]
for i, day in enumerate(dates):
    index_map[i, :] = f[day][:]
    
index_map[np.isnan(index_map)] = 0

cellnum = np.where(index_map == 0, 0, 1)
cellnum = np.nansum(cellnum, axis=0)
print(len(np.where(cellnum==26)[0]))
#[20230806, 20230808, 20230810, 20230812, 20230814, 20230816, 20230818, 20230820, 20230822, 20230824, 20230827, 20230829, 20230901],
        # 20230426, 20230428, 20230430, 20230502, 20230504, 20230506, 20230508,20230510, 20230512, 20230515, 20230517, 20230519, 20230521,
        # 20230703, 20230705, 20230707, 20230709, 20230711, 20230713,20230715, 20230717, 20230719, 20230721, 20230724, 20230726,20230728],

for n in range(25, 1, -1):
    
    idx = np.where(cellnum == n)[0]
    mkdir(join(loc, '10227-Stage 1+2-Maze 1', str(n)+' Cells'))
    start_i = 30 if n == 26 else 0
    MultiDayLayout.visualize_cells(
        index_map=index_map,
        cell_pairs=idx[start_i:],
        f=f1,
        #footprint_dirs=[os.path.join(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1", "SFP"+str(date)+".mat") for date in dates],
        mouse=10227,
        maze_type=1,
        session=2,
        dates=dates,
        save_loc=join(loc, '10227-Stage 1+2-Maze 1', str(n)+' Cells'),
        is_show=False,
        loctimecurve_kwargs = {'bar_kwargs':{'markeredgewidth': 3, 'markersize': 5}},
        layout_kw={"width_ratios": [2, 2, 10, 2]}
    )


