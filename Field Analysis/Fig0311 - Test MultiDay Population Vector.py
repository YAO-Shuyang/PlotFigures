from mylib.statistic_test import *
from mylib.multiday.pvc import MultiDayPopulationVectorCorrelation
from mylib.multiday.core import MultiDayCore

code_id = "0311 - Multiday Population Vector Correlation"
save_loc = join(figpath, 'Field Analysis', code_id)
mkdir(save_loc)
data_loc = join(figdata, code_id)
mkdir(data_loc)
"""Stage 2, 10209/10212
index_map = Read_and_Sort_IndexMap(
    path = cellReg_12_maze2,
    occur_num=1,
    name_label='SFP2023',
    order=np.array([
        '20230703', '20230705', '20230707', '20230709', '20230711', '20230713',
        '20230715', '20230717', '20230719', '20230721', '20230724', '20230726', 
        '20230728'])
)
mouse = 10212
session = 3
stage = 'Stage 2'
dates = [
    20230703, 20230705, 20230707, 20230709, 20230711, 20230713,
    20230715, 20230717, 20230719, 20230721, 20230724, 20230726,
    20230728
]
maze_type = 'Maze 2'
"""
"""Stage 1, 10209/10212
index_map = Read_and_Sort_IndexMap(
    path = cellReg_09_maze1,
    occur_num=1,
    name_label='SFP2023',
    order=np.array(['20230426', '20230428', '20230430', '20230502', '20230504', '20230506',
                    '20230508', '20230510', '20230512', '20230515', '20230517', '20230519',
                    '20230521'])
)
mouse = 10209
session = 2
stage = 'Stage 1'
dates = [
    20230426, 20230428, 20230430, 20230502, 20230504, 20230506,
    20230508, 20230510, 20230512, 20230515, 20230517, 20230519,
    20230521
]
maze_type = 'Maze 1'
"""

#"""Stage 1, 10209/10212
index_map = Read_and_Sort_IndexMap(
    path = cellReg_92_maze2,
    occur_num=1,
    name_label='SFP2022',
    order=np.array(['20220820', '20220822', '20220824', '20220826', '20220828', '20220830'])
)
mouse = 11092
session = 3
stage = 'Stage 2'
dates = [
    20220820, 20220822, 20220824, 20220826, 20220828, 20220830
]
maze_type = 'Maze 2'


file_indices = np.concatenate([np.where((f1['MiceID'] == mouse)&(f1['session'] == session)&(f1['date'] == d))[0] for d in dates])

core = MultiDayCore.get_core(
    f=f1,
    file_indices=file_indices,
    keys = ['old_map_clear', 'maze_type', 'is_placecell']
)

PVCModel = MultiDayPopulationVectorCorrelation(
    f=f1,
    file_indices=file_indices,
    keys = ['old_map_clear', 'maze_type', 'is_placecell'],
    core=core
)
PVCModel.get_pvc(
    f=f1,
    file_indices=file_indices,
    index_map=index_map,
    occu_num=6,
    align_idx=np.arange(6)
)

PVCModel.visualize(
    save_loc=save_loc,
    file_name=str(mouse)+'-'+stage+'-'+maze_type,
    cmap = 'jet',
    vmin=0,
    vmax=1
)

PVCModel.visualize_pvc_heatmap(
    save_loc=save_loc,
    file_name=str(mouse)+'-'+stage+'-'+maze_type+" PVC Heatmap",
    cmap = 'jet',
    vmin=0,
    vmax=0.8
)

with open(join(data_loc, str(mouse)+'-'+stage+'-'+maze_type+".pkl"), 'wb') as f:
    pickle.dump(PVCModel.PVC, f)

