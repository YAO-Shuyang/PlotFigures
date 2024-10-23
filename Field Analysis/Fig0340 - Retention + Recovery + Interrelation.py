from mylib.statistic_test import *
from mylib.field.tracker_v2 import Tracker2d

code_id = '0340 - Retention + Recovery + Interrelation'
loc = join(figpath, code_id)
mkdir(loc)

idx = np.where((f_CellReg_modi['Type'] == 'Real')&((f_CellReg_modi['maze_type'] != 0) | ((f_CellReg_modi['maze_type'] == 0)&(f_CellReg_modi['paradigm'] != 'CrossMaze'))))[0]
if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['I', 'A', 'P', 'Paradigm'], 
                             f = f_CellReg_modi, f_member=['Type'], file_idx=idx,
                             function = Retention_Recovery_Interrelation_Interface, 
                             file_name = code_id, behavior_paradigm = 'CrossMaze')

paradigms = ['MA', 'MB', 'MAf', 'MAb', 'HPf', 'HPb']

for paradigm in paradigms:
    idx = np.where(Data['Paradigm'] == paradigm)[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    
    fig = plt.figure(figsize=(4, 3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    sns.lineplot(
        x = 'I', y = 'P', hue = 'A', data = SubData, palette="rainbow"
    )
    ax.set_yticks(np.linspace(0, 1, 6))
    plt.show()