from mylib.statistic_test import *
from mylib.field.field_tracker import indept_test_for_evolution_events

code_id = "0325 - Coordination Index"
loc = join(figpath, code_id)
mkdir(loc)

idx = np.where(f_CellReg_modi['Type'] == 'Real')[0]
if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(
        variable_names = ['Training Session', 'Coordinate Index', 'Dimension', 'Pair Type', 'Pair Num',
                          'Paradigm'],
        f = f_CellReg_modi, f_member=['Type'],
        function = CoordinateIndex_Interface, 
        file_name = code_id, file_idx = idx,
        behavior_paradigm = 'CrossMaze'
    )