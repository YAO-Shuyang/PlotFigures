from mylib.statistic_test import *
from mylib.field.field_tracker import indept_test_for_evolution_events

code_id = "0318 - Independence test for field evolution events"
loc = join(figpath, code_id)
mkdir(loc)



if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(
        variable_names = ['Training Session', 'Chi-Square Statistic', 'MI', 'Dimension', 'Pair Type', 'Pair Num',
                          'Paradigm'],
        f = f_CellReg_modi, 
        function = IndependentEvolution_Interface, 
        file_name = code_id, 
        behavior_paradigm = 'CrossMaze'
    )
