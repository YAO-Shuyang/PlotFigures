from mylib.statistic_test import *

code_id = '0824 - Route Specific Place Fields and Allocentric Place Fields'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Segments', 'Routes', 'Group', 'Place Fields', 'SubSpace Type'],
                              f=f2, 
                              function = RouteSpecificPlaceFields_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)