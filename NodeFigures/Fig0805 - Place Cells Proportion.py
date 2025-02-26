from mylib.statistic_test import *

code_id = "0805 - Place Cells Proportion"
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Proportion'],
                              f=f2, 
                              function = PlaceCellsProportion_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:    
        Data = pickle.load(handle)
        
print_estimator(Data['Proportion'][Data['MiceID'] != 10209])