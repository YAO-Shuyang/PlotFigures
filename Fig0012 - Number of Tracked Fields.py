from mylib.statistic_test import *

code_id = '0012 - Number of Tracked Fields'
loc = os.path.join(figpath, code_id)
mkdir(loc)

idx = np.where(f_CellReg_modi['Type'] == 'Real')[0]
if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Num', "Paradigm"], 
                              f = f_CellReg_modi, function = Number_Tracked_Fields_Interface, file_idx=idx, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
    
for i in np.unique(Data['Paradigm']):
    idx = np.where(Data['Paradigm'] == i)[0]
    print(f"{i}: {np.sum(Data['Num'][idx])}")
