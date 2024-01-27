from mylib.statistic_test import *

code_id = '0070 - Cell Number - Reverse&Hairpin'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Cell Number', 'Place Cell Number', 'Direction'], 
                              f = f3, function = CellNum_Reverse_Interface, 
                              file_name = code_id, behavior_paradigm = 'ReverseMaze')

if os.path.exists(join(figdata, code_id+' Hairpin.pkl')):
    with open(join(figdata, code_id+' Hairpin.pkl'), 'rb') as handle:
        HPData = pickle.load(handle)
else:
    HPData = DataFrameEstablish(variable_names = ['Cell Number', 'Place Cell Number', 'Direction'], 
                              f = f4, function = CellNum_Reverse_Interface, 
                              file_name = code_id+' Hairpin', behavior_paradigm = 'HairpinMaze')
    
print_estimator(Data['Cell Number'][np.where(Data['Direction'] == 'cis')[0]])
print_estimator(HPData['Cell Number'][np.where(HPData['Direction'] == 'trs')[0]])