from mylib.statistic_test import *

code_id = '0064 - Place Field Num - Reverse&Hairpin'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Field Number', 'Direction'], 
                              f = f3, function = PlaceFieldNum_Reverse_Interface, 
                              file_name = code_id, behavior_paradigm = 'ReverseMaze')
    
if os.path.exists(join(figdata, code_id+' [Hairpin].pkl')):
    with open(join(figdata, code_id+' [Hairpin].pkl'), 'rb') as handle:
        HPData = pickle.load(handle)
else:
    HPData = DataFrameEstablish(variable_names = ['Field Number', 'Direction'], 
                              f = f4, function = PlaceFieldNum_Reverse_Interface, 
                              file_name = code_id + ' [Hairpin]', behavior_paradigm = 'HairpinMaze')
    
print_estimator(Data['Field Number'][np.where(Data['Direction'] == 'Cis')[0]])
print_estimator(Data['Field Number'][np.where(Data['Direction'] == 'Trs')[0]])
print_estimator(HPData['Field Number'][np.where(HPData['Direction'] == 'Cis')[0]])
print_estimator(HPData['Field Number'][np.where(HPData['Direction'] == 'Trs')[0]])