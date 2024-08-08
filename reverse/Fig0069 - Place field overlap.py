from mylib.statistic_test import *

code_id = '0069 - Place field overlap'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Cis Number', 'Trs Number', 'Overlap', 'Data Type'], 
                              f = f3, function = PlaceFieldOverlap_Reverse_Interface, 
                              file_name = code_id, behavior_paradigm = 'ReverseMaze')
    
if os.path.exists(join(figdata, code_id+' [Hairpin].pkl')):
    with open(join(figdata, code_id+' [Hairpin].pkl'), 'rb') as handle:
        HPData = pickle.load(handle)
else:
    HPData = DataFrameEstablish(variable_names = ['Cis Number', 'Trs Number', 'Overlap', 'Data Type'], 
                              f = f4, function = PlaceFieldOverlap_Reverse_Interface, 
                              file_name = code_id + ' [Hairpin]', behavior_paradigm = 'HairpinMaze')
    
print_estimator(Data['Overlap'][np.where(Data['Data Type'] == 'Data')[0]])
print_estimator(Data['Overlap'][np.where(Data['Data Type'] == 'Shuffle')[0]])
print(ttest_rel(Data['Overlap'][np.where(Data['Data Type'] == 'Data')[0]], Data['Overlap'][np.where(Data['Data Type'] == 'Shuffle')[0]]))

print_estimator(HPData['Overlap'][np.where(HPData['Data Type'] == 'Data')[0]])
print_estimator(HPData['Overlap'][np.where(HPData['Data Type'] == 'Shuffle')[0]])
print(ttest_rel(HPData['Overlap'][np.where(HPData['Data Type'] == 'Data')[0]], HPData['Overlap'][np.where(HPData['Data Type'] == 'Shuffle')[0]]))