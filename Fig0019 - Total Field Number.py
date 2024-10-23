from mylib.statistic_test import *

code_id = '0019 - Total Field Number'
loc = os.path.join(figpath, code_id)
mkdir(loc)

idx = np.where(np.isin(f1['MiceID'], [10209, 10212, 10224, 10227]))[0]
if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Field Num'], f = f1, file_idx=idx, 
                              function = TotalFieldNumber_Interface, file_name = code_id, 
                              behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata, code_id+' [Reverse].pkl')) == False:
    RData = DataFrameEstablish(variable_names = ['Field Num', "Direction"], f = f3,
                              function = TotalFieldNumber_Reverse_Interface, file_name = code_id + ' [Reverse]', 
                              behavior_paradigm = 'ReverseMaze')
else:
    with open(os.path.join(figdata, code_id+' [Reverse].pkl'), 'rb') as handle:
        RData = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata, code_id+' [Hairpin].pkl')) == False:
    HData = DataFrameEstablish(variable_names = ['Field Num', "Direction"], f = f4,  
                              function = TotalFieldNumber_Reverse_Interface, file_name = code_id + ' [Hairpin]', 
                              behavior_paradigm = 'HairpinMaze')
else:
    with open(os.path.join(figdata, code_id+' [Hairpin].pkl'), 'rb') as handle:
        HData = pickle.load(handle)
        
print(f"MA: {Data['Field Num'][Data['Maze Type'] == 'Maze 1'].mean()}, SD: {Data['Field Num'][Data['Maze Type'] == 'Maze 1'].std()}")
print(f"MB: {Data['Field Num'][Data['Maze Type'] == 'Maze 2'].mean()}, SD: {Data['Field Num'][Data['Maze Type'] == 'Maze 2'].std()}")
print(f"MAf: {RData['Field Num'][RData['Direction'] == 'cis'].mean()}, SD: {RData['Field Num'][RData['Direction'] == 'cis'].std()}")
print(f"MAb: {RData['Field Num'][RData['Direction'] == 'trs'].mean()}, SD: {RData['Field Num'][RData['Direction'] == 'trs'].std()}")
print(f"HPf: {HData['Field Num'][HData['Direction'] == 'cis'].mean()}, SD: {HData['Field Num'][HData['Direction'] == 'cis'].std()}")
print(f"HPb: {HData['Field Num'][HData['Direction'] == 'trs'].mean()}, SD: {HData['Field Num'][HData['Direction'] == 'trs'].std()}")