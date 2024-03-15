from mylib.statistic_test import *
from mylib.stats.kstest import nbinom_kstest, lognorm_kstest, gamma_kstest
from scipy.stats import lognorm

code_id = "0067 - Statistic for Field Size"
loc = os.path.join(figpath, "reverse", code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Log-normal P-value', 'Log-normal Statistics', 'Direction'], 
                              f = f3, function = FieldSizeTestLogNormal_Reverse_Interface, 
                              file_name = code_id, behavior_paradigm = 'ReverseMaze')
    
if os.path.exists(join(figdata, code_id+' [Hairpin].pkl')):
    with open(join(figdata, code_id+' [Hairpin].pkl'), 'rb') as handle:
        HPData = pickle.load(handle)
else:
    HPData = DataFrameEstablish(variable_names = ['Log-normal P-value', 'Log-normal Statistics', 'Direction'], 
                              f = f4, function = FieldSizeTestLogNormal_Reverse_Interface, 
                              file_name = code_id + ' [Hairpin]', behavior_paradigm = 'HairpinMaze')
    
print(Data['Log-normal P-value'][np.where(Data['Direction'] == 'cis')[0]])
print(Data['Log-normal P-value'][np.where(Data['Direction'] == 'trs')[0]])
print(np.where(HPData['Log-normal P-value'][np.where(HPData['Direction'] == 'cis')[0]] > 0.05)[0].shape[0])
print(np.where(HPData['Log-normal P-value'][np.where(HPData['Direction'] == 'trs')[0]] > 0.05)[0].shape[0], end='\n\n')