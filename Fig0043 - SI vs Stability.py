from mylib.statistic_test import *

code_id = '0043 - SI vs In-session stability'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Half-half Correlation', 'Odd-even Correlation', 'Cell Type'], f = f1, 
                              function = InterSessionCorrelation_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

def clear_nan_value(data):
    idx = np.where(np.isnan(data))[0]
    return np.delete(data, idx)