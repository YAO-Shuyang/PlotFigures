from mylib.statistic_test import *
from mylib.dsp.starting_cell import *

code_id = '0830 - Starting Cell Encoded Route Density Distribution'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

for i in range(len(f2)):
    with open(f2['Trace File'][i], 'rb') as handle:
        trace = pickle.load(handle)
    
    print(i, f2['MiceID'][i], f2['date'][i])
    print(np.where((trace['SC_OI'] == 1) & (trace['SC_OI'] > 0.8))[0], end='\n\n')