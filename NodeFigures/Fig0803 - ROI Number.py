from mylib.statistic_test import *


code_id = '0803 - ROI Number'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, f"{code_id}.pkl")) == False:
    Data = DataFrameEstablish(variable_names = ['ROI Number'],
                              f=f2, file_idx=np.where(np.isin(f2['MiceID'], [10212,10224, 10227, 10232, 10275, 10276]))[0], 
                              function = ROINumber_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, f"{code_id}.pkl"), 'rb') as handle:
        Data = pickle.load(handle)
        
print_estimator(Data['ROI Number'])