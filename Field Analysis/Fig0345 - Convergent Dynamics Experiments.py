from mylib.statistic_test import *

code_id = '0345 - Convergent Dynamics Experiments'
loc = os.path.join(figpath, code_id)
mkdir(loc)

from mylib.model import ContinuousHiddenStateModel
from mylib.model import ProbabilityRNN

from mylib.field.tracker_v2 import Tracker2d

file_loc = join(figpath, "0337 - Model Losses")

if exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = {
        "Step": [],
        "P": [],
        "MiceID": [],
        "Paradigm": [],
    }
    
    for i in tqdm(range(len(f_CellReg_modi))):
        if f_CellReg_modi['Type'][i] != 'Real' or f_CellReg_modi['maze_type'][i] == 0:
            continue
        
        with open(f_CellReg_modi['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)  
              
        if f_CellReg_modi['paradigm'][i] == 'CrossMaze':
            paradigm = 'MA' if f_CellReg_modi['maze_type'][i] == 1 else 'MB'
        
            with open(join(file_loc, f'{int(f_CellReg_modi["MiceID"][i])}_{paradigm}.pkl'), 'rb') as handle:
                res = pickle.load(handle)
            
            M53: ContinuousHiddenStateModel = res[-5]
            tracker = Tracker2d(field_reg=trace['field_reg'])
            sequences = tracker.convert_to_sequence()
            predicted_prob = M53.get_predicted_prob(sequences)
            
            for j in range(len(predicted_prob)):
                Data['Step'].append(np.arange(len(predicted_prob[j])))
                Data['P'].append(predicted_prob[j])
                Data['MiceID'].append(np.repeat(int(f_CellReg_modi['MiceID'][i]), len(predicted_prob[j])))
                Data['Paradigm'].append(np.repeat(paradigm, len(predicted_prob[j])))
            
        else:
            info = {"ReverseMaze": ["MAf", "MAb"], "HairpinMaze": ["HPf", "HPb"]}
            directions = ['cis', 'trs']
            for n, paradigm in enumerate(info[f_CellReg_modi['paradigm'][i]]):
                with open(join(file_loc, f'{int(f_CellReg_modi["MiceID"][i])}_{paradigm}.pkl'), 'rb') as handle:
                    res = pickle.load(handle)
                
                M53: ContinuousHiddenStateModel = res[-5]
                tracker = Tracker2d(field_reg=trace[directions[n]]['field_reg'])
                sequences = tracker.convert_to_sequence()
                predicted_prob = M53.get_predicted_prob(sequences)
                
                for j in range(len(predicted_prob)):
                    Data['Step'].append(np.arange(len(predicted_prob[j])))
                    Data['P'].append(predicted_prob[j])
                    Data['MiceID'].append(np.repeat(int(f_CellReg_modi['MiceID'][i]), len(predicted_prob[j])))
                    Data['Paradigm'].append(np.repeat(paradigm, len(predicted_prob[j])))
                    
    for k in Data.keys():
        Data[k] = np.concatenate(Data[k])
        
    with open(join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
    
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)

mat = np.zeros((np.max(Data['Step'])+1, 40)) * np.nan
for i in range(len(Data['Step'])):
    idx = np.where(Data['Step'] == i)[0]
    if len(idx) == 0:
        continue
    else:
        mat[i, :] = np.histogram(
            Data['P'][idx],
            bins=40,
            range=(0, 1),
            density=True
        )[0]
        
mat = mat / np.nanmax(mat, axis=1)

fig = plt.figure(figsize=(4, 8))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.imshow(mat, aspect='auto')

plt.show()