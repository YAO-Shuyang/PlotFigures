from mylib.statistic_test import *

code_id = '0338 - Model Losses with Time Steps'
prevloc = join(figpath, '0337 - Model Losses')
loc = join(figpath, code_id)
mkdir(loc)

from mylib.model import EqualRateDriftModel
from mylib.model import TwoProbabilityIndependentModel
from mylib.model import JointProbabilityModel
from mylib.model import HMM
from mylib.model import ContinuousHiddenStateModel
from mylib.model import ProbabilityRNN

from mylib.field.tracker_v2 import Tracker2d

def get_stepwise_loss(sequences: list[np.ndarray], file_name: str):
    with open(join(prevloc, file_name), 'rb') as handle:
        Models = pickle.load(handle)
        
    model_names = ["Model I", "Model II", "Model III", "Model IV - 5", "Model IV - 10", 
                   "Model IV - 20", "Model IV - 40", "Model V - reci", "Model V - logistic",
                   "Model V - poly2", "Model VI - 8", "Model VI - 16", "Model VI - 32"]
    
    res = {
        "Loss": [],
        "Steps": [],
        "Model Type": []
    }
    max_length = max([len(seq) for seq in sequences])
    padd_sequences = np.zeros((len(sequences), max_length-1)) * np.nan
    for i, seq in enumerate(sequences):
        padd_sequences[i, :len(seq)-1] = seq[1:]
    
    for i, model in enumerate(Models):
        model.get_predicted_prob(sequences)
        predicted_prob = model.predicted_prob
        padded_prob = np.full((len(sequences), max_length-1), np.nan)
        for j in range(len(sequences)):
            padded_prob[j, :len(predicted_prob[j])] = predicted_prob[j]
            
        Steps = np.arange(1, max_length)
        #NLL loss
        loss = - np.nanmean(
            padd_sequences * np.log(padded_prob) + (1 - padd_sequences) * np.log(1 - padded_prob), axis=0
        )
        res['Loss'].append(loss)
        res['Steps'].append(Steps)
        res['Model Type'].append([model_names[i]] * len(Steps))
        
    for k in res.keys():
        res[k] = np.concatenate(res[k])
        
    return res
    
    

if exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = {
        'Paradigm': [],
        'MiceID': [],
        'Model Type': [],
        'Loss': [],
        'Time Step': [],
    }
    
    for i in range(len(f_CellReg_modi)):
        if f_CellReg_modi['Type'][i] != 'Real' or f_CellReg_modi['maze_type'][i] == 0:
            continue
    
        print(f_CellReg_modi['Trace File'][i])
    
        with open(f_CellReg_modi['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)

        if f_CellReg_modi['paradigm'][i] == 'CrossMaze':
            mouse = int(f_CellReg_modi['MiceID'][i])
            maze_type = int(f_CellReg_modi['maze_type'][i])
            
            paradigm = 'MA' if maze_type == 1 else 'MB'
            
            with open(f_CellReg_modi['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
                
            tracker = Tracker2d(field_reg=trace['field_reg'])
            sequences = tracker.convert_to_sequence()
            
            res = get_stepwise_loss(sequences=sequences, file_name=f"{mouse}_{paradigm}.pkl")
            
            Data['MiceID'] += [mouse] * len(res['Loss'])
            Data['Paradigm'] += [paradigm] * len(res['Loss'])
            Data['Model Type'].append(res['Model Type'])
            Data['Loss'].append(res['Loss'])
            Data['Time Step'].append(res['Steps'])
            
        elif f_CellReg_modi['paradigm'][i] == 'ReverseMaze':
            paradigm = 'MAf'
            
            with open(f_CellReg_modi['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
                
            mouse = int(f_CellReg_modi['MiceID'][i])
            
            tracker = Tracker2d(field_reg=trace['cis']['field_reg'])
            sequences = tracker.convert_to_sequence()
            res = get_stepwise_loss(sequences=sequences, file_name=f"{mouse}_{paradigm}.pkl")
            
            Data['MiceID'] += [mouse] * len(res['Loss'])
            Data['Paradigm'] += [paradigm] * len(res['Loss'])
            Data['Model Type'].append(res['Model Type'])
            Data['Loss'].append(res['Loss'])
            Data['Time Step'].append(res['Steps'])
            
            paradigm = 'MAb'
            tracker = Tracker2d(field_reg=trace['trs']['field_reg'])
            sequences = tracker.convert_to_sequence()
            res = get_stepwise_loss(sequences=sequences, file_name=f"{mouse}_{paradigm}.pkl")
            
            Data['MiceID'] += [mouse] * len(res['Loss'])
            Data['Paradigm'] += [paradigm] * len(res['Loss'])
            Data['Model Type'].append(res['Model Type'])
            Data['Loss'].append(res['Loss'])
            Data['Time Step'].append(res['Steps'])
        elif f_CellReg_modi['paradigm'][i] == 'HairpinMaze':
            paradigm = 'HPf'
            
            with open(f_CellReg_modi['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
            mouse = int(f_CellReg_modi['MiceID'][i])
            
            tracker = Tracker2d(field_reg=trace['cis']['field_reg'])
            sequences = tracker.convert_to_sequence()
            res = get_stepwise_loss(sequences=sequences, file_name=f"{mouse}_{paradigm}.pkl")
            
            Data['MiceID'] += [mouse] * len(res['Loss'])
            Data['Paradigm'] += [paradigm] * len(res['Loss'])
            Data['Model Type'].append(res['Model Type'])
            Data['Loss'].append(res['Loss'])
            Data['Time Step'].append(res['Steps'])
            
            paradigm = 'HPb'
            tracker = Tracker2d(field_reg=trace['trs']['field_reg'])
            sequences = tracker.convert_to_sequence()
            res = get_stepwise_loss(sequences=sequences, file_name=f"{mouse}_{paradigm}.pkl")
            
            Data['MiceID'] += [mouse] * len(res['Loss'])
            Data['Paradigm'] += [paradigm] * len(res['Loss'])
            Data['Model Type'].append(res['Model Type'])
            Data['Loss'].append(res['Loss'])
            Data['Time Step'].append(res['Steps'])
            
    for k in ['MiceID', 'Paradigm']:
        Data[k] = np.array(Data[k])
    
    for k in ['Model Type', 'Loss', 'Time Step']:
        Data[k] = np.concatenate(Data[k])
    
    with open(join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
    
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)
    

fig = plt.figure(figsize=(8, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Time Step',
    y = 'Loss',
    data = Data,
    hue="Model Type",
    palette=ModelPalette,
    err_style="bars",
    err_kws={'linewidth': 0.5, 'capsize': 3, 'capthick': 0.5},
    alpha = 0.8,
    linewidth = 0.5,
    ax = ax
)
ax.set_ylim(0.35, 0.75)
ax.set_yticks(np.linspace(0.35, 0.75, 9))
plt.savefig(join(loc, 'Model Losses with Time Steps.png'), dpi = 600)
plt.savefig(join(loc, 'Model Losses with Time Steps.svg'), dpi = 600)
plt.close()