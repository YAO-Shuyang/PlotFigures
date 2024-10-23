from mylib.statistic_test import *

code_id = '0341 - Whether GatedRNN Follow SFER General Sense'
loc = os.path.join(figpath, code_id)
mkdir(loc)
from mylib.model import EqualRateDriftModel
from mylib.model import TwoProbabilityIndependentModel
from mylib.model import JointProbabilityModel
from mylib.model import HMM
from mylib.model import ContinuousHiddenStateModel
from mylib.model import ProbabilityRNN
from mylib.field.tracker_v2 import Tracker2d

def counts(model: ProbabilityRNN, sequences: list[np.ndarray]):
    seq = np.concatenate([i[1:] for i in sequences])
    n_one = np.sum(seq == 1)
    n_zero = np.sum(seq == 0)
    
    n_one_violate, n_zero_violate = 0, 0
    predicted_prob = model.get_predicted_prob(sequences)
    
    for i in range(len(predicted_prob)):
        prob = np.ediff1d(predicted_prob[i])
        is_correct = (sequences[i][1:-1]-0.5) * prob > 0
        
        n_one_violate += np.sum((is_correct == False) & (sequences[i][1:-1] == 1))
        n_zero_violate += np.sum((is_correct == False) & (sequences[i][1:-1] == 0))
            
    return n_one, n_zero, n_one_violate, n_zero_violate
    

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)  
else:    
    Data = {"MiceID": [], "Num One": [], "Num Zero": [], "Num Violate One": [], "Num Violate Zero": [], "Paradigm": [], "Model": []}
    
    for i in range(len(f_CellReg_modi)):
        if f_CellReg_modi['Type'][i] != "Real" or f_CellReg_modi['include'][i] == 0:
            continue

        print(f_CellReg_modi['Trace File'][i])
    
        with open(f_CellReg_modi['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)

        if f_CellReg_modi['paradigm'][i] == 'CrossMaze':
            if f_CellReg_modi['maze_type'][i] == 0:
                continue
            
            maze_type = int(f_CellReg_modi['maze_type'][i])
            paradigm = 'MA' if maze_type == 1 else 'MB'
            mouse = int(f_CellReg_modi['MiceID'][i])
            
            tracker = Tracker2d(field_reg=trace['field_reg'])
            sequences = tracker.convert_to_sequence()   
                     
            file_dir = join(
                figpath,
                f"0337 - Model Losses",
                f"{mouse}_{paradigm}.pkl"
            )
            
            with open(file_dir, 'rb') as handle:
                res = pickle.load(handle)
            
            M8, M16, M32 = res[-3], res[-2], res[-1]
            M5, M10, M20, M40 = res[-11], res[-10], res[-9], res[-8]
            
            n_one, n_zero, n_one_violate, n_zero_violate = counts(M8, sequences)
            Data['MiceID'] += [mouse] * 7
            Data['Paradigm'] += [paradigm] * 7
            Data['Num One'].append(n_one)
            Data['Num Zero'].append(n_zero)
            Data['Num Violate One'].append(n_one_violate)
            Data['Num Violate Zero'].append(n_zero_violate)
            
            n_one, n_zero, n_one_violate, n_zero_violate = counts(M16, sequences)
            Data['Num One'].append(n_one)
            Data['Num Zero'].append(n_zero)
            Data['Num Violate One'].append(n_one_violate)
            Data['Num Violate Zero'].append(n_zero_violate)
            
            n_one, n_zero, n_one_violate, n_zero_violate = counts(M32, sequences)
            Data['Num One'].append(n_one)
            Data['Num Zero'].append(n_zero)
            Data['Num Violate One'].append(n_one_violate)
            Data['Num Violate Zero'].append(n_zero_violate)
            
            n_one, n_zero, n_one_violate, n_zero_violate = counts(M5, sequences)
            Data['Num One'].append(n_one)
            Data['Num Zero'].append(n_zero)
            Data['Num Violate One'].append(n_one_violate)
            Data['Num Violate Zero'].append(n_zero_violate)
            
            n_one, n_zero, n_one_violate, n_zero_violate = counts(M10, sequences)
            Data['Num One'].append(n_one)
            Data['Num Zero'].append(n_zero)
            Data['Num Violate One'].append(n_one_violate)
            Data['Num Violate Zero'].append(n_zero_violate)
            
            n_one, n_zero, n_one_violate, n_zero_violate = counts(M20, sequences)
            Data['Num One'].append(n_one)
            Data['Num Zero'].append(n_zero)
            Data['Num Violate One'].append(n_one_violate)
            Data['Num Violate Zero'].append(n_zero_violate)
            
            n_one, n_zero, n_one_violate, n_zero_violate = counts(M40, sequences)
            Data['Num One'].append(n_one)
            Data['Num Zero'].append(n_zero)
            Data['Num Violate One'].append(n_one_violate)
            Data['Num Violate Zero'].append(n_zero_violate)
            
            Data['Model'] += ['M8', 'M16', 'M32', 'M5', 'M10', 'M20', 'M40']
        else:
            mouse = int(f_CellReg_modi['MiceID'][i])
            if f_CellReg_modi['paradigm'][i] == 'ReverseMaze':
                params = ["MAf", "MAb"]
            elif f_CellReg_modi['paradigm'][i] == 'HairpinMaze':
                params = ["HPf", "HPb"]
                
            for paradigm in params:
                file_dir = join(
                    figpath,
                    f"0337 - Model Losses",
                    f"{mouse}_{paradigm}.pkl"
                )
                
                with open(file_dir, 'rb') as handle:
                    res = pickle.load(handle)     
                               
                M8, M16, M32 = res[-3], res[-2], res[-1]
                M5, M10, M20, M40 = res[-11], res[-10], res[-9], res[-8]
                
                n_one, n_zero, n_one_violate, n_zero_violate = counts(M8, sequences)
                Data['MiceID'] += [mouse] * 7
                Data['Paradigm'] += [paradigm] * 7
                Data['Num One'].append(n_one)
                Data['Num Zero'].append(n_zero)
                Data['Num Violate One'].append(n_one_violate)
                Data['Num Violate Zero'].append(n_zero_violate)
                
                n_one, n_zero, n_one_violate, n_zero_violate = counts(M16, sequences)
                Data['Num One'].append(n_one)
                Data['Num Zero'].append(n_zero)
                Data['Num Violate One'].append(n_one_violate)
                Data['Num Violate Zero'].append(n_zero_violate)
                
                n_one, n_zero, n_one_violate, n_zero_violate = counts(M32, sequences)
                Data['Num One'].append(n_one)
                Data['Num Zero'].append(n_zero)
                Data['Num Violate One'].append(n_one_violate)
                Data['Num Violate Zero'].append(n_zero_violate)
                
                n_one, n_zero, n_one_violate, n_zero_violate = counts(M5, sequences)
                Data['Num One'].append(n_one)
                Data['Num Zero'].append(n_zero)
                Data['Num Violate One'].append(n_one_violate)
                Data['Num Violate Zero'].append(n_zero_violate)
                
                n_one, n_zero, n_one_violate, n_zero_violate = counts(M10, sequences)
                Data['Num One'].append(n_one)
                Data['Num Zero'].append(n_zero)
                Data['Num Violate One'].append(n_one_violate)
                Data['Num Violate Zero'].append(n_zero_violate)
                
                n_one, n_zero, n_one_violate, n_zero_violate = counts(M20, sequences)
                Data['Num One'].append(n_one)
                Data['Num Zero'].append(n_zero)
                Data['Num Violate One'].append(n_one_violate)
                Data['Num Violate Zero'].append(n_zero_violate)
                
                n_one, n_zero, n_one_violate, n_zero_violate = counts(M40, sequences)
                Data['Num One'].append(n_one)
                Data['Num Zero'].append(n_zero)
                Data['Num Violate One'].append(n_one_violate)
                Data['Num Violate Zero'].append(n_zero_violate)
                
                Data['Model'] += ['M8', 'M16', 'M32', 'M5', 'M10', 'M20', 'M40']
    
    for k in Data.keys():
        Data[k] = np.array(Data[k])
            
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)
    
    with open(join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)

for i in ["M8", "M16", "M32", "M5", "M10", "M20", "M40"]:
    idx = np.where(Data['Model'] == i)[0]
    print(f"{i}: {(np.sum(Data['Num Violate Zero'][idx]) + np.sum(Data['Num Violate One'][idx])) / (np.sum(Data['Num Zero'][idx]) + np.sum(Data['Num One'][idx]))}, {(np.sum(Data['Num Zero'][idx]) + np.sum(Data['Num One'][idx]))}")

idx = np.where(Data['Model'] != 'M5')[0]
Data = SubDict(Data, Data.keys(), idx=idx)
P = 1-(Data['Num Violate Zero'] + Data['Num Violate One']) / (Data['Num Zero'] + Data['Num One'])

fig = plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = Data['Model'],
    y = P,
    ax = ax,
    hue=Data['Model'],
    err_kws={'linewidth': 0.5, 'color': 'black'},
    capsize=0.5,
    palette=sns.color_palette("Purples", 3) + sns.color_palette("Greens", 4)[1:],
    width=0.8
)
sns.stripplot(
    x = Data['Model'],
    y = P,
    edgecolor='black',
    size=4,
    linewidth=0.20,
    jitter=0.2,
    alpha = 0.8,
    dodge=False,
    ax = ax
)
ax.set_ylim(0, 1.03)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, 'M8-32.png'), dpi = 600)
plt.savefig(join(loc, 'M8-32.svg'), dpi = 600)
plt.show()