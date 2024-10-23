from mylib.statistic_test import *
from mylib.model import *

code_id = '0337 - Model Losses'
loc = join(figpath, code_id)
mkdir(loc)


from mylib.model import EqualRateDriftModel
from mylib.model import TwoProbabilityIndependentModel
from mylib.model import JointProbabilityModel
from mylib.model import HMM
from mylib.model import ContinuousHiddenStateModel
from mylib.model import ProbabilityRNN

from mylib.field.tracker_v2 import Tracker2d


def fit_models(field_reg, file_name: str):
    Models = {}

    res = {
        "Loss": [],
        "Model Type": []
    }

    tracker = Tracker2d(field_reg=field_reg)
    sequences = tracker.convert_to_sequence()
    lengths = np.array([len(seq) for seq in sequences])
    max_length = np.max(lengths)
    if max_length > 10:
        idx = np.where(lengths >= 10)[0]
        sequences = [sequences[i] for i in idx]
    else:
        idx = np.where(lengths >= 5)[0]
        sequences = [sequences[i] for i in idx]
    
    train_size = int(len(sequences) * 0.8)
    train_indices = np.random.choice(len(sequences), train_size, replace=False)
    test_indices = np.setdiff1d(np.arange(len(sequences)), train_indices)
    res['train_indices'] = train_indices
    res['train_size'] = 0.8
    res['sequences'] = sequences
    
    train_seq = [sequences[i] for i in train_indices]
    test_seq = [sequences[i] for i in test_indices]
    
    M1 = EqualRateDriftModel()
    M1.fit(train_seq)
    res['Loss'].append(M1.calc_loss(test_seq))
    res['Model Type'].append("Model I")

    M2 = TwoProbabilityIndependentModel()
    M2.fit(train_seq)
    res['Loss'].append(M2.calc_loss(test_seq))
    res['Model Type'].append("Model II")

    M3 = JointProbabilityModel()
    M3.fit(train_seq)
    res['Loss'].append(M3.calc_loss(test_seq))
    res['Model Type'].append("Model III")

    M41 = HMM.process_fit(N=5, sequences=train_seq, n_iterations=100)
    res['Loss'].append(M41.calc_loss(test_seq))
    res['Model Type'].append("Model IV - 5")
    
    M42 = HMM.process_fit(N=10, sequences=train_seq, n_iterations=100)
    res['Loss'].append(M42.calc_loss(test_seq))
    res['Model Type'].append("Model IV - 10")

    M43 = HMM.process_fit(N=20, sequences=train_seq, n_iterations=100)
    res['Loss'].append(M43.calc_loss(test_seq))
    res['Model Type'].append("Model IV - 20")

    M44 = HMM.process_fit(N=40, sequences=train_seq, n_iterations=100)
    res['Loss'].append(M44.calc_loss(test_seq))
    res['Model Type'].append("Model IV - 40")

    M51 = ContinuousHiddenStateModel('reci')
    M51.fit(train_seq)
    res['Loss'].append(M51.calc_loss(test_seq))
    res['Model Type'].append("Model V - reci")

    M52 = ContinuousHiddenStateModel('logistic')
    M52.fit(train_seq)
    res['Loss'].append(M52.calc_loss(test_seq))
    res['Model Type'].append("Model V - logistic")

    M53 = ContinuousHiddenStateModel('poly2')
    M53.fit(train_seq)
    res['Loss'].append(M53.calc_loss(test_seq))
    res['Model Type'].append("Model V - poly2")
    
    M54 = ContinuousHiddenStateModel('poly3')
    M54.fit(train_seq)
    res['Loss'].append(M54.calc_loss(test_seq))
    res['Model Type'].append("Model V - poly3")

    M61 = ProbabilityRNN.process_fit(
        sequences,
        train_index=train_indices,
        hidden_size=8,
        lr=0.001,
        epochs=1000, 
        batch_size=2048
    )
    res['Loss'].append(M61.calc_loss(test_seq))
    res['Model Type'].append("Model VI - 8") 

    M62 = ProbabilityRNN.process_fit(
        sequences,
        train_index=train_indices,
        hidden_size=16,
        lr=0.001,
        epochs=1000, 
        batch_size=2048
    )
    res['Loss'].append(M62.calc_loss(test_seq))
    res['Model Type'].append("Model VI - 16") 

    M63 = ProbabilityRNN.process_fit(
        sequences,
        train_index=train_indices,
        hidden_size=32,
        lr=0.001,
        epochs=1000, 
        batch_size=2048
    )
    res['Loss'].append(M63.calc_loss(test_seq))
    res['Model Type'].append("Model VI - 32")

    Models = [M1, M2, M3, M41, M42, M43, M44, M51, M52, M53, M54, M61, M62, M63]
    
    with open(join(loc, file_name), 'wb') as f:
        pickle.dump(Models, f)

    return res

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = {
        'Paradigm': [],
        'MiceID': [],
        'Model Type': [],
        'Loss': []
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
        
            res = fit_models(trace['field_reg'], file_name=f"{mouse}_{paradigm}.pkl")
        
            Data['MiceID'] += [mouse] * len(res['Loss'])
            Data['Paradigm'] += [paradigm] * len(res['Loss'])
            Data['Model Type'] += res['Model Type']
            Data['Loss'] += res['Loss']
            
        elif f_CellReg_modi['paradigm'][i] == 'ReverseMaze':
            mouse = int(f_CellReg_modi['MiceID'][i])
            maze_type = int(f_CellReg_modi['maze_type'][i])
            paradigm = 'MAf'
        
            res = fit_models(trace['cis']['field_reg'], file_name=f"{mouse}_{paradigm}.pkl")
        
            Data['MiceID'] += [mouse] * len(res['Loss'])
            Data['Paradigm'] += [paradigm] * len(res['Loss'])
            Data['Model Type'] += res['Model Type']
            Data['Loss'] += res['Loss']
            
            paradigm = 'MAb'
        
            res = fit_models(trace['trs']['field_reg'], file_name=f"{mouse}_{paradigm}.pkl")
        
            Data['MiceID'] += [mouse] * len(res['Loss'])
            Data['Paradigm'] += [paradigm] * len(res['Loss'])
            Data['Model Type'] += res['Model Type']
            Data['Loss'] += res['Loss']
        elif f_CellReg_modi['paradigm'][i] == 'HairpinMaze':
            mouse = int(f_CellReg_modi['MiceID'][i])
            maze_type = int(f_CellReg_modi['maze_type'][i])
            paradigm = 'HPf'
        
            res = fit_models(trace['cis']['field_reg'], file_name=f"{mouse}_{paradigm}.pkl")
        
            Data['MiceID'] += [mouse] * len(res['Loss'])
            Data['Paradigm'] += [paradigm] * len(res['Loss'])
            Data['Model Type'] += res['Model Type']
            Data['Loss'] += res['Loss']
            
            paradigm = 'HPb'
            
            res = fit_models(trace['trs']['field_reg'], file_name=f"{mouse}_{paradigm}.pkl")
        
            Data['MiceID'] += [mouse] * len(res['Loss'])
            Data['Paradigm'] += [paradigm] * len(res['Loss'])
            Data['Model Type'] += res['Model Type']
            Data['Loss'] += res['Loss']
        print("\n\n\n\n")
        
    for k in Data.keys():
        Data[k] = np.array(Data[k])
        
    with open(join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
        
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)
    
fig = plt.figure(figsize=(6, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Model Type',
    y = 'Loss',
    data = Data,
    hue="Model Type",
    width=0.8,
    palette=ModelPalette,
    err_kws={'linewidth': 0.5, 'color': 'black'},
    capsize=0.5,
    ax = ax
)
sns.stripplot(
    x = 'Model Type',
    y = 'Loss',
    data = Data,
    hue="Paradigm",
    palette=SpatialMapPalette,
    edgecolor='black',
    size=4,
    linewidth=0.20,
    jitter=0.2,
    alpha = 0.8,
    dodge=False,
    ax = ax
)
ax.set_ylim(0.45, 0.75),
ax.set_yticks(np.linspace(0.45, 0.75, 7))
plt.savefig(join(loc, 'Model Losses.png'), dpi = 600)
plt.savefig(join(loc, 'Model Losses.svg'), dpi = 600)
plt.close()

models = np.unique(Data['Model Type'])

model_loss = np.zeros(len(models))
for i in range(len(models)):
    idx = np.where(Data['Model Type'] == models[i])[0]
    print(f"{models[i]}:   {np.mean(Data['Loss'][idx])} ± {np.std(Data['Loss'][idx])}")
    model_loss[i] = np.nanmean(Data['Loss'][idx])

indices_order = np.argsort(model_loss)
print()
for i in range(len(indices_order)):
    print(f"Rank {i+1}: {models[indices_order[i]]}")