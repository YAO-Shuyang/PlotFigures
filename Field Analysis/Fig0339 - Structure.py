from mylib.statistic_test import *
from mylib.field.sfer import Feature
from mylib.field.tracker_v2 import Tracker2d

code_id = '0339 - Structure'
prevloc = join(figpath, '0337 - Model Losses')
loc = join(figpath, code_id)
mkdir(loc)

def compute_kd(field_reg, file_name):
    with open(join(prevloc, file_name), 'rb') as handle:
        Models = pickle.load(handle)
    
    tracker = Tracker2d(field_reg=field_reg)
    sequences = tracker.convert_to_sequence()
    
    model_names = ["Model I", "Model II", "Model III", "Model IV - 5", "Model IV - 10", 
                   "Model IV - 20", "Model IV - 40", "Model V - reci", "Model V - logistic",
                   "Model V - poly2", "Model VI - 8", "Model VI - 16", "Model VI - 32"]
    
    res = {
        "D": [],
        "Model Type": [],
        "Feature": []
    }
    
    for i, model in enumerate(Models):
        model.get_predicted_prob(sequences)
        simu_seq = model.simulate(sequences)
        print(len(simu_seq), len(sequences))
        if len(simu_seq) != len(sequences):
            print(model_names[i])
        
        # A_distribution
        A = Feature.A_distribution(sequences)
        AS = Feature.A_distribution(simu_seq)
        d = ks_2samp(A, AS)[0]
        res['D'].append(d)
        res['Model Type'].append(model_names[i])
        res['Feature'].append("A_distribution")
        
        # I_distribution
        I = Feature.I_distribution(sequences)
        IS = Feature.I_distribution(simu_seq)
        d = ks_2samp(I, IS)[0]
        res['D'].append(d)
        res['Model Type'].append(model_names[i])
        res['Feature'].append("I_distribution")
        
        # switch_frequency
        d = ks_2samp(Feature.switch_frequency(sequences), 
                     Feature.switch_frequency(simu_seq))[0]
        res['D'].append(d)
        res['Model Type'].append(model_names[i])
        res['Feature'].append("switch moment")
        
        alla = Feature.all_A(sequences)
        allas = Feature.all_A(simu_seq)
        d = ks_2samp(alla, allas)[0]
        res['D'].append(d)
        res['Model Type'].append(model_names[i])
        res['Feature'].append("all A")
        
    return res

if exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
    
    for k in Data.keys():
        Data[k] = np.array(Data[k])
else:
    Data = {
        "MiceID": [],
        "D": [],
        "Model Type": [],
        "Paradigm": [],
        "Feature": []
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
        
            res = compute_kd(trace['field_reg'], file_name=f"{mouse}_{paradigm}.pkl")
        
            Data['MiceID'] += [mouse] * len(res['D'])
            Data['Paradigm'] += [paradigm] * len(res['D'])
            Data['Model Type'] += res['Model Type']
            Data['Feature'] += res['Feature']
            Data['D'] += res['D']
        
        elif f_CellReg_modi['paradigm'][i] == 'ReverseMaze':
            mouse = int(f_CellReg_modi['MiceID'][i])
            maze_type = int(f_CellReg_modi['maze_type'][i])
            paradigm = 'MAf'
        
            res = compute_kd(trace['cis']['field_reg'], file_name=f"{mouse}_{paradigm}.pkl")
        
            Data['MiceID'] += [mouse] * len(res['D'])
            Data['Paradigm'] += [paradigm] * len(res['D'])
            Data['Model Type'] += res['Model Type']
            Data['Feature'] += res['Feature']
            Data['D'] += res['D']
            
            paradigm = 'MAb'
        
            res = compute_kd(trace['trs']['field_reg'], file_name=f"{mouse}_{paradigm}.pkl")
        
            Data['MiceID'] += [mouse] * len(res['D'])
            Data['Paradigm'] += [paradigm] * len(res['D'])  
            Data['Model Type'] += res['Model Type']
            Data['Feature'] += res['Feature']
            Data['D'] += res['D']
        
        elif f_CellReg_modi['paradigm'][i] == 'HairpinMaze':
            mouse = int(f_CellReg_modi['MiceID'][i])
            maze_type = int(f_CellReg_modi['maze_type'][i])
            paradigm = 'HPf'
        
            res = compute_kd(trace['cis']['field_reg'], file_name=f"{mouse}_{paradigm}.pkl")
        
            Data['MiceID'] += [mouse] * len(res['D'])
            Data['Paradigm'] += [paradigm] * len(res['D'])
            Data['Model Type'] += res['Model Type']
            Data['Feature'] += res['Feature']
            Data['D'] += res['D']
            
            paradigm = 'HPb'
            
            res = compute_kd(trace['trs']['field_reg'], file_name=f"{mouse}_{paradigm}.pkl")
        
            Data['MiceID'] += [mouse] * len(res['D'])
            Data['Paradigm'] += [paradigm] * len(res['D'])
            Data['Model Type'] += res['Model Type']
            Data['Feature'] += res['Feature']
            Data['D'] += res['D']

    with open(join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)

with open(r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\test\demo_seq.pkl", 'rb') as handle:
    sequences = pickle.load(handle)

with open(join(prevloc, "10227_MA.pkl"), 'rb') as handle:
    Models = pickle.load(handle)

model_names = ["Real", "Model I", "Model II", "Model III", "Model IV - 5", "Model IV - 10",
               "Model IV - 20", "Model IV - 40", "Model V - reci", "Model V - logistic",
               "Model V - poly2", "Model VI - 8", "Model VI - 16", "Model VI - 32"]

simu_seqs = []
for i, model in enumerate(Models):
    model.get_predicted_prob(sequences)
    simu_seq = model.simulate(sequences)
    simu_seqs.append(simu_seq)
    
fig, axes = plt.subplots(2, 2, figsize=(6, 6))
ax1 = Clear_Axes(axes[0, 0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
A = np.concatenate(
    [Feature.A_distribution(sequences)] + 
    [Feature.A_distribution(simu_seq) for simu_seq in simu_seqs]
)



X = np.where(Data['Feature'] == 'A_distribution')[0]
Y = np.where(Data['Feature'] == 'I_distribution')[0]
fig = plt.figure(figsize=(3, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.scatterplot(
    x=Data['D'][X],
    y=Data['D'][Y],
    hue = Data['Model Type'][X],
    palette = ModelPalette,
    size=Data['D'][Y],
    edgecolor = None,
    sizes=(8,10),
    alpha = 0.8
)
ax.set_xlim(0, 0.3)
ax.set_ylim(0, 0.3)
ax.set_xticks(np.linspace(0, 0.3, 7))
ax.set_yticks(np.linspace(0, 0.3, 7))
plt.savefig(join(loc, 'A_I_Distribution.png'), dpi=600)
plt.savefig(join(loc, 'A_I_Distribution.svg'), dpi=600)
plt.close()

X = np.where(Data['Feature'] == 'switch moment')[0]
Y = np.where(Data['Feature'] == 'all A')[0]
fig = plt.figure(figsize=(3, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.scatterplot(
    x=Data['D'][X],
    y=Data['D'][Y],
    hue = Data['Model Type'][X],
    palette = ModelPalette,
    size=Data['D'][Y],
    edgecolor = None,
    sizes=(8,10),
    alpha = 0.8
)
ax.set_xlim(0, 0.3)
ax.set_ylim(0, 0.3)
ax.set_xticks(np.linspace(0, 0.3, 7))
ax.set_yticks(np.linspace(0, 0.3, 7))
plt.savefig(join(loc, 'switch.png'), dpi=600)
plt.savefig(join(loc, 'switch.svg'), dpi=600)
plt.close()