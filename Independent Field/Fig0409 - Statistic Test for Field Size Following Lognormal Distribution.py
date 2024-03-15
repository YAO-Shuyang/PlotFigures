from mylib.statistic_test import *
from mylib.stats.kstest import nbinom_kstest, lognorm_kstest, gamma_kstest
from scipy.stats import lognorm

code_id = "0409 - Statistic Test for Field Size Following Lognormal Distribution"
loc = os.path.join(figpath, "Independent Field", code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = {"MiceID": [], "date": [], "Stage": [], "Training Day": [], "Maze Type": [],
            "Log-normal P-value": [], "Gamma P-value": [], "Neg-Binom P-value": []}
    
    mazes = ['Open Field', 'Maze 1', 'Maze 2']
    
    for i in tqdm(range(len(f1))):
        if f1['include'][i] == 0:
            continue
        
        with open(os.path.join(figdata, f1['Trace File'][i]), 'rb') as handle:
            trace = pickle.load(handle)
            
        size = []
        for j in range(len(trace['place_field_all'])):
            if trace['is_placecell'][j] == 1:
                for k in trace['place_field_all'][j].keys():
                    size.append(len(trace['place_field_all'][j][k]))
        
        if len(size) < 200:
            continue
        
        size = np.array(size)
        _, lognorm_p = lognorm_kstest(size, resample_size=1629)
        print_estimator(size)
        print("Lognormal P-value: ", lognorm_p, end="\n\n")
        
        Data['Log-normal P-value'].append(lognorm_p)
        Data['Maze Type'].append(mazes[int(f1['maze_type'][i])])
        Data['MiceID'].append(int(f1['MiceID'][i]))
        Data['date'].append(int(f1['date'][i]))
        Data['Stage'].append(f1['Stage'][i])
        Data['Training Day'].append(f1['training_day'][i])

    for k in Data.keys():
        Data[k] = np.array(Data[k])
        
    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)