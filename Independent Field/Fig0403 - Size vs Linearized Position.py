from mylib.statistic_test import *

code_id = "0403 - Field Size vs Linearized Position"
loc = join(figpath, "Independent Field", code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, 'Field Pool.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['FSC Stability', 'OEC Stability', 'Field Size', 'Field Length', 'Peak Rate', 'Position'], f = f1,
                              function = WithinFieldBasicInfo_Interface, func_kwgs = {'is_placecell': True},
                              file_name = 'Field Pool', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, 'Field Pool.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata, 'Field Statistics in Cell Pool.pkl')) == False:
    CellData = DataFrameEstablish(variable_names = ['Mean FSC', 'Std. FSC', 'Median FSC', 'Error FSC',
                                                'Mean OEC', 'Std. OEC', 'Median OEC', 'Error OEC',
                                                'Mean Size', 'Std. Size', 'Median Size', 'Error Size',
                                                'Mean Length', 'Std. Length', 'Median Length', 'Error Length',
                                                'Mean Rate', 'Std. Rate', 'Median Rate', 'Error Rate',
                                                'Mean Position', 'Std. Position', 'Median Position', 'Error Position',
                                                'Mean Interdistance', 'Std. Interdistance', 'Median Interdistance', 'Error Interdistance',
                                                'Cell ID', 'Field Number'], f = f1,
                              function = WithinCellFieldStatistics_Interface, func_kwgs = {'is_placecell': True},
                              file_name = 'Field Statistics in Cell Pool', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, 'Field Statistics in Cell Pool.pkl'), 'rb') as handle:
        CellData = pickle.load(handle)


if os.path.exists(os.path.join(figdata, '0028 - Place Field Number Distribution Statistics [Monte Carlo].pkl')) == False:
    PData = DataFrameEstablish(variable_names = ['lam', 'Poisson KS Statistics', 'Poisson KS P-Value', 
                                                      'r', 'p', 'nbinom KS Statistics', 'nbinom KS P-Value',
                                                      'mean', 'sigma', 'Normal KS Statistics', 'Normal KS P-Value'], f = f1,
                              function = FieldDistributionStatistics_TestAll_Interface, 
                              file_name = '0028 - Place Field Number Distribution Statistics [Monte Carlo]', behavior_paradigm = 'HairpinMaze')
else:
    with open(os.path.join(figdata,'0028 - Place Field Number Distribution Statistics [Monte Carlo].pkl'), 'rb') as handle:
        PData = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    IndeptData = DataFrameEstablish(variable_names = ['Statistic', 'P-Value', 
                                             'Pearson r', 'Pearson P-Value', 
                                             'Spearman r', 'Spearman P-Value'], f = f1,
                              function = IndeptTestForPositionAndFieldLength_Interface,
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        IndeptData = pickle.load(handle)

if os.path.exists(join(figdata, code_id+' [spearman].pkl')) == False:
    Corr = {"MiceID": [], "Maze Type": [], "date": [], "Stage": [], 'Training Day': [],
            "pearson r": [], "pearson p": [], "spearman r": [], "spearman p": []}
    
    for i in tqdm(range(PData['Poisson KS P-Value'].shape[0])):
        if PData['Maze Type'][i] == 'Open Field':
            continue
    
        maze_type = 1 if PData['Maze Type'][i] == 'Maze 1' else 2
        Corr['Maze Type'].append(PData['Maze Type'][i])
        Corr['MiceID'].append(PData['MiceID'][i])
        Corr['date'].append(PData['date'][i])
        Corr['Stage'].append(PData['Stage'][i])
    
        idx = np.where((f1['maze_type'] == maze_type)&(f1['MiceID'] == PData['MiceID'][i])&(f1['date'] == PData['date'][i]))[0]
        if len(idx) != 1:
            print(idx, "error length")
            continue
    
        k = idx[0]
        Corr['Training Day'].append(f1['training_day'][k])
        idx = np.where((Data['MiceID'] == f1['MiceID'][k])&(Data['date'] == f1['date'][k])&(Data['Stage'] == f1['Stage'][k])&(Data['Maze Type'] == PData['Maze Type'][i]))[0]
        SubData = SubDict(Data, Data.keys(), idx=idx)
        x, y = cp.deepcopy(SubData['Position']), cp.deepcopy(SubData['Field Length'])
        corr, p = spearmanr(x, y)
        
        Corr['spearman r'].append(corr)
        Corr['spearman p'].append(p)
        
        corr, p = pearsonr(x, y)
        Corr['pearson r'].append(corr)
        Corr['pearson p'].append(p)
    
    for k in Corr.keys():
        Corr[k] = np.array(Corr[k])
        
    with open(join(figdata, code_id+' [spearman].pkl'), 'wb') as handle:
        pickle.dump(Corr, handle)
        
    C = pd.DataFrame(Corr)
    C.to_excel(join(figdata, code_id+' [spearman].xlsx'), sheet_name = 'data', index = False)
else:
    with open(join(figdata, code_id+' [spearman].pkl'), 'rb') as handle:
        Corr = pickle.load(handle)
        
    idx = np.concatenate([np.where(Corr['Training Day'] == day)[0] 
                                 for day in ['Day '+str(i) for i in range(1, 10)]+['>=Day 10']])
    Corr = SubDict(Corr, Corr.keys(), idx=idx)


CorrCollection = {
    "MiceID": [], "Maze Type": [], "Significance": [], "N Sessions": []
}
mice = [10209, 10212, 10224, 10227]
mazes = ['Maze 1', 'Maze 2']
for mouse in mice:
    for maze in mazes:
        CorrCollection['N Sessions'].append(np.where((Corr['MiceID'] == mouse)&(Corr['Maze Type'] == maze)&(Corr['spearman p'] >= 0.05))[0].shape[0])
        CorrCollection['Maze Type'].append(maze)
        CorrCollection['Significance'].append('ns')
        CorrCollection['MiceID'].append(mouse)
        
        CorrCollection['N Sessions'].append(np.where((Corr['MiceID'] == mouse)&(Corr['Maze Type'] == maze)&(Corr['spearman p'] < 0.05)&(Corr['spearman p'] >= 0.01))[0].shape[0])
        CorrCollection['Maze Type'].append(maze)
        CorrCollection['Significance'].append('*')
        CorrCollection['MiceID'].append(mouse)   

        CorrCollection['N Sessions'].append(np.where((Corr['MiceID'] == mouse)&(Corr['Maze Type'] == maze)&(Corr['spearman p'] < 0.01)&(Corr['spearman p'] >= 0.001))[0].shape[0])
        CorrCollection['Maze Type'].append(maze)
        CorrCollection['Significance'].append('**')
        CorrCollection['MiceID'].append(mouse)  

        CorrCollection['N Sessions'].append(np.where((Corr['MiceID'] == mouse)&(Corr['Maze Type'] == maze)&(Corr['spearman p'] < 0.001)&(Corr['spearman p'] >= 0.0001))[0].shape[0])
        CorrCollection['Maze Type'].append(maze)
        CorrCollection['Significance'].append('***')
        CorrCollection['MiceID'].append(mouse)  

        CorrCollection['N Sessions'].append(np.where((Corr['MiceID'] == mouse)&(Corr['Maze Type'] == maze)&(Corr['spearman p'] < 0.0001))[0].shape[0])
        CorrCollection['Maze Type'].append(maze)
        CorrCollection['Significance'].append('****')
        CorrCollection['MiceID'].append(mouse) 

for k in CorrCollection.keys():
    CorrCollection[k] = np.array(CorrCollection[k])
    
fig = plt.figure(figsize=(4,2)) 
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Significance', 
    y = 'N Sessions', 
    hue = 'Maze Type', 
    data = CorrCollection, 
    palette=sns.color_palette('rocket', 3)[1:],
    ax = ax,
    errwidth=0.5,
    capsize=0.15,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x = 'Significance', 
    y = 'N Sessions', 
    hue = 'Maze Type', 
    data = CorrCollection, 
    palette=sns.color_palette('Blues', 3)[1:],
    ax = ax,
    edgecolor='black',
    size=4,
    linewidth=0.2,
    dodge=True,
    jitter=0.2
)
plt.savefig(join(loc, 'Spearman_P-N_Sessions.png'), dpi=600)
plt.savefig(join(loc, 'Spearman_P-N_Sessions.svg'), dpi=600)
plt.close()
"""
file_indices = np.where((IndeptData['Maze Type'] != ['Open Field']))[0]
Data = SubDict(IndeptData, IndeptData.keys(), file_indices)
m1_num = np.where(Data['Maze Type'] == 'Maze 1')[0].shape[0]
m2_num = np.where(Data['Maze Type'] == 'Maze 2')[0].shape[0]

m1_pearson_sig = np.where((Data['Pearson P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 1'))[0].shape[0]
m2_pearson_sig = np.where((Data['Pearson P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 2'))[0].shape[0]

m1_spearman_sig = np.where((Data['Spearman P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 1'))[0].shape[0]
m2_spearman_sig = np.where((Data['Spearman P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 2'))[0].shape[0]

m1_indept_sig = np.where((Data['P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 1'))[0].shape[0]
m2_indept_sig = np.where((Data['P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 2'))[0].shape[0]

print(m1_num, m2_num)
print("Pearson", m1_pearson_sig, m2_pearson_sig)
print("Spearman", m1_spearman_sig, m2_spearman_sig)
print("Independent", m1_indept_sig, m2_indept_sig)

m1_pearson_spearman = np.where((Data['Spearman P-Value'] >= 0.05)&(Data['Pearson P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 1'))[0].shape[0]
m2_pearson_spearman = np.where((Data['Spearman P-Value'] >= 0.05)&(Data['Pearson P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 2'))[0].shape[0]

m1_pearson_indept = np.where((Data['P-Value'] >= 0.05)&(Data['Pearson P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 1'))[0].shape[0]
m2_pearson_indept = np.where((Data['P-Value'] >= 0.05)&(Data['Pearson P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 2'))[0].shape[0]

m1_spearman_indept = np.where((Data['P-Value'] >= 0.05)&(Data['Spearman P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 1'))[0].shape[0]
m2_spearman_indept = np.where((Data['P-Value'] >= 0.05)&(Data['Spearman P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 2'))[0].shape[0]

print(m1_pearson_spearman, m2_pearson_spearman)
print(m1_pearson_indept, m2_pearson_indept)
print(m1_spearman_indept, m2_spearman_indept)

m1_allsig = np.where((Data['P-Value'] >= 0.05)&(Data['Spearman P-Value'] >= 0.05)&(Data['Pearson P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 1'))[0].shape[0]
m2_allsig = np.where((Data['P-Value'] >= 0.05)&(Data['Spearman P-Value'] >= 0.05)&(Data['Pearson P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 2'))[0].shape[0]

print(m1_allsig, m2_allsig)
"""
#|(PData['nbinom KS P-Value'] >= 0.05))
"""
file_indices = np.where((PData['Poisson KS P-Value'] >= 0.05)&(PData['Maze Type'] != ['Open Field']))[0]
mkdir(join(loc, 'per session'))

xs, ys = np.array([]), np.array([])
fig = plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
for i in range(PData['Poisson KS P-Value'].shape[0]):
    if PData['Maze Type'][i] == 'Open Field':
        continue
    
    maze_type = 1 if PData['Maze Type'][i] == 'Maze 1' else 2
    if maze_type == 1:
        continue
    
    idx = np.where((f1['maze_type'] == maze_type)&(f1['MiceID'] == PData['MiceID'][i])&(f1['date'] == PData['date'][i]))[0]
    if len(idx) != 1:
        print(idx, "error length")
        continue
    
    k = idx[0]
    idx = np.where((Data['MiceID'] == f1['MiceID'][k])&(Data['date'] == f1['date'][k])&(Data['Stage'] == f1['Stage'][k])&(Data['Maze Type'] == PData['Maze Type'][i]))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    x, y = cp.deepcopy(SubData['Position']), cp.deepcopy(SubData['Field Length'])
    corr, p = spearmanr(x, y)
    print(i, PData['MiceID'][i], PData['date'][i], PData['Maze Type'][i], "length:", len(x))
    print("  ", corr,'  p:', p)
    xs = np.concatenate([xs, x])
    ys = np.concatenate([ys, y])
    ax.plot(x, y, 'o', color='black', markeredgewidth=0, markersize=2)
    ax.set_title(f"corr {corr}, p {p}")
    ax.plot(SubData['Position'], SubData['Field Length'], 'o', color='black', markeredgewidth=0, markersize=2)
    ax.set_xlabel("Position / m")
    ax.set_ylabel("Field Length / bins")
        
    plt.tight_layout()
    plt.savefig(join(loc, 'per session', f"Maze {f1['maze_type'][k]}-{f1['MiceID'][k]}-{f1['date'][k]}.png"), dpi = 600)
    plt.savefig(join(loc, 'per session', f"Maze {f1['maze_type'][k]}-{f1['MiceID'][k]}-{f1['date'][k]}.svg"), dpi = 600)
    ax.clear()
plt.close()

fig = plt.figure(figsize=(4, 3))
colors = [(255, 255, 255)]
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(xs, ys, 'o', color='black', markeredgewidth=0, markersize=1.5)
plt.tight_layout()
plt.savefig(join(loc, f"Maze A.png"), dpi = 600)
plt.savefig(join(loc, f"Maze A.svg"), dpi = 600)
plt.close()
print(pearsonr(xs, ys))
print(spearmanr(xs, ys))
#print(indeptest(xs, ys, simu_times=10000, shuffle_method='permutation'))


mkdir(join(loc, 'per session'))
xs, ys = np.array([]), np.array([])
fig = plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
for i in range(PData['Poisson KS P-Value'].shape[0]):
    if PData['Maze Type'][i] == 'Open Field':
        continue
    
    maze_type = 1 if PData['Maze Type'][i] == 'Maze 1' else 2
    if maze_type == 2:
        continue
    
    idx = np.where((f1['maze_type'] == maze_type)&(f1['MiceID'] == PData['MiceID'][i])&(f1['date'] == PData['date'][i]))[0]
    if len(idx) != 1:
        print(idx, "error length")
        continue
    
    k = idx[0]
    idx = np.where((Data['MiceID'] == f1['MiceID'][k])&(Data['date'] == f1['date'][k])&(Data['Stage'] == f1['Stage'][k])&(Data['Maze Type'] == PData['Maze Type'][i]))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    x, y = cp.deepcopy(SubData['Position']), cp.deepcopy(SubData['Field Length'])
    corr, p = spearmanr(x, y)
    print(i, PData['MiceID'][i], PData['date'][i], PData['Maze Type'][i], "length:", len(x))
    print("  ", corr,'  p:', p)
    xs = np.concatenate([xs, x])
    ys = np.concatenate([ys, y])
    ax.plot(x, y, 'o', color='black', markeredgewidth=0, markersize=2)
    ax.set_title(f"corr {corr}, p {p}")
    ax.plot(SubData['Position'], SubData['Field Length'], 'o', color='black', markeredgewidth=0, markersize=2)
    ax.set_xlabel("Position / m")
    ax.set_ylabel("Field Length / bins")
        
    plt.tight_layout()
    plt.savefig(join(loc, 'per session', f"Maze {f1['maze_type'][k]}-{f1['MiceID'][k]}-{f1['date'][k]}.png"), dpi = 600)
    plt.savefig(join(loc, 'per session', f"Maze {f1['maze_type'][k]}-{f1['MiceID'][k]}-{f1['date'][k]}.svg"), dpi = 600)
    ax.clear()
plt.close()

fig = plt.figure(figsize=(4, 3))
colors = [(255, 255, 255)]
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(xs, ys, 'o', color='black', markeredgewidth=0, markersize=1.5)
plt.tight_layout()
plt.savefig(join(loc, f"Maze B.png"), dpi = 600)
plt.savefig(join(loc, f"Maze B.svg"), dpi = 600)
plt.close()
print(pearsonr(xs, ys))
print(spearmanr(xs, ys))
print(indeptest(xs, ys, simu_times=10000, shuffle_method='permutation'))

mice = [10209, 10212]
date = {
    10209: [20230703],
    10212: [20230426, 20230428, 20230430, 20230502, 20230504, 20230506, 20230508, 20230510, 20230512, 20230515, 20230517, 20230519, 20230521],
    10224: [20230806, 20230808, 20230810, 20230812, 20230814, 20230816, 20230818, 20230820, 20230822, 20230824, 20230827, 20230829, 20230901],
    10227: [20230806, 20230808, 20230810, 20230812, 20230814, 20230816, 20230818, 20230820, 20230822, 20230824, 20230827, 20230829, 20230901]
}

# Maze 1 Stage 1
fig, axes = plt.subplots(ncols=13, nrows=4, figsize=(4*13,3*4))
for i, m in enumerate(mice):
    for j, d in enumerate(date[m]):
        idx = np.where((Data['MiceID'] == m)&(Data['Maze Type'] == 'Maze 1')&(Data['date'] == d))[0]
        
        if len(idx) == 0:
            ax = Clear_Axes(axes[i, j])
            continue
        
        SubData = SubDict(Data, Data.keys(), idx=idx)
        ax = Clear_Axes(axes[i, j], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        ax.plot(SubData['Position'], SubData['Field Length'], 'o', color='black', markeredgewidth=0, markersize=3)
        ax.set_xlabel("Position / m")
        ax.set_ylabel("Field Length / bins")
        
plt.tight_layout()
plt.savefig(join(loc, 'Position vs Field Length [Maze 1 Stage 1].png'), dpi = 600)
plt.savefig(join(loc, 'Position vs Field Length [Maze 1 Stage 1].svg'), dpi = 600)
plt.close()
"""