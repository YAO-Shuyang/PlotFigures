from mylib.statistic_test import *

code_id = '0206 - Cell-pair Placecell fraction'
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = {"Env Pair": [], "Stage": [], "Training Day": [], "Placecell Fraction": []}
    
    for i in tqdm(range(len(f_CellReg_env))):
        if f_CellReg_env['include'][i] == 0:
            continue
        
        if f_CellReg_env['Stage'][i] == 'Stage 1':
            mazes = [["OF-MA", "OF-OF"], ["OF-MA"]]
            idx = np.where((f1['date'] == f_CellReg_env['date'][i])&(f1['MiceID'] == f_CellReg_env['MiceID'][i]))[0]
            if len(idx) != 3:
                print(idx)
                assert False
                
            index_map = ReadCellReg(join(f_CellReg_env['cellreg_folder'][i], "cellRegistered.mat"))
            
            for j in range(2):
                for k in range(j+1, 3):
                    Data['Env Pair'].append(mazes[j][k-j-1])
                    Data['Stage'].append(f_CellReg_env['Stage'][i])
                    Data['Training Day'].append(f_CellReg_env['Training Day'][i])
                    
                    with open(f1['Trace File'][idx[j]], 'rb') as handle:
                        trace1 = pickle.load(handle)
                    
                    with open(f1['Trace File'][idx[k]], 'rb') as handle:
                        trace2 = pickle.load(handle)
                        
                    cell_idx = np.where((index_map[j, :] >= 1) &
                                        (index_map[k, :] >= 1))[0]

                    pc_idx = np.where((trace1['is_placecell'][index_map[j, cell_idx].astype(np.int64)-1] == 1) &
                                      (trace2['is_placecell'][index_map[k, cell_idx].astype(np.int64)-1] == 1))[0]
                    
                    Data['Placecell Fraction'].append(len(pc_idx) / len(cell_idx))
                    
        elif f_CellReg_env['Stage'][i] == 'Stage 2':
            mazes = [["OF-MA", "OF-MB", "OF-OF"], ["MA-MB", "OF-MA"], ["OF-MB"]]
            idx = np.where((f1['date'] == f_CellReg_env['date'][i])&(f1['MiceID'] == f_CellReg_env['MiceID'][i]))[0]
            if len(idx) != 4:
                print(idx)
                assert False
                
            index_map = ReadCellReg(join(f_CellReg_env['cellreg_folder'][i], "cellRegistered.mat"))
            
            for j in range(3):
                for k in range(j+1, 4):
                    Data['Env Pair'].append(mazes[j][k-j-1])
                    Data['Stage'].append(f_CellReg_env['Stage'][i])
                    Data['Training Day'].append(f_CellReg_env['Training Day'][i])
                    
                    with open(f1['Trace File'][idx[j]], 'rb') as handle:
                        trace1 = pickle.load(handle)
                    
                    with open(f1['Trace File'][idx[k]], 'rb') as handle:
                        trace2 = pickle.load(handle)
                    
                    cell_idx = np.where((index_map[j, :] >= 1) &
                                        (index_map[k, :] >= 1))[0]
                    
                    pc_idx = np.where((trace1['is_placecell'][index_map[j, cell_idx].astype(np.int64)-1] == 1) &
                                      (trace2['is_placecell'][index_map[k, cell_idx].astype(np.int64)-1] == 1))[0]
                    
                    Data['Placecell Fraction'].append(len(pc_idx) / len(cell_idx))

    for k in Data.keys():
        Data[k] = np.array(Data[k])    
    
    with open(join(figdata, code_id+'.pkl'), 'wb') as f:
        pickle.dump(Data, f)
        
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)

idx = np.where((Data['Training Day'] != 'Day 1') & 
               (Data['Training Day'] != 'Day 2') &
               (Data['Training Day'] != 'Day 3'))[0]
Data = SubDict(Data, Data.keys(), idx=idx)

fig = plt.figure(figsize=(3, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x='Stage',
    hue='Env Pair',
    y='Placecell Fraction',
    hue_order=['OF-OF', 'OF-MA', 'OF-MB', 'MA-MB'],
    data=Data,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    ax=ax,
    errwidth=0.5,
    capsize=0.1,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x='Stage',
    hue='Env Pair',
    y='Placecell Fraction',
    hue_order=['OF-OF', 'OF-MA', 'OF-MB', 'MA-MB'],
    data=Data,
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    edgecolor='black',
    size=2,
    linewidth=0.15,
    ax = ax,
    jitter=0.2,
    dodge=True
)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(os.path.join(loc, 'Shared Place Cell Fraction.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Shared Place Cell Fraction.svg'), dpi=600)
plt.close()

print("Stage 1 ----------------------")
print("    OF - MA")
idx1 = np.where((Data['Env Pair'] == 'OF-MA')&(Data['Stage'] == 'Stage 1'))[0]
print_estimator(Data['Placecell Fraction'][idx1])

print("    OF - OF")
idx2 = np.where((Data['Env Pair'] == 'OF-OF')&(Data['Stage'] == 'Stage 1'))[0]
print_estimator(Data['Placecell Fraction'][idx2])

print("Stage 2 ----------------------")
print("    OF - MA")
idx3 = np.where((Data['Env Pair'] == 'OF-MA')&(Data['Stage'] == 'Stage 2'))[0]
print_estimator(Data['Placecell Fraction'][idx3])

print("    OF - MB")
idx4 = np.where((Data['Env Pair'] == 'OF-MB')&(Data['Stage'] == 'Stage 2'))[0]
print_estimator(Data['Placecell Fraction'][idx4])

print("    OF - OF")
idx5 = np.where((Data['Env Pair'] == 'OF-OF')&(Data['Stage'] == 'Stage 2'))[0]
print_estimator(Data['Placecell Fraction'][idx5])

print("    MA - MB")
idx6 = np.where((Data['Env Pair'] == 'MA-MB')&(Data['Stage'] == 'Stage 2'))[0]
print_estimator(Data['Placecell Fraction'][idx6])