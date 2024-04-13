from mylib.statistic_test import *

code_id = '0204 - Cross Env field number'
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = {
        "MiceID": np.array([]),
        "date": np.array([]),
        "Training Day": np.array([]),
        "Stage": np.array([]),
        "Field Number": np.array([])
    }
    
    Corr = {
        "MiceID": np.array([]),
        "date": np.array([]),
        "Training Day": np.array([]),
        "Stage": np.array([]),
        "Corr": np.array([], np.float64)
    }
    
    for i in tqdm(range(len(f_CellReg_env))):
        if f_CellReg_env['include'][i] == 0 or f_CellReg_env['Stage'][i] != 'Stage 2':
            continue
        else:
            index_map = ReadCellReg(join(f_CellReg_env['cellreg_folder'][i], 'cellregistered.mat'))
            
            idx = np.where((f1['MiceID'] == f_CellReg_env['MiceID'][i])&(f1['date'] == f_CellReg_env['date'][i]))[0]
            
            if len(idx)!= 4:
                print(idx)
                assert False
                
            with open(f1['Trace File'][idx[1]], 'rb') as handle:
                trace1 = pickle.load(handle)
                
            with open(f1['Trace File'][idx[2]], 'rb') as handle:
                trace2 = pickle.load(handle)
                
            temp = np.where(index_map!= 0, 1, 0)
            cellpair_idx = np.where((temp[1, :] == 1) & (temp[2, :] == 1))[0]
            index_map = index_map[:, cellpair_idx].astype(np.int64)
            
            cellpair_idx = np.where((trace1['place_field_num_multiday'][index_map[1, :]-1] > 0)&
                                    (trace2['place_field_num_multiday'][index_map[2, :]-1] > 0))[0]
            index_map = index_map[:, cellpair_idx]
            
            Data['Field Number'] = np.concatenate([Data['Field Number'], trace1['place_field_num_multiday'][index_map[1, :]-1], trace2['place_field_num_multiday'][index_map[2, :]-1]])

            Data['Stage'] = np.concatenate([Data['Stage'], np.repeat(f_CellReg_env['Stage'][i], index_map.shape[1]*2)])
            Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(f_CellReg_env['MiceID'][i], index_map.shape[1]*2)])
            Data['date'] = np.concatenate([Data['date'], np.repeat(f_CellReg_env['date'][i], index_map.shape[1]*2)])
            Data['Training Day'] = np.concatenate([Data['Training Day'], np.repeat(f_CellReg_env['Training Day'][i], index_map.shape[1]*2)])
            
            Corr['Stage'] = np.concatenate([Corr['Stage'], np.repeat(f_CellReg_env['Stage'][i], 1)])
            Corr['MiceID'] = np.concatenate([Corr['MiceID'], np.repeat(f_CellReg_env['MiceID'][i], 1)])
            Corr['date'] = np.concatenate([Corr['date'], np.repeat(f_CellReg_env['date'][i], 1)])
            Corr['Training Day'] = np.concatenate([Corr['Training Day'], np.repeat(f_CellReg_env['Training Day'][i], 1)])
            pear_corr, _ = pearsonr(trace1['place_field_num_multiday'][index_map[1, :]-1], trace2['place_field_num_multiday'][index_map[2, :]-1])
            
            Corr['Corr'] = np.append(Corr['Corr'], pear_corr)
            
    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
        
    with open(os.path.join(figdata, code_id+' [Corr].pkl'), 'wb') as handle:
        pickle.dump(Corr, handle)
            
    D = pd.DataFrame(Data)
    D.to_excel(os.path.join(figdata, code_id+'.xlsx'), index=False)
        
    C = pd.DataFrame(Corr)
    C.to_excel(os.path.join(figdata, code_id+' [Corr].xlsx'), index=False)
        
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
    with open(os.path.join(figdata, code_id+' [Corr].pkl'), 'rb') as handle:
        Corr = pickle.load(handle)

idx = np.where((Data['MiceID'] != 11095)&(Data['MiceID'] != 11092))[0]
Data = SubDict(Data, Data.keys(), idx=idx)

idx = np.where((Corr['MiceID'] != 11095)&(Corr['MiceID'] != 11092))[0]
Corr = SubDict(Corr, Corr.keys(), idx=idx)

idx = np.where((Corr['Training Day'] == '>=Day 10')&(Corr['Stage'] == 'Stage 2'))[0]
SubData = SubDict(Corr, Corr.keys(), idx=idx)


fig = plt.figure(figsize=(3,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.where((Data['MiceID'] == 10224)&(Data['date'] == 20230918))[0]
print(pearsonr(Data['Field Number'][idx[:int(idx.shape[0]/2)]], Data['Field Number'][idx[int(idx.shape[0]/2):]]))
ax.plot(
    Data['Field Number'][idx[:int(idx.shape[0]/2)]] + 0.4*(np.random.rand(int(idx.shape[0]/2))-0.5), 
    Data['Field Number'][idx[int(idx.shape[0]/2):]] + 0.4*(np.random.rand(int(idx.shape[0]/2))-0.5), 
    'o',
    color = 'k',
    markeredgewidth = 0,
    markersize = 2
)
ax.set_aspect('equal')
ax.axis([0, 16, 0, 16])
ax.set_xticks(np.linspace(0, 16, 5))
ax.set_yticks(np.linspace(0, 16, 5))
ax.set_xlabel('Maze A')
ax.set_ylabel('Maze B')
plt.savefig(os.path.join(loc, 'Maze A vs Maze B.png'))
plt.savefig(os.path.join(loc, 'Maze A vs Maze B.svg'))
plt.close()


print_estimator(Corr['Corr'])
print(Corr['Corr'].shape)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(Corr['Corr'], bins=16, range=(-0.2, 0.6), color='gray')
ax.set_ylim(0, 15)
ax.set_yticks(np.linspace(0, 15, 4))
ax.set_xticks(np.linspace(-0.2, 0.6, 5))
plt.savefig(os.path.join(loc, 'Field Number Correlation.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Field Number Correlation.svg'), dpi=600)
plt.close()