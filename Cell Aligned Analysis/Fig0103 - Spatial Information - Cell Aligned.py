from mylib.statistic_test import *

code_id = '0103 - Spatial Information'
loc = os.path.join(figpath, 'Cell Aligned', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = {"MiceID": np.array([], np.int64), "Stage": np.array([]), "Training Day": np.array([]), "Date": np.array([], np.int64), 
            "SI OP": np.array([], dtype=np.float64), 
            "SI M1": np.array([], dtype=np.float64), 
            "SI M2": np.array([], dtype=np.float64),
            "PC OP": np.array([], dtype=np.float64),
            "PC M1": np.array([], dtype=np.float64),
            "PC M2": np.array([], dtype=np.float64),
            }
    
    for i in tqdm(range(len(f_CellReg_env))):
        if f_CellReg_env['include'][i] == 0:
            continue
        
        idx = np.where((f1['MiceID'] == f_CellReg_env['MiceID'][i])&(f1['date'] == f_CellReg_env['date'][i]))[0]
        if len(idx) <= 2:
            continue
    
        SIs, PCs = [], []
        for j in idx:
            if os.path.exists(f1['Trace File'][j]):
                with open(f1['Trace File'][j], 'rb') as handle:
                    trace = pickle.load(handle)
                    
                if trace['maze_type'] == 0:
                    SIs.append(trace['SI_all'])
                    PCs.append(trace['is_placecell'])
                else:
                    SIs.append(trace['LA']['SI_all'])
                    PCs.append(trace['LA']['is_placecell'])
            else:
                SIs.append([])
                PCs.append([])
        
        index_map = ReadCellReg(join(f_CellReg_env['cellreg_folder'][i], 'cellregistered.mat'))
        
        a = np.where(index_map != 0, 1, 0)
        alinged_num = np.sum(a, axis=0)
        
        idx = np.where(alinged_num > 1)[0]
        SI = np.zeros((index_map.shape[0], idx.shape[0]), dtype=np.float64)
        PC = np.zeros((index_map.shape[0], idx.shape[0]), dtype=np.float64)
        
        for j, k in enumerate(idx):
            for d in range(index_map.shape[0]):
                if index_map[d, k] != 0:
                    SI[d, j] = SIs[d][int(index_map[d, k])-1]
                    PC[d, j] = PCs[d][int(index_map[d, k])-1]
                else:
                    SI[d, j] = np.nan
                    PC[d, j] = np.nan
        
        Data['SI OP'] = np.concatenate([Data['SI OP'], np.nanmean(SI[[0,-1], :], axis=0)])
        Data['SI M1'] = np.concatenate([Data['SI M1'],SI[1, :]])
        
        if index_map.shape[0] == 4:
            Data['SI M2'] = np.concatenate([Data['SI M2'], SI[2, :]])
            Data['PC M2'] = np.concatenate([Data['PC M2'], PC[2, :]])
        else:
            Data['SI M2'] = np.concatenate([Data['SI M2'], np.repeat(np.nan, len(idx))])
            Data['PC M2'] = np.concatenate([Data['PC M2'], np.repeat(np.nan, len(idx))])
        
        Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(f_CellReg_env['MiceID'][i], len(idx))])
        Data['Stage'] = np.concatenate([Data['Stage'], np.repeat(f_CellReg_env['Stage'][i], len(idx))])
        Data['Training Day'] = np.concatenate([Data['Training Day'], np.repeat(f_CellReg_env['Training Day'][i], len(idx))])
        Data['Date'] = np.concatenate([Data['Date'], np.repeat(f_CellReg_env['date'][i], len(idx))])
        Data['PC OP'] = np.concatenate([Data['PC OP'], np.nanmax(PC[[0,-1], :], axis=0)])
        Data['PC M1'] = np.concatenate([Data['PC M1'], PC[1, :]])
    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)

else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

# Open Field vs Maze 1
idx1 = np.where((np.isnan(Data['SI OP']) == False)&(np.isnan(Data['SI M1']) == False)&(Data['PC OP'] == 1)&(Data['PC M1'] == 1))[0]
idx2 = np.where((np.isnan(Data['SI OP']) == False)&(np.isnan(Data['SI M1']) == False)&(Data['PC OP'] == 0)&(Data['PC M1'] == 1))[0]
print(idx1.shape, idx2.shape, 
      np.where((np.isnan(Data['SI OP']) == False)&(np.isnan(Data['SI M1']) == False)&(Data['PC OP'] == 1)&(Data['PC M1'] == 0))[0].shape,
      np.where((np.isnan(Data['SI OP']) == False)&(np.isnan(Data['SI M1']) == False)&(Data['PC OP'] == 0)&(Data['PC M1'] == 0))[0].shape)
print("Paired ttest:", ttest_rel(Data['SI OP'][idx1], Data['SI M1'][idx1]), cohen_d(Data['SI OP'][idx1], Data['SI M1'][idx1]))
print("Paired ttest:", ttest_rel(Data['SI OP'][idx2], Data['SI M1'][idx2]), cohen_d(Data['SI OP'][idx2], Data['SI M1'][idx2]))
fig = plt.figure(figsize=(4,4))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.set_aspect("equal")
ax.plot([0, 10], [0, 10], 'k:', linewidth = 0.5)
print(pearsonr(Data['SI OP'][idx1], Data['SI M1'][idx1]))
print(pearsonr(Data['SI OP'][idx2], Data['SI M1'][idx2]))
ax.plot(Data['SI OP'][idx1], Data['SI M1'][idx1], 'o', markeredgewidth=0, markersize=1)
ax.plot(Data['SI OP'][idx2], Data['SI M1'][idx2], 'o', markeredgewidth=0, markersize=1)
ax.axis([0, 8, 0, 8])
plt.savefig(os.path.join(loc, 'Open Field vs Maze 1.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Open Field vs Maze 1.svg'), dpi=600)
plt.close()

idx1 = np.where((np.isnan(Data['SI OP']) == False)&(np.isnan(Data['SI M2']) == False)&(Data['PC OP'] == 1)&(Data['PC M2'] == 1))[0]
idx2 = np.where((np.isnan(Data['SI OP']) == False)&(np.isnan(Data['SI M2']) == False)&(Data['PC OP'] == 0)&(Data['PC M2'] == 1))[0]
print(idx1.shape, idx2.shape, 
      np.where((np.isnan(Data['SI OP']) == False)&(np.isnan(Data['SI M2']) == False)&(Data['PC OP'] == 1)&(Data['PC M2'] == 0))[0].shape,
      np.where((np.isnan(Data['SI OP']) == False)&(np.isnan(Data['SI M2']) == False)&(Data['PC OP'] == 0)&(Data['PC M2'] == 0))[0].shape)
print("Paired ttest:", ttest_rel(Data['SI OP'][idx1], Data['SI M2'][idx1]), cohen_d(Data['SI OP'][idx1], Data['SI M2'][idx1]))
print("Paired ttest:", ttest_rel(Data['SI OP'][idx2], Data['SI M2'][idx2]), cohen_d(Data['SI OP'][idx2], Data['SI M2'][idx2]))
fig = plt.figure(figsize=(4,4))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.set_aspect("equal")
ax.plot([0, 10], [0, 10], 'k:', linewidth = 0.5)
print(pearsonr(Data['SI OP'][idx1], Data['SI M2'][idx1]))
print(pearsonr(Data['SI OP'][idx2], Data['SI M2'][idx2]))
ax.plot(Data['SI OP'][idx1], Data['SI M2'][idx1], 'o', markeredgewidth=0, markersize=1)
ax.plot(Data['SI OP'][idx2], Data['SI M2'][idx2], 'o', markeredgewidth=0, markersize=1)
ax.axis([0, 8, 0, 8])
plt.savefig(os.path.join(loc, 'Open Field vs Maze 2.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Open Field vs Maze 2.svg'), dpi=600)
plt.close()

idx = np.where((np.isnan(Data['SI M1']) == False)&(np.isnan(Data['SI M2']) == False)&(Data['PC M1'] == 1)&(Data['PC M2'] == 1))[0]
print(idx.shape, 
      np.where((np.isnan(Data['SI M1']) == False)&(np.isnan(Data['SI M2']) == False)&(Data['PC M1'] == 0)&(Data['PC M2'] == 1))[0].shape,
      np.where((np.isnan(Data['SI M1']) == False)&(np.isnan(Data['SI M2']) == False)&(Data['PC M1'] == 1)&(Data['PC M2'] == 0))[0].shape,
      np.where((np.isnan(Data['SI M1']) == False)&(np.isnan(Data['SI M2']) == False)&(Data['PC M1'] == 0)&(Data['PC M2'] == 0))[0].shape)
print("Paired ttest:", ttest_rel(Data['SI M1'][idx], Data['SI M2'][idx]), cohen_d(Data['SI M1'][idx], Data['SI M2'][idx]))
fig = plt.figure(figsize=(4,4))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.set_aspect("equal")
ax.plot([0, 10], [0, 10], 'k:', linewidth = 0.5)
print(pearsonr(Data['SI M1'][idx], Data['SI M2'][idx]))
ax.plot(Data['SI M1'][idx], Data['SI M2'][idx], 'o', markeredgewidth=0, markersize=1)
ax.axis([0, 8, 0, 8])
plt.savefig(os.path.join(loc, 'Maze 1 vs Maze 2.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Maze 1 vs Maze 2.svg'), dpi=600)
plt.close()
