from mylib.statistic_test import *
from scipy.stats import linregress

code_id = '0108 - Pairwise Correlation - Cell Aligned'
loc = os.path.join(figpath, code_id)
mkdir(loc)

def get_correlation(
    index_map: np.ndarray,
    file_indices: np.ndarray,
    f: pd.DataFrame = f1
):
    assert index_map.shape[0] == len(file_indices)
    index_map[np.where(np.isnan(index_map))] = 0
    index_map = index_map.astype(int)
    
    session = np.arange(1, index_map.shape[0])
    corr = np.zeros(index_map.shape[0]-1)
    
    for i in range(index_map.shape[0]-1):
        indexes = np.where((index_map[i, :] > 0) & (index_map[i+1, :] > 0))[0]
        
        with open(f['Trace File'][file_indices[i]], 'rb') as handle:
            trace1 = pickle.load(handle)

        with open(f['Trace File'][file_indices[i+1]], 'rb') as handle:
            trace2 = pickle.load(handle)
            
        pc_idx = np.where(
            (trace1['is_placecell'][index_map[i, indexes]-1] == 1) &
            (trace2['is_placecell'][index_map[i+1, indexes]-1] == 1)
        )[0]
        indexes = indexes[pc_idx]
        
        corrs = np.zeros(len(indexes))
        for j in range(len(indexes)):
            corrs[j], _ = pearsonr(
                trace1['smooth_map_all'][index_map[i, indexes[j]] - 1, :],
                trace2['smooth_map_all'][index_map[i+1, indexes[j]] - 1, :]
            )
        
        corr[i] = np.nanmean(corrs)
    
    return session, corr 
        
            
if exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = {
        'Maze Type': np.array([]),
        'MiceID': np.array([], np.int64),
        'Session Number': np.array([], np.int64),
        'Correlation': np.array([], np.float64),
        'Aligned Methods': np.array([]),
        'Paradigm': np.array([]),
    }
    
    
    for i in tqdm(range(len(f_CellReg_day))):
        if f_CellReg_day['include'][i] == 0:
            continue
        
        if f_CellReg_day['Type'][i] == 'Shuffle':
            continue
            
        try:
            index_map = GetMultidayIndexmap(
                    i=i,
                    occu_num=2
            )            
        except:
            index_map = ReadCellReg(f_CellReg_day['cellreg_folder'][i])
        index_map[np.where((index_map < 0)|np.isnan(index_map))] = 0
        mat = np.where(index_map>0, 1, 0)
        num = np.sum(mat, axis = 0)
        index_map = index_map[:, np.where(num >= 2)[0]]  

        mouse = f_CellReg_day['MiceID'][i]
        stage = f_CellReg_day['Stage'][i]
        session = f_CellReg_day['session'][i]

        file_indices = np.where((f1['MiceID'] == mouse) & (f1['Stage'] == stage) & (f1['session'] == session))[0]
        if stage == 'Stage 1+2':
            file_indices = np.where((f1['MiceID'] == mouse) & (f1['session'] == session) & ((f1['Stage'] == 'Stage 1') | (f1['Stage'] == 'Stage 2')))[0]
        
        if stage == 'Stage 1' and mouse in [10212] and session == 2:
            file_indices = np.where((f1['MiceID'] == mouse) & (f1['session'] == session) & (f1['Stage'] == 'Stage 1') & (f1['date'] != 20230506))[0]
        
        session, corr = get_correlation(
            index_map=index_map,
            file_indices=file_indices
        )
        
        days = len(session)
        maze_type = 'Open Field' if f_CellReg_day['maze_type'][i] == 0 else 'Maze '+str(f_CellReg_day['maze_type'][i])
        Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat(maze_type, days)])
        Data['Aligned Methods'] = np.concatenate([Data['Aligned Methods'], np.repeat('CellReg', days)])
        Data['Paradigm'] = np.concatenate([Data['Paradigm'], np.repeat('CrossMaze', days)])
        Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(f_CellReg_day['MiceID'][i], days)])
        Data['Session Number'] = np.concatenate([Data['Session Number'], session])
        Data['Correlation'] = np.concatenate([Data['Correlation'], corr])
        
    for i in tqdm(range(len(f_CellReg_modi))):
        if f_CellReg_modi['include'][i] == 0 :
            continue
        
        if f_CellReg_modi['Type'][i] == 'Shuffle':
            continue
            
        if f_CellReg_modi['paradigm'][i] == 'CrossMaze':
            if f_CellReg_modi['maze_type'][i] == 0:
                index_map = GetMultidayIndexmap(
                    mouse=f_CellReg_modi['MiceID'][i],
                    stage=f_CellReg_modi['Stage'][i],
                    session=f_CellReg_modi['session'][i],
                    occu_num=2
                )    
            else:
                with open(f_CellReg_modi['cellreg_folder'][i], 'rb') as handle:
                    index_map = pickle.load(handle)
        else:
            continue
            index_map = ReadCellReg(f_CellReg_modi['cellreg_folder'][i])
                
        index_map[np.where((index_map < 0)|np.isnan(index_map))] = 0
        mat = np.where(index_map>0, 1, 0)
        num = np.sum(mat, axis = 0)
        index_map = index_map[:, np.where(num >= 2)[0]]  
        
        mouse = f_CellReg_modi['MiceID'][i]
        stage = f_CellReg_modi['Stage'][i]
        session = f_CellReg_modi['session'][i]

        file_indices = np.where((f1['MiceID'] == mouse) & (f1['Stage'] == stage) & (f1['session'] == session))[0]
        if stage == 'Stage 1+2':
            file_indices = np.where((f1['MiceID'] == mouse) & (f1['session'] == session) & ((f1['Stage'] == 'Stage 1') | (f1['Stage'] == 'Stage 2')))[0]
        
        if stage == 'Stage 1' and mouse in [10212] and session == 2:
            file_indices = np.where((f1['MiceID'] == mouse) & (f1['session'] == session) & (f1['Stage'] == 'Stage 1') & (f1['date'] != 20230506))[0]
        
        session, corr = get_correlation(
            index_map=index_map,
            file_indices=file_indices
        )
        
        days = len(session)
        maze_type = 'Open Field' if f_CellReg_modi['maze_type'][i] == 0 else 'Maze '+str(f_CellReg_modi['maze_type'][i])
        Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat(maze_type, days)])
        Data['Aligned Methods'] = np.concatenate([Data['Aligned Methods'], np.repeat('NeuroMatch', days)])
        Data['Paradigm'] = np.concatenate([Data['Paradigm'], np.repeat('CrossMaze', days)])
        Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(f_CellReg_modi['MiceID'][i], days)])
        Data['Session Number'] = np.concatenate([Data['Session Number'], session])
        Data['Correlation'] = np.concatenate([Data['Correlation'], corr])
        
    with open(join(figdata, code_id+'.pkl'), 'wb') as f:
        pickle.dump(Data, f)
        
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)

Data['Aligned Methods'][337:] = 'NeuroMatch' 
idx = np.where((Data['Paradigm']=='CrossMaze') &
               (Data['Maze Type']=='Maze 1') &
               (Data['MiceID'] != 10224) &
               (Data['MiceID'] != 10227))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(3, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
sns.barplot(
    x='Session Number',
    y='Correlation',
    data=SubData,
    hue='Aligned Methods',
    palette=['#336699', '#C3AED6'],
    capsize=0.2,
    width=0.8,
    ax=ax,
    errcolor='black',
    errwidth=0.5
)
sns.stripplot(
    x='Session Number',
    y='Correlation',
    data=SubData,
    hue='Aligned Methods',
    palette=['#C3DEF1', '#FED7D7'],
    jitter=0.2,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True
)
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, '[Maze A 09&12] Pair-wise Correlation [session interval = 1].png'), dpi=600)
plt.savefig(join(loc, '[Maze A 09&12] Pair-wise Correlation [session interval = 1].svg'), dpi=600)
plt.close()

idx = np.where((Data['Paradigm']=='CrossMaze') &
               (Data['Maze Type']=='Maze 1') &
               (Data['MiceID'] != 10209) &
               (Data['MiceID'] != 10212))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(5, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
sns.barplot(
    x='Session Number',
    y='Correlation',
    data=SubData,
    hue='Aligned Methods',
    palette=['#336699', '#C3AED6'],
    capsize=0.2,
    width=0.8,
    ax=ax,
    errcolor='black',
    errwidth=0.5
)
sns.stripplot(
    x='Session Number',
    y='Correlation',
    data=SubData,
    hue='Aligned Methods',
    palette=['#C3DEF1', '#FED7D7'],
    jitter=0.2,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True
)
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, '[Maze A 24&27] Pair-wise Correlation [session interval = 1].png'), dpi=600)
plt.savefig(join(loc, '[Maze A 24&27] Pair-wise Correlation [session interval = 1].svg'), dpi=600)
plt.close()

idx = np.where((Data['Paradigm']=='CrossMaze') &
               (Data['Maze Type']=='Maze 2'))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(3, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
sns.barplot(
    x='Session Number',
    y='Correlation',
    data=SubData,
    hue='Aligned Methods',
    palette=['#336699', '#C3AED6'],
    capsize=0.2,
    width=0.8,
    ax=ax,
    errcolor='black',
    errwidth=0.5
)
sns.stripplot(
    x='Session Number',
    y='Correlation',
    data=SubData,
    hue='Aligned Methods',
    palette=['#C3DEF1', '#FED7D7'],
    jitter=0.2,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True
)
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, '[Maze B] Pair-wise Correlation [session interval = 1].png'), dpi=600)
plt.savefig(join(loc, '[Maze B] Pair-wise Correlation [session interval = 1].svg'), dpi=600)
plt.close()

print("Maze 1")
idx = np.where((Data['Paradigm']=='CrossMaze') &
               (Data['Maze Type']=='Maze 1'))[0]
SubData = SubDict(Data, Data.keys(), idx)
x = np.unique(SubData['Session Number'])
for i in x:
    idx1 = np.where((SubData['Session Number'] == i) & (SubData['Aligned Methods'] == 'CellReg'))[0]
    idx2 = np.where((SubData['Session Number'] == i) & (SubData['Aligned Methods'] == 'NeuroMatch'))[0]
    print("  ", i, ttest_rel(SubData['Correlation'][idx1], SubData['Correlation'][idx2]))
print(end='\n\n')

print("Maze 2")
idx = np.where((Data['Paradigm']=='CrossMaze') &
               (Data['Maze Type']=='Maze 2'))[0]
SubData = SubDict(Data, Data.keys(), idx)
x = np.unique(SubData['Session Number'])
for i in x:
    idx1 = np.where((SubData['Session Number'] == i) & (SubData['Aligned Methods'] == 'CellReg'))[0]
    idx2 = np.where((SubData['Session Number'] == i) & (SubData['Aligned Methods'] == 'NeuroMatch'))[0]
    print("  ", i, ttest_rel(SubData['Correlation'][idx1], SubData['Correlation'][idx2]))
print(end='\n\n')