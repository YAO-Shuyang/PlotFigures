from mylib.statistic_test import *

code_id = "0109 - Cells detected in how many sessions"
loc = join(figpath, 'Cell Aligned', code_id)
mkdir(loc)

def get_detected_num(index_map: np.ndarray):
    days = index_map.shape[0]
    res = np.zeros(days, dtype=np.int64)
    res2 = np.zeros(days, dtype=np.int64)
    
    detected_mat = np.where(index_map>0, 1, 0)
    cell_num = np.nansum(detected_mat, axis=0)
    
    for d in range(1, days+1):
        res[d-1] = len(np.where(cell_num==d)[0])
        res2[d-1] = len(np.where(cell_num>=d)[0])
    
    return res, res2, np.arange(1, days+1)


if exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = {
        'Maze Type': np.array([]),
        'MiceID': np.array([], np.int64),
        'Session Number': np.array([], np.int64),
        'Count': np.array([], np.int64),
        'Cumulative Count': np.array([], np.int64),
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
            
        count, cumulative_count, num = get_detected_num(index_map)
        days = len(num)
        maze_type = 'Open Field' if f_CellReg_day['maze_type'][i] == 0 else 'Maze '+str(f_CellReg_day['maze_type'][i])
        Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat(maze_type, days)])
        Data['Aligned Methods'] = np.concatenate([Data['Aligned Methods'], np.repeat('CellReg', days)])
        Data['Paradigm'] = np.concatenate([Data['Paradigm'], np.repeat('CrossMaze', days)])
        Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(f_CellReg_day['MiceID'][i], days)])
        Data['Session Number'] = np.concatenate([Data['Session Number'], num])
        Data['Count'] = np.concatenate([Data['Count'], count])
        Data['Cumulative Count'] = np.concatenate([Data['Cumulative Count'], cumulative_count])
        
    for i in tqdm(range(len(f_CellReg_modi))):
        if f_CellReg_modi['include'][i] == 0:
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
            index_map = ReadCellReg(f_CellReg_modi['cellreg_folder'][i])
                
        index_map[np.where((index_map < 0)|np.isnan(index_map))] = 0
        mat = np.where(index_map>0, 1, 0)
        num = np.sum(mat, axis = 0)
        index_map = index_map[:, np.where(num >= 2)[0]]  
        
        count, cumulative_count, num = get_detected_num(index_map)
        days = len(num)
        maze_type = 'Open Field' if f_CellReg_modi['maze_type'][i] == 0 else 'Maze '+str(f_CellReg_modi['maze_type'][i])
        Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat(maze_type, days)])
        Data['Aligned Methods'] = np.concatenate([Data['Aligned Methods'], np.repeat('NeuroMatch', days)])
        Data['Paradigm'] = np.concatenate([Data['Paradigm'], np.repeat(f_CellReg_modi['paradigm'][i], days)])
        Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(f_CellReg_modi['MiceID'][i], days)])
        Data['Session Number'] = np.concatenate([Data['Session Number'], num])
        Data['Count'] = np.concatenate([Data['Count'], count])
        Data['Cumulative Count'] = np.concatenate([Data['Cumulative Count'], cumulative_count])
        
    with open(join(figdata, code_id+'.pkl'), 'wb') as f:
        pickle.dump(Data, f)
        
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)

idx = np.where((Data['Paradigm']!='CrossMaze') &
               (Data['Session Number'] > 1))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(3, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
sns.barplot(
    x='Session Number',
    y='Count',
    data=SubData,
    hue='Paradigm',
    palette=['#003366', '#66CCCC'],
    capsize=0.2,
    width=0.8,
    ax=ax,
    errcolor='black',
    errwidth=0.5
)
sns.stripplot(
    x='Session Number',
    y='Count',
    data=SubData,
    hue='Paradigm',
    palette=['#F2E8D4', '#8E9F85'],
    jitter=0.2,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True
)
ax.set_ylim(0, 400)
ax.set_yticks(np.linspace(0, 400, 5))
plt.savefig(join(loc, '[Hairpin & Reverse] Aligned Number.png'), dpi=600)
plt.savefig(join(loc, '[Hairpin & Reverse] Aligned Number.svg'), dpi=600)
plt.close()

idx = np.where((Data['Paradigm']=='CrossMaze') &
               (Data['Maze Type']=='Open Field') &
               (Data['Aligned Methods']=='CellReg') &
               (Data['Session Number'] > 1))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(3, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
sns.barplot(
    x='Session Number',
    y='Count',
    data=SubData,
    hue='Maze Type',
    palette=sns.color_palette("rocket", 3)[:1],
    capsize=0.3,
    width=0.8,
    ax=ax,
    errcolor='black',
    errwidth=0.5
)
sns.stripplot(
    x='Session Number',
    y='Count',
    data=SubData,
    hue='Maze Type',
    palette=['#B9EBD3'],
    jitter=0.2,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True
)
ax.set_ylim(0, 250)
ax.set_yticks(np.linspace(0, 250, 6))
plt.savefig(join(loc, '[Open Field] Aligned Number.png'), dpi=600)
plt.savefig(join(loc, '[Open Field] Aligned Number.svg'), dpi=600)
plt.close()

idx = np.where((Data['Paradigm']=='CrossMaze') &
               (Data['Maze Type']=='Maze 1') &
               (Data['MiceID'] != 10224) &
               (Data['MiceID'] != 10227) &
               (Data['Session Number'] > 1))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(3, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
sns.barplot(
    x='Session Number',
    y='Count',
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
    y='Count',
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
ax.set_ylim(0, 200)
ax.set_yticks(np.linspace(0, 200, 5))
plt.savefig(join(loc, '[Maze A 09&12] Aligned Number.png'), dpi=600)
plt.savefig(join(loc, '[Maze A 09&12] Aligned Number.svg'), dpi=600)
plt.close()

idx = np.where((Data['Paradigm']=='CrossMaze') &
               (Data['Maze Type']=='Maze 1') &
               (Data['MiceID'] != 10209) &
               (Data['MiceID'] != 10212) &
               (Data['Session Number'] > 1))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(5, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
sns.barplot(
    x='Session Number',
    y='Count',
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
    y='Count',
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
ax.set_ylim(0, 400)
ax.set_yticks(np.linspace(0, 400, 5))
plt.savefig(join(loc, '[Maze A 24&27] Aligned Number.png'), dpi=600)
plt.savefig(join(loc, '[Maze A 24&27] Aligned Number.svg'), dpi=600)
plt.close()

idx = np.where((Data['Paradigm']=='CrossMaze') &
               (Data['Maze Type']=='Maze 2') &
               (Data['Session Number'] > 1))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(3, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
sns.barplot(
    x='Session Number',
    y='Count',
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
    y='Count',
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
ax.set_ylim(0, 200)
ax.set_yticks(np.linspace(0, 200, 5))
plt.savefig(join(loc, '[Maze B] Aligned Number.png'), dpi=600)
plt.savefig(join(loc, '[Maze B] Aligned Number.svg'), dpi=600)
plt.close()
"""
colors = sns.color_palette("rocket", 3)
plt.figure(figsize=(6,4))
idx = np.where(Data['Session Number']>1)[0]
SubData = SubDict(Data, Data.keys(), idx)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
sns.barplot(
    x='Session Number',
    y='Count',
    data=SubData,
    hue='Maze Type',
    palette=colors,
    capsize=0.1,
    width=0.8,
    ax=ax,
    errcolor='black',
    errwidth=0.2
)
ax.set_yticks(np.linspace(0, 200,5))
plt.tight_layout()
plt.savefig(join(loc, 'Aligned Number.png'), dpi=600)
plt.savefig(join(loc, 'Aligned Number.svg'), dpi=600)
plt.close()

colors = sns.color_palette("rocket", 3)
plt.figure(figsize=(6,4))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
sns.barplot(
    x='Session Number',
    y='Cumulative Count',
    data=Data,
    hue='Maze Type',
    palette=colors,
    capsize=0.2,
    width=0.8,
    ax=ax,
    errcolor='black',
    errwidth=0.5
)
ax.set_yticks(np.linspace(0,1600, 5))
plt.tight_layout()
plt.savefig(join(loc, 'Cumulative Number.png'), dpi=600)
plt.savefig(join(loc, 'Cumulative Number.svg'), dpi=600)
plt.close()

def get_detected_num_crossenv(index_map: np.ndarray, stage: str):
    detected_mat = np.where(index_map!=0, 1, 0)
    cell_num = np.nansum(detected_mat, axis=0)
    
    if stage == 'Stage 1':
        assert index_map.shape[0] == 3
        
        all_3 = len(np.where(cell_num == 3)[0])
        op_op = len(np.where((cell_num == 2)&(detected_mat[0, :] == 1)&(detected_mat[2, :] == 1))[0])
        op_m1 = len(np.where((cell_num == 2)&(detected_mat[0, :] == 1)&(detected_mat[1, :] == 1))[0])+len(np.where((cell_num == 2)&(detected_mat[2, :] == 1)&(detected_mat[1, :] == 1))[0])
        single = len(np.where(cell_num == 1)[0])
        
        return [all_3, op_op, op_m1, single], ["All Sessions", "Open Field - Open Field", "Open Field - Maze 1", "No Match"]
        
    elif stage == 'Stage 2':
        assert index_map.shape[0] == 4

        all_4 = len(np.where(cell_num == 4)[0])
        op_op = len(np.where((detected_mat[0, :] == 1)&(detected_mat[3, :] == 1))[0])
        op_m1 = len(np.where((detected_mat[0, :] == 1)&(detected_mat[1, :] == 1))[0])+len(np.where((detected_mat[3, :] == 1)&(detected_mat[1, :] == 1))[0])
        op_m2 = len(np.where((detected_mat[0, :] == 1)&(detected_mat[2, :] == 1))[0])+len(np.where((detected_mat[3, :] == 1)&(detected_mat[2, :] == 1))[0])
        m1_m2 = len(np.where((detected_mat[1, :] == 1)&(detected_mat[2, :] == 1))[0])
        single = len(np.where(cell_num == 1)[0])
        
        return [all_4, op_op, op_m1, op_m2, m1_m2, single], ["All Sessions", "Open Field - Open Field", "Open Field - Maze 1", "Open Field - Maze 2", "Maze 1 - Maze 2", "No Match"]
        
    else:
        raise ValueError(f"{stage} is an invalid value for stage that is currently not supported.")
    
    

if exists(join(figdata, code_id+' [cross env].pkl')):
    with open(join(figdata, code_id+' [cross env].pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = {
        'date': np.array([], np.int64),
        'Mice ID': np.array([], np.int64),
        'Stage': np.array([]),
        'Class': np.array([], np.int64),
        'Count': np.array([], np.int64),
    }
    
    
    for i in tqdm(range(len(f_CellReg_env))):
        if f_CellReg_env['Stage'][i] not in ['Stage 1', 'Stage 2'] or f_CellReg_env['include'][i] == 0:
            continue
            
        index_map = ReadCellReg(os.path.join(f_CellReg_env['cellreg_folder'][i], 'cellregistered.mat'))
        count, dclass = get_detected_num_crossenv(index_map, f_CellReg_env['Stage'][i])
        class_num = len(dclass)
        Data['date'] = np.concatenate([Data['date'], np.repeat(f_CellReg_env['date'][i], class_num)])
        Data['Stage'] = np.concatenate([Data['Stage'], np.repeat(f_CellReg_env['Stage'][i], class_num)])
        Data['Mice ID'] = np.concatenate([Data['Mice ID'], np.repeat(f_CellReg_env['MiceID'][i], class_num)])
        Data['Class'] = np.concatenate([Data['Class'], dclass])
        Data['Count'] = np.concatenate([Data['Count'], count])
        
    with open(join(figdata, code_id+' [cross env].pkl'), 'wb') as f:
        pickle.dump(Data, f)
        
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+' [cross env].xlsx'), index=False)
    

colors = sns.color_palette("rocket", 3)
plt.figure(figsize=(6,4))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
sns.barplot(
    x='Stage',
    y='Count',
    data=Data,
    hue = 'Class',
    palette='rocket',
    capsize=0.2,
    width=0.9,
    errcolor='black',
    errwidth=0.5
)
plt.tight_layout()
plt.savefig(join(loc, 'Aligned Number [env].png'), dpi=600)
plt.savefig(join(loc, 'Aligned Number [env].svg'), dpi=600)
plt.close()
"""