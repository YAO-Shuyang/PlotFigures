from mylib.statistic_test import *

code_id = "0109 - Cells detected in how many sessions"
loc = join(figpath, 'Cell Aligned', code_id)
mkdir(loc)

def get_detected_num(index_map: np.ndarray):
    days = index_map.shape[0]
    res = np.zeros(days, dtype=np.int64)
    res2 = np.zeros(days, dtype=np.int64)
    
    detected_mat = np.where(index_map!=0, 1, 0)
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
        'Maze Type': np.array([], np.int64),
        'Mice ID': np.array([], np.int64),
        'Session Number': np.array([], np.int64),
        'Count': np.array([], np.int64),
        'Cumulative Count': np.array([], np.int64),
    }
    
    
    for i in tqdm(range(len(f_CellReg_day))):
        if f_CellReg_day['Stage'][i] not in ['Stage 1', 'Stage 2'] or f_CellReg_day['include'][i] == 0:
            continue
            
        index_map = GetMultidayIndexmap(
            i=i,
            occu_num=1
        )
        count, cumulative_count, num = get_detected_num(index_map)
        days = len(num)
        maze_type = 'Open Field' if f_CellReg_day['maze_type'][i] == 0 else 'Maze '+str(f_CellReg_day['maze_type'][i])
        Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat(maze_type, days)])
        Data['Mice ID'] = np.concatenate([Data['Mice ID'], np.repeat(f_CellReg_day['MiceID'][i], days)])
        Data['Session Number'] = np.concatenate([Data['Session Number'], num])
        Data['Count'] = np.concatenate([Data['Count'], count])
        Data['Cumulative Count'] = np.concatenate([Data['Cumulative Count'], cumulative_count])
        
    with open(join(figdata, code_id+'.pkl'), 'wb') as f:
        pickle.dump(Data, f)
        
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)

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