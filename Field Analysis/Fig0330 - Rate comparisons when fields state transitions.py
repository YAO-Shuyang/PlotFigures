from mylib.statistic_test import *
from scipy.stats import linregress

code_id = '0330 - Rate comparisons when fields state transitions'
loc = os.path.join(figpath, code_id)
mkdir(loc)

def get_rate_changes(
    field_reg: np.ndarray,
    field_info: np.ndarray,
    file_indices: np.ndarray,
    f: pd.DataFrame = f1,
    prefix = None
):
   
    category, rate, order = [], [], []
    session = []
    
    for i in range(field_reg.shape[0]-1):
        cate_per_session, rate_per_session, order_per_session = [], [], []
        indexes = np.where((np.isnan(field_reg[i, :]) == False) & 
                           (np.isnan(field_reg[i+1, :]) == False))[0]
        
        with open(f['Trace File'][file_indices[i]], 'rb') as handle:
            trace1 = pickle.load(handle)

        with open(f['Trace File'][file_indices[i+1]], 'rb') as handle:
            trace2 = pickle.load(handle)
            
        if prefix is not None:
            trace1 = trace1[prefix]
            trace2 = trace2[prefix]
            
        for j in indexes:
            cell_index1 = int(field_info[i, j, 0])
            cell_index2 = int(field_info[i+1, j, 0])
            
            if field_reg[i, j] == 1 and field_reg[i+1, j] == 1:
                # retention
                field_center1 = int(field_info[i, j, 2])
                field_center2 = int(field_info[i+1, j, 2])
                rate_per_session.append(trace1['smooth_map_all'][cell_index1-1, field_center1-1])
                rate_per_session.append(trace2['smooth_map_all'][cell_index2-1, field_center2-1])
                cate_per_session = cate_per_session + [3, 3]
                order_per_session = order_per_session + ['Prev', 'Next']
            elif field_reg[i, j] == 1 and field_reg[i+1, j] == 0:
                # disappearance
                field_center = int(field_info[i, j, 2])
                rate_per_session.append(trace1['smooth_map_all'][cell_index1-1, field_center-1])
                rate_per_session.append(trace2['smooth_map_all'][cell_index2-1, field_center-1])
                cate_per_session = cate_per_session + [1, 1]
                order_per_session = order_per_session + ['Prev', 'Next']
            elif field_reg[i, j] == 0 and field_reg[i+1, j] == 1:
                # formation
                field_center = int(field_info[i+1, j, 2])
                rate_per_session.append(trace1['smooth_map_all'][cell_index1-1, field_center-1])
                rate_per_session.append(trace2['smooth_map_all'][cell_index2-1, field_center-1])
                cate_per_session = cate_per_session + [2, 2]
                order_per_session = order_per_session + ['Prev', 'Next']
            else:
                continue
            
        for ord in ['Prev', 'Next']:
            for cate in [1, 2, 3]:
                idx = np.where((np.array(cate_per_session) == cate)&(np.array(order_per_session) == ord))[0]
                rate.append(np.mean(np.array(rate_per_session)[idx]))
                category.append(cate)
                order.append(ord)
                session.append(i+1)
    
    return np.array(category, np.int64), np.array(rate, np.float64), np.array(order), np.array(session, np.int64)
        
            
if exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = {
        'Maze Type': np.array([]),
        'MiceID': np.array([], np.int64),
        'Category': np.array([], np.int64),
        'Rate': np.array([], np.float64),
        'Order': np.array([]),
        'Session': np.array([], np.int64),
        'Paradigm': np.array([]),
    }
    
        
    for i in tqdm(range(len(f_CellReg_modi))):
        if f_CellReg_modi['include'][i] == 0 or f_CellReg_modi['maze_type'][i] == 0:
            continue
        
        if f_CellReg_modi['Type'][i] == 'Shuffle':
            continue

        mouse = f_CellReg_modi['MiceID'][i]
        stage = f_CellReg_modi['Stage'][i]
        session = f_CellReg_modi['session'][i]
                    
        if f_CellReg_modi['paradigm'][i] == 'CrossMaze':

            file_indices = np.where((f1['MiceID'] == mouse) & (f1['Stage'] == stage) & (f1['session'] == session))[0]
            if stage == 'Stage 1+2':
                file_indices = np.where((f1['MiceID'] == mouse) & (f1['session'] == session) & ((f1['Stage'] == 'Stage 1') | (f1['Stage'] == 'Stage 2')))[0]
        
            if stage == 'Stage 1' and mouse in [10212] and session == 2:
                file_indices = np.where((f1['MiceID'] == mouse) & (f1['session'] == session) & (f1['Stage'] == 'Stage 1') & (f1['date'] != 20230506))[0]
        
            with open(f_CellReg_modi['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
                
            category, rate, order, session = get_rate_changes(
                field_reg=trace['field_reg'],
                field_info=trace['field_info'],
                file_indices=file_indices,
                f = f1
            )
            days = len(session)
            maze_type = 'Open Field' if f_CellReg_modi['maze_type'][i] == 0 else 'Maze '+str(f_CellReg_modi['maze_type'][i])
            Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat(maze_type, days)])
            Data['Paradigm'] = np.concatenate([Data['Paradigm'], np.repeat('CrossMaze', days)])
            Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(f_CellReg_modi['MiceID'][i], days)])
            Data['Session'] = np.concatenate([Data['Session'], session])
            Data['Category'] = np.concatenate([Data['Category'], category])
            Data['Rate'] = np.concatenate([Data['Rate'], rate])
            Data['Order'] = np.concatenate([Data['Order'], order])
        else:
            if f_CellReg_modi['paradigm'][i] == 'ReverseMaze':
                f = f3
                file_indices = np.where(f['MiceID'] == mouse)[0]
            else:
                f = f4
                file_indices = np.where(f['MiceID'] == mouse)[0]
                
            with open(f_CellReg_modi['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
                
            category_cis, rate_cis, order_cis, session_cis = get_rate_changes(
                field_reg=trace['cis']['field_reg'],
                field_info=trace['cis']['field_info'],
                file_indices=file_indices,
                f = f,
                prefix='cis'
            )
            
            category_trs, rate_trs, order_trs, session_trs = get_rate_changes(
                field_reg=trace['trs']['field_reg'],
                field_info=trace['trs']['field_info'],
                file_indices=file_indices,
                f = f,
                prefix='trs'
            )
            
            n_cis, n_trs = len(session_cis), len(session_trs)
            maze_type = 'Open Field' if f_CellReg_modi['maze_type'][i] == 0 else 'Maze '+str(f_CellReg_modi['maze_type'][i])
            Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat(maze_type, n_cis+n_trs)])
            Data['Paradigm'] = np.concatenate([Data['Paradigm'], 
                                               np.repeat(f_CellReg_modi['paradigm'][i]+' cis', n_cis),
                                               np.repeat(f_CellReg_modi['paradigm'][i]+' trs', n_trs)])
            Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(f_CellReg_modi['MiceID'][i], n_cis+n_trs)])
            Data['Session'] = np.concatenate([Data['Session'], session_cis, session_trs])
            Data['Category'] = np.concatenate([Data['Category'], category_cis, category_trs])
            Data['Rate'] = np.concatenate([Data['Rate'], rate_cis, rate_trs])
            Data['Order'] = np.concatenate([Data['Order'], order_cis, order_trs])
        
    with open(join(figdata, code_id+'.pkl'), 'wb') as f:
        pickle.dump(Data, f)
    
    try:
        D = pd.DataFrame(Data)
        D.to_excel(join(figdata, code_id+'.xlsx'), index=False)
    except:
        pass

Data['hue'] = np.array([Data['Paradigm'][i] + Data['Maze Type'][i] for i in range(Data['Paradigm'].shape[0])])

# Field disappearance
idx = np.where(Data['Category'] == 1)[0]
SubData = SubDict(Data, Data.keys(), idx)

print("Field Disappearance:")
for hue in np.unique(SubData['hue']):
    prev_idx = np.where((SubData['Order'] == 'Prev')&(SubData['hue'] == hue))[0]
    next_idx = np.where((SubData['Order'] == 'Next')&(SubData['hue'] == hue))[0]
    print("   ", hue, "   ", ttest_rel(SubData['Rate'][prev_idx], SubData['Rate'][next_idx]))
print()
    
print(ttest_rel(SubData['Rate'][prev_idx], SubData['Rate'][next_idx]))
plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
sns.barplot(
    x = 'hue',
    y = 'Rate',
    data = SubData,
    hue = 'Order',
    palette=['#F8B195', '#6C5B7B', ],
    capsize = 0.2,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5,
    ax=ax
)
sns.stripplot(
    x = 'hue',
    y = 'Rate',
    data = SubData,
    hue = 'Order',
    palette=['#F8B195', '#6C5B7B', ],
    edgecolor='black',
    size=3,
    linewidth=0.1,
    ax=ax,
    dodge=True,
    zorder = 1
)
ax.set_ylim([0, 1.5])
ax.set_yticks(np.linspace(0, 1.5, 7))
plt.savefig(join(loc, 'Field disappearance.png'), dpi = 600)
plt.savefig(join(loc, 'Field disappearance.svg'), dpi = 600)
plt.close()

# Field Formation
idx = np.where(Data['Category'] == 2)[0]
SubData = SubDict(Data, Data.keys(), idx)

print("Field Formation:")
for hue in np.unique(SubData['hue']):
    prev_idx = np.where((SubData['Order'] == 'Prev')&(SubData['hue'] == hue))[0]
    next_idx = np.where((SubData['Order'] == 'Next')&(SubData['hue'] == hue))[0]
    print("   ", hue, "   ", ttest_rel(SubData['Rate'][prev_idx], SubData['Rate'][next_idx]))
print()

plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
sns.barplot(
    x = 'hue',
    y = 'Rate',
    data = SubData,
    hue = 'Order',
    palette=['#F8B195', '#6C5B7B', ],
    capsize = 0.2,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5,
    ax=ax
)
sns.stripplot(
    x = 'hue',
    y = 'Rate',
    data = SubData,
    hue = 'Order',
    palette=['#F8B195', '#6C5B7B', ],
    edgecolor='black',
    size=3,
    linewidth=0.1,
    ax=ax,
    dodge=True,
    zorder = 1
)
ax.set_ylim([0, 1.5])
ax.set_yticks(np.linspace(0, 1.5, 7))
plt.savefig(join(loc, 'Field formation.png'), dpi = 600)
plt.savefig(join(loc, 'Field formation.svg'), dpi = 600)
plt.close()


# Field Retention
idx = np.where(Data['Category'] == 3)[0]
SubData = SubDict(Data, Data.keys(), idx)

print("Field Retention:")
for hue in np.unique(SubData['hue']):
    prev_idx = np.where((SubData['Order'] == 'Prev')&(SubData['hue'] == hue))[0]    
    next_idx = np.where((SubData['Order'] == 'Next')&(SubData['hue'] == hue))[0]
    print("   ", hue, "   ", ttest_rel(SubData['Rate'][prev_idx], SubData['Rate'][next_idx]))
print()

fig = plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
sns.barplot(
    x = 'hue',
    y = 'Rate',
    data = SubData,
    hue = 'Order',
    palette=['#F8B195', '#6C5B7B', ],
    capsize = 0.2,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5,
    ax=ax
)
sns.stripplot(
    x = 'hue',
    y = 'Rate',
    data = SubData,
    hue = 'Order',
    palette=['#F8B195', '#6C5B7B', ],
    edgecolor='black',
    size=3,
    linewidth=0.1,
    ax=ax,
    dodge=True,
    zorder = 1
)
ax.set_ylim(0, 1.8)
ax.set_yticks(np.linspace(0, 1.8, 10))
plt.savefig(join(loc, 'Field retention.png'), dpi = 600)
plt.savefig(join(loc, 'Field retention.svg'), dpi = 600)
plt.close()