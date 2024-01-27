from mylib.statistic_test import *

code_id = '0316 - Estimate Independent evolution'
loc = join(figpath, code_id)
mkdir(loc)

def analyze_transitions(binary_matrix, is_shuffle: bool = False, transition_num_thre: int = 3):
    """
    Analyze the 0-1 and 1-0 transitions in each column of a binary matrix.

    Args:
    binary_matrix (list of lists): A binary matrix represented as a list of lists.
    
    Returns:
    dict: A dictionary with keys as column indices and values as lists containing the transitions.
    """
    import numpy as np

    # Convert the list of lists to a numpy array for easier column-wise analysis
    matrix = np.array(binary_matrix)
    
    # Get the number of rows and columns
    rows, cols = matrix.shape[0], matrix.shape[1]
    
    # Initialize the transition matrix
    transition_matrix = np.zeros((rows-1, cols))
    
    # Analyze transitions for each column
    for col in range(cols):
        # Iterate over each row, except the last one, to check the transition
        for row in range(rows - 1):
            if matrix[row, col] == 0 and matrix[row + 1, col] == 1:
                transition_matrix[row, col] = 1
            elif matrix[row, col] == 1 and matrix[row + 1, col] == 0:
                transition_matrix[row, col] = 1

    idx = np.where(np.sum(transition_matrix, axis=0) >= transition_num_thre)[0]
    if len(idx) <= 2:
        return np.array([])
    
    if is_shuffle: 
        for i in range(cols):
            np.random.shuffle(transition_matrix[:, i])

    correlation_matrix = np.corrcoef(transition_matrix[:, idx], rowvar=False)
    
    return np.array(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)].tolist())

def analysis_field_reg_correlation(binary_matrix: np.ndarray, is_shuffle: bool = False):
    idx = np.where(np.sum(binary_matrix, axis=0) >= 3)[0]
    if len(idx) <= 2:
        return np.array([])

    binary_matrix = binary_matrix[:, idx]
    if is_shuffle: 
        for i in range(binary_matrix.shape[1]):
            np.random.shuffle(binary_matrix[:, i])
    
    correlation_matrix = np.corrcoef(binary_matrix, rowvar=False)
    return np.array(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)].tolist())

def analysis_all_fields(trace, distance_thre: int = 4, transition_num_thre: int = 3):
    field_reg = cp.deepcopy(trace['field_reg'])
    field_info = cp.deepcopy(trace['field_info'])
    
    mean_corr = np.array([])
    shuf_corr = np.array([])
    perm_corr = np.array([])
    
    is_detect = np.where(np.isnan(field_reg), 0, 1)
    
    for i in tqdm(range(field_reg.shape[0]-distance_thre)):
        for j in range(i+distance_thre,field_reg.shape[0]):
            idx = np.array([])
            if i > 0 and j < field_reg.shape[0]-1:
                idx = np.where((is_detect[i-1, :] == 0) & (is_detect[j+1, :] == 0) & (np.sum(is_detect[i:j+1, :], axis=0) == j-i+1))[0]
            elif i == 0 and j < field_reg.shape[0]-1:
                idx = np.where((is_detect[j+1, :] == 0) & (np.sum(is_detect[i:j+1, :], axis=0) == j-i+1))[0]
            elif j == field_reg.shape[0]-1 and i > 0:
                idx = np.where((is_detect[i-1, :] == 0) & (np.sum(is_detect[i:j+1, :], axis=0) == j-i+1))[0]
            else:
                idx = np.where((np.sum(is_detect, axis=0) == j-i+1))[0]
            
            reg = field_reg[i:j+1, :]
            reg = reg[:, idx]
            info = field_info[i:j+1, :, 0]
            info = info[:, idx]
            
            unique_cell = np.unique(info[0, :])
            
            for k in range(len(unique_cell)):
                idx = np.where(info[0, :] == unique_cell[k])[0]
                
                if len(idx) > 2:
                    mean_corr = np.concatenate([mean_corr, analysis_field_reg_correlation(reg[:, idx])])
                    for d in range(100):
                        shuf_corr = np.concatenate([shuf_corr, analysis_field_reg_correlation(reg[:, idx], is_shuffle=True)])
                        perm_corr = np.concatenate([perm_corr, analysis_field_reg_correlation(reg[:, np.random.choice(np.arange(reg.shape[1]), size = len(idx), replace=False)])])
                    """
                    mean_corr = np.concatenate([mean_corr, analyze_transitions(reg[:, idx], transition_num_thre=transition_num_thre)])
                    for d in range(100):
                        shuf_corr = np.concatenate([shuf_corr, analyze_transitions(reg[:, idx], is_shuffle=True, transition_num_thre=transition_num_thre)])
                        perm_corr = np.concatenate([perm_corr, analyze_transitions(reg[:, np.random.choice(np.arange(reg.shape[1]), size = len(idx), replace=False)],
                                                                                               transition_num_thre=transition_num_thre)])
                    """
    return np.nanmean(mean_corr), np.nanmean(shuf_corr), np.nanmean(perm_corr)
        

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = {"MiceID": np.array([], np.int64), "Maze Type": np.array([]), 
            "Mean Corr": np.array([], np.int64), "Shuffle Corr": np.array([], np.int64), "Perm Corr": np.array([], np.float64)}
    
    for i in range(len(f_CellReg_day)):
        if f_CellReg_day['include'][i] == 0 or f_CellReg_day['maze_type'][i] == 0:
            continue
        
        with open(f_CellReg_day['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
            
        print(f_CellReg_day['Trace File'][i])
            
        mean_corr, shuf_corr, perm_corr = analysis_all_fields(trace)
        
        Data['MiceID'] = np.append(Data['MiceID'], f_CellReg_day['MiceID'][i])
        Data['Maze Type'] = np.append(Data['Maze Type'], "Maze "+str(int(f_CellReg_day['maze_type'][i])))
        Data['Mean Corr'] = np.append(Data['Mean Corr'], mean_corr)
        Data['Shuffle Corr'] = np.append(Data['Shuffle Corr'], shuf_corr)
        Data['Perm Corr'] = np.append(Data['Perm Corr'], perm_corr)
        
        del trace
        gc.collect()
        
    with open(join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
        
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)

print(ttest_rel(Data['Mean Corr'][Data['Maze Type'] == 'Maze 1'], Data['Shuffle Corr'][Data['Maze Type'] == 'Maze 1']))
print(ttest_rel(Data['Mean Corr'][Data['Maze Type'] == 'Maze 1'], Data['Perm Corr'][Data['Maze Type'] == 'Maze 1']))
print(ttest_rel(Data['Mean Corr'][Data['Maze Type'] == 'Maze 2'], Data['Shuffle Corr'][Data['Maze Type'] == 'Maze 2']))
print(ttest_rel(Data['Mean Corr'][Data['Maze Type'] == 'Maze 2'], Data['Perm Corr'][Data['Maze Type'] == 'Maze 2']))
fig = plt.figure(figsize=(1.5,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.boxplot(
    x = np.concatenate([Data['Maze Type'], Data['Maze Type'], Data['Maze Type']]),
    y = np.concatenate([Data['Mean Corr'], Data['Perm Corr'], Data['Shuffle Corr']]),
    hue = np.concatenate([np.repeat("Mean", len(Data['Mean Corr'])), 
                        np.repeat("Shuffle", len(Data['Shuffle Corr'])),
                        np.repeat("Perm", len(Data['Perm Corr']))]),
    ax=ax,
    linewidth=0.5,
    width = 0.8
)


plt.savefig(join(loc, 'corr.png'), dpi=600)
plt.savefig(join(loc, 'corr.svg'), dpi=600)
plt.close()