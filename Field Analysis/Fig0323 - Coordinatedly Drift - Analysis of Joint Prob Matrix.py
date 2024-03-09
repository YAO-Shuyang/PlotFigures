from mylib.statistic_test import *
from mylib.field.field_tracker import indept_test_for_evolution_events, compute_joint_probability_matrix

code_id = "0323 - Coordinatedly Drift - Analysis of Joint Prob Matrix"
loc = join(figpath, code_id)
mkdir(loc)


if __name__ == '__main__':
    from tqdm import tqdm
"""
# Plot examples for all paradigms and mice.
for i in range(len(f_CellReg_modi)):
    if f_CellReg_modi['Type'][i] != 'Real':
        continue
    
    if f_CellReg_modi['paradigm'][i] == 'CrossMaze':
        print(f_CellReg_modi['Trace File'][i])
        with open(f_CellReg_modi['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
        
        session, mat = compute_joint_probability_matrix(
            trace['field_reg'],
            trace['field_ids']
        )
        
        n_sessions = len(session)
        fig, axes = plt.subplots(ncols=6, nrows=int((n_sessions-1)//6 + 1), figsize=(3*6, 2*int((n_sessions-1)//6 + 1)))
        sup = max(np.abs(np.max(mat)), np.abs(np.min(mat)))
        for j in tqdm(range(len(session))):
            ax = Clear_Axes(axes[j//6, j%6])
            ax.set_title(session[j])
            sns.heatmap(
                mat[j, :, :],
                vmax=sup,
                vmin=-sup,
                cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True),
                ax=ax
            )
            ax.set_aspect("equal")
            ax = Clear_Axes(ax)
            # colorbar
        plt.savefig(join(loc, f"Example {i+1} {f_CellReg_modi['Type'][i]} {f_CellReg_modi['paradigm'][i]} Maze {f_CellReg_modi['maze_type'][i]} {f_CellReg_modi['MiceID'][i]}.png"), dpi = 600)
        plt.savefig(join(loc, f"Example {i+1} {f_CellReg_modi['Type'][i]} {f_CellReg_modi['paradigm'][i]} Maze {f_CellReg_modi['maze_type'][i]} {f_CellReg_modi['MiceID'][i]}.svg"), dpi = 600)
        plt.close()
        
    else:
        print(f_CellReg_modi['Trace File'][i])
        with open(f_CellReg_modi['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
        
        session, mat = compute_joint_probability_matrix(
            trace['cis']['field_reg'],
            trace['cis']['field_ids']
        )
        
        n_sessions = len(session)
        fig, axes = plt.subplots(ncols=6, nrows=int((n_sessions-1)//6 + 1), figsize=(3*6, 2*int((n_sessions-1)//6 + 1)))
        sup = max(np.abs(np.max(mat)), np.abs(np.min(mat)))
        for j in tqdm(range(len(session))):
            ax = Clear_Axes(axes[j//6, j%6]) if n_sessions > 6 else Clear_Axes(axes[j])
            ax.set_title(session[j])
            sns.heatmap(
                mat[j, :, :],
                vmax=sup,
                vmin=-sup,
                cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True),
                ax=ax
            )
            ax.set_aspect("equal")
            ax = Clear_Axes(ax)
            # colorbar
        plt.savefig(join(loc, f"[{trace['paradigm'] + ' cis'}] Example {i+1} {f_CellReg_modi['Type'][i]} {f_CellReg_modi['paradigm'][i]} Maze {f_CellReg_modi['maze_type'][i]} {f_CellReg_modi['MiceID'][i]}.png"), dpi = 600)
        plt.savefig(join(loc, f"[{trace['paradigm'] + ' cis'}] Example {i+1} {f_CellReg_modi['Type'][i]} {f_CellReg_modi['paradigm'][i]} Maze {f_CellReg_modi['maze_type'][i]} {f_CellReg_modi['MiceID'][i]}.svg"), dpi = 600)
        plt.close()
        
        session, mat = compute_joint_probability_matrix(
            trace['trs']['field_reg'],
            trace['trs']['field_ids']
        )
        
        n_sessions = len(session)
        fig, axes = plt.subplots(ncols=6, nrows=int((n_sessions-1)//6 + 1), figsize=(3*6, 2*int((n_sessions-1)//6 + 1)))
        sup = max(np.abs(np.max(mat)), np.abs(np.min(mat)))
        for j in tqdm(range(len(session))):
            ax = Clear_Axes(axes[j//6, j%6]) if n_sessions > 6 else Clear_Axes(axes[j])
            ax.set_title(session[j])
            sns.heatmap(
                mat[j, :, :],
                vmax=sup,
                vmin=-sup,
                cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True),
                ax=ax
            )
            ax.set_aspect("equal")
            ax = Clear_Axes(ax)
            # colorbar
        plt.savefig(join(loc, f"[{trace['paradigm'] + ' trs'}] Example {i+1} {f_CellReg_modi['Type'][i]} {f_CellReg_modi['paradigm'][i]} Maze {f_CellReg_modi['maze_type'][i]} {f_CellReg_modi['MiceID'][i]}.png"), dpi = 600)
        plt.savefig(join(loc, f"[{trace['paradigm'] + ' trs'}] Example {i+1} {f_CellReg_modi['Type'][i]} {f_CellReg_modi['paradigm'][i]} Maze {f_CellReg_modi['maze_type'][i]} {f_CellReg_modi['MiceID'][i]}.svg"), dpi = 600)
        plt.close()
        
"""

idx = np.where(f_CellReg_modi['Type'] == 'Real')[0]
if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(
        variable_names = ['Training Session', 'delta-P', 'Dimension', 'Axis', 'Pair Type',
                          'Paradigm', 'X'],
        f = f_CellReg_modi, f_member=['Type'],
        function = CoordinatedDrift_Interface, 
        file_name = code_id, file_idx=idx,
        behavior_paradigm = 'CrossMaze'
    )