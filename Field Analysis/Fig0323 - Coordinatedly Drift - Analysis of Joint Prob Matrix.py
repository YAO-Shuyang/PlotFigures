from mylib.statistic_test import *
from mylib.field.field_tracker import indept_test_for_evolution_events, compute_joint_probability_matrix

code_id = "0323 - Coordinatedly Drift - Analysis of Joint Prob Matrix"
loc = join(figpath, code_id)
mkdir(loc)


if __name__ == '__main__':
    from tqdm import tqdm

for i in range(len(f_CellReg_modi)):
    if f_CellReg_modi['paradigm'][i] == 'CrossMaze' and f_CellReg_modi['Type'][i] == 'Real':
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
            ax = Clear_Axes(axes[j])
            # colorbar
        plt.savefig(join(loc, f"Example {f_CellReg_modi['Type'][i]} {f_CellReg_modi['paradigm'][i]} Maze {f_CellReg_modi['maze_type'][i]} {f_CellReg_modi['MiceID'][i]}.png"), dpi = 600)
        plt.savefig(join(loc, f"Example {f_CellReg_modi['Type'][i]} {f_CellReg_modi['paradigm'][i]} Maze {f_CellReg_modi['maze_type'][i]} {f_CellReg_modi['MiceID'][i]}.svg"), dpi = 600)
        plt.close()