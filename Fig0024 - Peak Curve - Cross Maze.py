from mylib.statistic_test import *

code_id = '0024'

def plot_0024(mice = 11095, maze_type = 1, code_id = '0024'):
    if maze_type == 0:
        return
    elif maze_type == 1:
        idx = np.where((f1['MiceID'] == mice)&(f1['date'] >= 20220810)&(f1['date'] <= 20220830)&(f1['session'] == maze_type+1))[0]
    elif maze_type == 2:
        idx = np.where((f1['MiceID'] == mice)&(f1['date'] >= 20220813)&(f1['date'] <= 20220830)&(f1['session'] == maze_type+1))[0]

    trace_set = TraceFileSet(idx = idx)

    fig, axes = plt.subplots(ncols = 6, nrows = 2, figsize = (6*4, 2*6))
    for r in [0,1]:
        for l in range(6):
            if maze_type == 2 and r*6 + l >= 9:
                axes[r,l] = Clear_Axes(axes=axes[r,l])
                continue
            trace = trace_set[r*6+l]
            old_map_all = cp.deepcopy(trace['old_map_clear'])
            old_map_all = np.delete(old_map_all, trace['SilentNeuron'], axis = 0)
            x_order = np.concatenate([CorrectPath_maze_1-1, IncorrectPath_maze_1-1]) if trace['maze_type'] == 1 else np.concatenate([CorrectPath_maze_2-1, IncorrectPath_maze_2-1])
            old_map_all = old_map_all[:, x_order] # sort x order
            old_map_all = sortmap(old_map_all)
            old_map_all = Norm_ratemap(old_map_all)

            axes[r,l] = Clear_Axes(axes = axes[r,l], xticks = np.linspace(0,144,9), yticks = ColorBarsTicks(peak_rate = old_map_all.shape[0], intervals = 50))
            axes[r,l].imshow(old_map_all, aspect = 'auto')
            axes[r,l].set_title(trace['date']+', Mouse #'+str(mice))
            axes[r,l].invert_yaxis()
            axes[r,l].axis([0,144,0,old_map_all.shape[0]])

    plt.savefig(os.path.join(figpath, code_id+'-Peak Curve Without Aligning-Cross Maze-Maze '+str(maze_type)+'-'+str(mice)+'.png'), dpi = 600)
    plt.savefig(os.path.join(figpath, code_id+'-Peak Curve Without Aligning-Cross Maze-Maze '+str(maze_type)+'-'+str(mice)+'.svg'), dpi = 600)
    plt.close()


# Fig 0024-1, Maze 1, 11095
plot_0024(mice = 11095, maze_type = 1, code_id='0024-1')
plot_0024(mice = 11095, maze_type = 2, code_id='0024-2')



        