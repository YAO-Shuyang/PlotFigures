
from mylib.statistic_test import *

# Fig0002
# Get y_order for plotting figure 0002(to synchronously rearrange the peakcurve map)
def Get_Y_Order(rate_map_all = None, nx = 12):
    maxidx = np.nanargmax(rate_map_all, axis = 1)
    y_order = np.array([], dtype = np.int64)
    for i in range(nx**2,0,-1):
        y_order = np.concatenate([y_order,np.where(maxidx == i-1)[0]])
    return y_order

# This function should not be used to plot figures of other behavioral paradigm and should be used only for cross day figures.
# Sample for maze 1, 11095 and maze 2, 11095
def AlignedPeakCurve(trace_set = None, cellreg_path = None, save_loc = None, path_type = 'c'):
    index_map = Read_and_Sort_IndexMap(path = cellreg_path, align_type = 'cross_day', occur_num = 3)
    
    # check if trace_set and cellreg index_map have the same length.
    if index_map.shape[0] != len(trace_set):
        print("ShapeLengthError! Index_map and trace_set have different length. Abort plotting!")
        return
    
    #Generate x_order
    maze_type = trace_set[0]['maze_type']
    if path_type == 'c':
        CorrectPath = CorrectPath_maze_1 if maze_type == 1 else CorrectPath_maze_2
        length = len(CorrectPath)
        x_order = CorrectPath
    elif path_type == 'i':
        IncorrectPath = IncorrectPath_maze_1 if maze_type == 1 else IncorrectPath_maze_2
        length = len(IncorrectPath)
        x_order = IncorrectPath        
    else:
        x_order = np.concatenate([CorrectPath_maze_1,IncorrectPath_maze_1]) if maze_type == 1 else np.concatenate([CorrectPath_maze_2, IncorrectPath_maze_2])
    
    # ordered by each day
    for d in range(len(trace_set)):
        print('Reference Day '+str(d+1))
        # sort the index_map with the order of reference day.
        # Generate old map sorted and fixed by index_map
        idx = np.where(index_map[d,:] != 0)[0]
        max_idx = np.nanmax(idx)
        # Only keep correct path firing rate map
        if path_type == 'c':
            old_map_ref = np.zeros((index_map.shape[1], length), dtype = np.float64)
            old_map_clear = trace_set[d]['old_map_clear'][:, CorrectPath-1]
            old_map_ref[idx,:] = old_map_clear[index_map[d,idx]-1, :]
            old_map_ref = Norm_ratemap(old_map_ref)
        # Only keep incorrect path firing rate map
        elif path_type == 'i':
            old_map_ref = np.zeros((index_map.shape[1], length), dtype = np.float64)
            old_map_clear = trace_set[d]['old_map_clear'][:, IncorrectPath-1]
            old_map_ref[idx,:] = old_map_clear[index_map[d,idx]-1, :]
            old_map_ref = Norm_ratemap(old_map_ref)
        # Keep both correct path and incorrect path rate map
        else:
            old_map_ref = np.zeros((index_map.shape[1], 144), dtype = np.float64)

            old_map_ref[idx,:] = trace_set[d]['old_map_clear'][index_map[d,idx]-1, :]
            old_map_ref = Norm_ratemap(old_map_ref[:, x_order-1])

        # Get y order according to yaxis
        y_order = Get_Y_Order(rate_map_all = old_map_ref)
        PlotAlignedPeakCurve(trace_set = trace_set, index_map = index_map, save_loc = save_loc, Reference_day = d+1, 
                             maze_type = maze_type, x_order = x_order, y_order = y_order, path_type = path_type, max_idx = max_idx)
    
def PlotAlignedPeakCurve(trace_set = None, index_map = None, save_loc = None, Reference_day = 1, maze_type = 1, x_order = None,
                         y_order = None, path_type = 'c', max_idx = None):
    nrow = 1
    ncol = len(trace_set)
    CorrectPath = CorrectPath_maze_1 if maze_type == 1 else CorrectPath_maze_2
    IncorrectPath = IncorrectPath_maze_1 if maze_type == 1 else IncorrectPath_maze_2

    # Define a figure with 6 axes.
    fig, ax = plt.subplots(nrows = nrow, ncols = ncol, figsize = (ncol * 2, nrow * 4))
    for d in tqdm(range(len(trace_set))):
        # Generate old map sorted and fixed by index_map
        idx = np.where(index_map[d,:] != 0)[0]
        # Only keep correct path firing rate map
        if path_type == 'c':
            length = len(CorrectPath)
            old_map_all = np.zeros((index_map.shape[1], length), dtype = np.float64)
            old_map_clear = trace_set[d]['old_map_clear'][:, CorrectPath-1]
            old_map_all[idx,:] = old_map_clear[index_map[d,idx]-1, :]
            old_map_all = Norm_ratemap(old_map_all)
            old_map_all = cp.deepcopy(old_map_all[y_order,:])
            
            # plotting peak curve
            ax[d] = Clear_Axes(ax[d], xticks = ColorBarsTicks(peak_rate = length, intervals = 20), yticks = ColorBarsTicks(peak_rate = max_idx+1, intervals = 50))
            ax[d].axis([-0.5,length, -0.5,max_idx+1])
        # Only keep incorrect path firing rate map
        elif path_type == 'i':
            length = len(IncorrectPath)
            old_map_all = np.zeros((index_map.shape[1], length), dtype = np.float64)
            old_map_clear = trace_set[d]['old_map_clear'][:, IncorrectPath-1]
            old_map_all[idx,:] = old_map_clear[index_map[d,idx]-1, :]
            old_map_all = Norm_ratemap(old_map_all)
            old_map_all = cp.deepcopy(old_map_all[y_order,:])
            
            # plotting peak curve
            ax[d] = Clear_Axes(ax[d], xticks = ColorBarsTicks(peak_rate = length, intervals = 10), yticks = ColorBarsTicks(peak_rate = max_idx+1, intervals = 50))
            ax[d].axis([-0.5,length, -0.5,max_idx+1])
        # Keep both correct path and incorrect path rate map
        else:
            old_map_all = np.zeros((index_map.shape[1], 144), dtype = np.float64)
            old_map_all[idx,:] = trace_set[d]['old_map_clear'][index_map[d,idx]-1, :]
            old_map_all = Norm_ratemap(old_map_all[:, x_order-1])
            old_map_all = cp.deepcopy(old_map_all[y_order,:])

            # plotting peak curve
            ax[d] = Clear_Axes(ax[d], xticks=np.linspace(0,144,9), yticks = ColorBarsTicks(peak_rate = max_idx+1, intervals = 50))
            ax[d].axvline(length-0.5)
            ax[d].axis([-0.5,144, -0.5,max_idx+1])

        im = ax[d].imshow(old_map_all[0:max_idx+1,:], aspect = 'auto', cmap = 'hot')
        ax[d].set_xlabel('Linearized maze ID')
        ax[d].set_title("Training Day\n"+str(trace_set[d]['date']))

    ax[0].set_ylabel('Neuron ID')
    if os.path.exists(save_loc) == False:
        mkdir(save_loc)
    plt.savefig(os.path.join(save_loc, 'Aligned Peak Curve - Ref day '+str(Reference_day)+' - Maze '+str(maze_type)+'.png'), dpi = 600)
    plt.savefig(os.path.join(save_loc, 'Aligned Peak Curve - Ref day '+str(Reference_day)+' - Maze '+str(maze_type)+'.svg'), dpi = 600)
    plt.close()

code_id = '0002'

mice = 11095

save_loc = r"E:\Data\FinalResults\0002 - Population Align Across Days"
mkdir(save_loc)
# 11095 - maze1
maze_type = 1
idx = np.where((f1['MiceID'] == mice)&(f1['date'] >= 20220820)&(f1['date'] <= 20220830)&(f1['session'] == maze_type+1))[0]
trace_set = TraceFileSet(idx = idx, tp=r"E:\Data\Cross_maze")
AlignedPeakCurve(trace_set = trace_set, cellreg_path = cellReg_95_maze1, save_loc = r'E:\Data\FinalResults\0002 - Population Align Across Days\11095-maze1')

# 11095 - maze2
maze_type = 2
idx = np.where((f1['MiceID'] == mice)&(f1['date'] >= 20220820)&(f1['date'] <= 20220830)&(f1['session'] == maze_type+1))[0]
trace_set = TraceFileSet(idx = idx, tp=r"E:\Data\Cross_maze")
AlignedPeakCurve(trace_set = trace_set, cellreg_path = cellReg_95_maze2, save_loc = r'E:\Data\FinalResults\0002 - Population Align Across Days\11095-maze2')