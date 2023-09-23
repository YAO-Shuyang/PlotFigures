# Fig0001
# Mice: 11095
# Maze: 1
# 19 example cells
# date: Aug 20th to Aug 30th, 6 days in total
# data ara saved in 0001data.pkl, data is a dict type with 2 members: 
#                                     'index_matrix': numpy.ndarray, shape(19,6), dtype(np.int64)
#                                     'trace_set': list, length = 6, object in data['trace_set'] are dict (trace)
from mylib.statistic_test import *

# ================================================= Data Generation ==================================================================================
mice = 11095
maze_type = 1
row = 21

code_id = '0001 - Sample cells'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'data.pkl')) == False:
    idx = np.where((f1['MiceID'] == mice)&(f1['date'] >= 20220820)&(f1['date'] <= 20220830)&(f1['session'] == maze_type+1))[0]
    print(idx)
    file = pd.read_excel(r'E:\Data\Cross_maze\cell_to_index_map_correction.xlsx', sheet_name = '11095-Maze1')
    dateset = ['20220820','20220822','20220824','20220826','20220828','20220830']
    traceset = TraceFileSet(idx = idx, tp = r"E:\Data\Cross_maze")

    index_matrix = np.zeros((row,len(idx)), dtype = np.int64)

    for d in tqdm(range(len(idx))):
        index_matrix[:,d] = file[dateset[d]][0:row]

    data = {'index_matrix':index_matrix,
            'trace_set':traceset}
    with open(os.path.join(figdata, code_id+'data.pkl'), 'wb') as f:
        pickle.dump(data,f)
else:
    with open(os.path.join(figdata, code_id+'data.pkl'), 'rb') as handle:
        data = pickle.load(handle)

# ============================================================================================================================================================

# ================================================================== Plot Figure =============================================================================

def cross_day_correlation(trace_set, IndexMatrix):
    corr_mat = np.zeros((IndexMatrix.shape[0], IndexMatrix.shape[1], IndexMatrix.shape[1]), dtype = np.float64)
    for n in tqdm(range(IndexMatrix.shape[0])):
        for i in range(IndexMatrix.shape[1]):
            for j in range(IndexMatrix.shape[1]):
                x, y = int(IndexMatrix[n, i]), int(IndexMatrix[n, j])
                if x != 0 and y != 0:
                    corr_mat[n, i, j] = corr_mat[n, j, i] = pearsonr(trace_set[i]['smooth_map_all'][x-1],
                                                                     trace_set[j]['smooth_map_all'][y-1])[0]
                else:
                    corr_mat[n, i, j] = corr_mat[n, j, i] = np.nan
    return corr_mat

def DrawFigure0001(trace_set = [], IndexMatrix = np.zeros((2,6), dtype = np.int64), maze_type = 1, 
                   save_loc = None, file_name = None, inverse = True):
    if len(trace_set) != IndexMatrix.shape[1]:
        print('Warning! The length of trace_set is not equal to the length of IndexMatrix!',len(trace_set),len(IndexMatrix))
        return
    
    corr_mat = cross_day_correlation(trace_set, IndexMatrix)
    
    fig, axes = plt.subplots(ncols = IndexMatrix.shape[1]+1, nrows = IndexMatrix.shape[0], 
                             figsize = (IndexMatrix.shape[1]*3+3,IndexMatrix.shape[0]*2))
    for n in tqdm(range(IndexMatrix.shape[0])):
        for i in range(len(trace_set)):
            if inverse == True:
                c = IndexMatrix.shape[1]-1 - i
            else:
                c = i
                
            if IndexMatrix[n,i] != 0:
                ax = Clear_Axes(axes[n,c])
                rate_map = trace_set[i]['smooth_map_all'][int(IndexMatrix[n,i]) - 1]
                im = ax.imshow(np.reshape(rate_map,[48,48]), cmap = 'jet')
                cbar = plt.colorbar(im, ax = ax)
                cbar.set_ticks(ColorBarsTicks(peak_rate=round(np.nanmax(rate_map),2), is_auto=True, tick_number=2))
                cbar.outline.set_visible(False)
                color = 'red' if trace_set[i]['is_placecell_isi'][int(IndexMatrix[n,i]) - 1] == 1 else 'black'
                ax.set_title('SI = '+str(round(trace_set[i]['SI_all'][int(IndexMatrix[n,i]) - 1], 3))+', Cell '+str(int(IndexMatrix[n,i])), 
                         color = color)
                ax.axis([-1,48,-1,48])
                if maze_type in [1,2]:
                    DrawMazeProfile(maze_type = maze_type, axes = ax, nx = 48, linewidth = 0.5)
                ax.invert_yaxis()
                
            else:
                ax = Clear_Axes(axes[n,c])
    
        ax = Clear_Axes(axes[n, IndexMatrix.shape[1]])
        Min = np.nanmin(corr_mat[n, :, :]) if np.nanmin(corr_mat[n, :, :]) < 0 else 0
        im = ax.imshow(corr_mat[n, :, :], vmin = Min, 
                       vmax = np.nanmax(corr_mat[n, :, :]))
        cbar = plt.colorbar(im, ax = ax)
        cbar.set_ticks(np.linspace(0, np.nanmax(corr_mat[n, :, :]), 6))
        cbar.outline.set_visible(False)
                
    plt.savefig(os.path.join(save_loc, file_name+'.png'), dpi = 600, bbox_inches='tight',pad_inches=0)
    plt.savefig(os.path.join(save_loc, file_name+'.svg'), dpi = 600, bbox_inches='tight',pad_inches=0)
    #plt.savefig(os.path.join(loc,'0001.pdf'), dpi = 600, bbox_inches='tight',pad_inches=0)
    #plt.savefig(os.path.join(loc,'0001.svg'), dpi = 600, bbox_inches='tight',pad_inches=0)
    plt.close()

"""
DrawFigure0001(trace_set = data['trace_set'], IndexMatrix = data['index_matrix'], maze_type = data['trace_set'][0]['maze_type'],
               save_loc = loc, file_name = 'Maze 1 Sample Cells', inverse = False)
"""

num = 0
for i in tqdm(range(len(f1))):
    if exists(f1['Trace File'][i]):
        with open(f1['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
            
        num += trace['n_neuron']
        
print(num)