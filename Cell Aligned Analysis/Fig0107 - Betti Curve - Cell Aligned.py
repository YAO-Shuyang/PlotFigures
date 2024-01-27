from mylib.statistic_test import *
from mylib.betti_curves import betti_curves

code_id = '0107 - Betti Curve'
p = os.path.join(figpath, 'Cell Aligned', code_id)
mkdir(p)
data_loc = os.path.join(figdata, code_id)
mkdir(data_loc)

def AlignedRateMapForBetti(row:int, map1:int, map2:int, f_trace:pd.DataFrame = f1, f:pd.DataFrame = f_CellReg, rate_type:str = 'smooth_map_all', **kwargs):
    '''
    Parameters
    ----------
    row: int, required
        To select the line in f that you want to choose to analyze.
    map1: int, required
        Only 0,1,2,3 are valid value. 0 and 3 represent open field, while 1 and 2 represent Maze 1 and Maze 2 respectively.
    map2: int, required
        Only 0,1,2,3 are valid value. 0 and 3 represent open field, while 1 and 2 represent Maze 1 and Maze 2 respectively.
        Note that map2 should be bigger than map1
    f_trace: pd.DataFrame
        default value is f1 (see in mylib.statistic_test)
    f: pd.DataFrame
        default value is f_CellReg that saves all of the directories of cellRegistered.mat
    rate_type:str, optional
        There's three types of rate map are saved in traces:
            1) 'clear_map_all', 48x48 rate map without smoothing. NAN values are cleared.
            2) 'old_map_clear', 12x12 rate map without smoothing. NAN values are cleared.
            3) 'smooth_map_all', 48x48 rate map smoothed by A Gaussian kernal. NAN values are cleared.
        Default value is 'smooth_map_all'. Only these 3 values listed above are valid data. Ohter inputs will raise an error.

    ** kwargs

    Returns
    -------
    Two np.ndarray matrix with same shape [rate map]
    
    Author
    ------
    YAO Shuyang

    Date
    ----
    Jan 30th, 2023
    '''

    assert row < len(f) # row should not be bigger than the length of f, or it will overflow.
    assert map1 < map2 # map 1 should be bigger than map 2
    ValueErrorCheck(map1, [0,1,2])
    ValueErrorCheck(map2, [1,2,3])
    ValueErrorCheck(rate_type, ['smooth_map_all', 'clear_map_all', 'old_map_all'])

    # Read and Sort Index Map
    print("Step 1 - Read And Sort Index Map")
    if os.path.exists(f['Cell Reg Path'][row]):
        index_map = Read_and_Sort_IndexMap(path = f_CellReg['Cell Reg Path'][row], occur_num = 2, align_type = 'cross_session')
    else:
        print(f_CellReg['Cell Reg Path'][row], 'is not exist!')
        return None, None
    
    # Select Cell Pairs that Both exist in index_map in map1 and map2
    is_cell_detected = np.where(index_map == 0, 0, 1)
    cellpair = np.where(np.nansum(is_cell_detected[[map1,map2],:], axis = 0) == 2)[0]
    index_map = index_map[:, cellpair]
    index_map = index_map.astype(np.int64)

    # Get Trace File Set
    print("Step 2 - Get Trace File Set")
    idx = np.where((f_trace['MiceID'] == f['MiceID'][row])&(f_trace['date'] == f['date'][row]))[0]
    trace_set = TraceFileSet(idx = idx, file = f_trace, Behavior_Paradigm = 'Cross Maze')
    KeyWordErrorCheck(trace_set[0], __file__, keys = [rate_type])

    return trace_set[map1][rate_type][index_map[map1,:]-1, :], trace_set[map2][rate_type][index_map[map2,:]-1, :]


if __name__ == '__main__':
    
    for i in range(len(f_CellReg)):
        print(i, '------------------------------------------------------------------------')        
        mat_size = 200
        loc = os.path.join(p, str(int(f_CellReg['MiceID'][i]))+' - '+str(int(f_CellReg['date'][i])))
        mkdir(loc)
        mkdir(os.path.join(figdata, str(int(f_CellReg['MiceID'][i]))+' - '+str(int(f_CellReg['date'][i]))))

        print('  1. Maze1 - Maze2')
        A, B = AlignedRateMapForBetti(i, 1, 2)
        Betti = betti_curves(A[0:mat_size], B[0:mat_size], corr_type = 'pearson')
        Betti.plot_betti_curve(save_loc = os.path.join(loc, 'Maze1-Maze2'))

        with open(os.path.join(data_loc, 'Maze1-Maze2 - MatSize['+str(mat_size)+'].pkl'), 'wb') as f:
            pickle.dump(Betti, f)

        print('  2. OpenF - Maze1')
        A, B = AlignedRateMapForBetti(i, 0, 1)
        Betti = betti_curves(A[0:mat_size], B[0:mat_size], corr_type = 'pearson')
        Betti.plot_betti_curve(save_loc = os.path.join(loc, 'OpenF-Maze1'))

        with open(os.path.join(data_loc, 'OpenF-Maze1 - MatSize['+str(mat_size)+'].pkl'), 'wb') as f:
            pickle.dump(Betti, f)

        print('  3. OpenF - Maze2')
        A, B = AlignedRateMapForBetti(i, 0, 2)
        Betti = betti_curves(A[0:mat_size], B[0:mat_size], corr_type = 'pearson')
        Betti.plot_betti_curve(save_loc = os.path.join(loc, 'OpenF-Maze2'))

        with open(os.path.join(data_loc, 'OpenF-Maze2 - MatSize['+str(mat_size)+'].pkl'), 'wb') as f:
            pickle.dump(Betti, f)

        print('  4. OpenF - OpenF')
        A, B = AlignedRateMapForBetti(i, 0, 1)
        Betti = betti_curves(A[0:mat_size], B[0:mat_size], corr_type = 'pearson')
        Betti.plot_betti_curve(save_loc = os.path.join(loc, 'OpenF-OpenF'))

        with open(os.path.join(data_loc, 'OpenF-OpenF - MatSize['+str(mat_size)+'].pkl'), 'wb') as f:
            pickle.dump(Betti, f)
        print("Done. ---------------------------------------------------", end = '\n\n\n')



import ripser
# Analyze time-consuming

def plot_time_consuming(time_consuming:np.ndarray, save_loc:str = None):
    mat_size = np.array([1,2,3,4,5,6,7,8,9,10,12,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,180,200,250,300], dtype = np.int64)

    fig = plt.figure(figsize=(2,3))
    ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
    ax.plot(mat_size, time_consuming)
    ax.set_ylabel("Time Consumption [s]")
    ax.set_xlabel("Size of Matrix")
    ax.set_xticks([0,100,200,300])
    plt.tight_layout()

    if save_loc is None:
        plt.show()
    else:
        plt.savefig(save_loc+'.png', dpi = 1200)
        plt.savefig(save_loc+'.svg', dpi = 1200)
        plt.close()

    

if os.path.exists(os.path.join(data_loc, 'time_consuming.pkl')):
    with open(os.path.join(data_loc, 'time_consuming.pkl'), 'rb') as handle:
        time_consuming = pickle.load(handle)

else:
    mat_size = np.array([1,2,3,4,5,6,7,8,9,10,12,15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,180,200,250,300], dtype = np.int64)
    time_consuming = np.zeros(len(mat_size), dtype = np.float64)
    for i in range(len(mat_size)):
        print(i,'----------------------------------------------')
        s = mat_size[i]
        TestMat = np.ones((s,s))
        TestMat[[np.arange(s), np.arange(s)]] = 0
        t1 = time.time()
        dgm = ripser.ripser(TestMat, maxdim = 3, distance_matrix = True)
        time_consuming[i] = time.time()-t1
        print(end='\n\n')

    with open(os.path.join(data_loc, 'time_consuming.pkl'), 'wb') as f:
        pickle.dump(time_consuming, f)

plot_time_consuming(time_consuming, os.path.join(p, 'Time-consumption vs Size of Matrix'))


    

