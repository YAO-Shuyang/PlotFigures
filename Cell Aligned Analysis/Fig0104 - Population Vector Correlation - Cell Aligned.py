from mylib.statistic_test import *

code_id = '0104 - Population Vector Correlation'
p = os.path.join(figpath, 'Cell Aligned', code_id)
mkdir(p)

def PVCMatrixAnalysis(PVC:np.ndarray, map1:int = 1, map2:int = 2, save_loc:str = None, training_day:str = None, pvc_thre:np.float = 0.6, sep1:int = 110, sep2 = 101, **kwargs):
    '''
    Author: YAO Shuyang
    Date: Jan 27th, 2023
    Note: This function is written to analysis the PVC matrix

    Parameters:
    -----------
    - PVC: numpy.ndarray, (144,144)
    - map1: int, default value is 1 and other values are deprecated.
    - map2: int, default value is 2 and other values are deprecated.
    - save_loc: str, the location to save figure.
    - training_day: str, the training day that will be shown on the x label of the violin figure.
    - pvc_thre: float, and should belongs to [0,1]
    
    Returns:
    --------
    - PVCResults: dict contains toughly divided data.
    '''
    assert save_loc is not None and training_day is not None
    assert pvc_thre <= 1 and pvc_thre >= 0
    assert PVC.shape[0] == 144 and PVC.shape[1] == 144
    
    # Generate figure data. A Global view
    # Global view means to toughly divide data into 4 groups and analysis, and it will be influenced by many noise point.
    # So take a deeper look may be good, after this global views.

    # the separate point to divide matrix.
    # sep1 = CorrectPath_maze_1.shape[0]
    # sep2 = CorrectPath_maze_2.shape[0]

    # Cut the slice to fetch the data.
    co_co = PVC[0:sep1, 0:sep2].flatten()
    co_co = np.delete(co_co, np.where(np.isnan(co_co))[0])

    ic_co = PVC[sep1::, 0:sep2].flatten()
    ic_co = np.delete(ic_co, np.where(np.isnan(ic_co))[0])

    co_ic = PVC[0:sep1, sep2::].flatten()
    co_ic = np.delete(co_ic, np.where(np.isnan(co_ic))[0])

    ic_ic = PVC[sep1::, sep2::].flatten()
    ic_ic = np.delete(ic_ic, np.where(np.isnan(ic_ic))[0])

    _, pvalue_12 = scipy.stats.ttest_ind(co_co, co_ic)
    _, pvalue_13 = scipy.stats.ttest_ind(co_co, ic_co)
    _, pvalue_14 = scipy.stats.ttest_ind(co_co, ic_ic)
    _, pvalue_23 = scipy.stats.ttest_ind(co_ic, ic_co)
    _, pvalue_24 = scipy.stats.ttest_ind(co_ic, ic_ic)
    _, pvalue_34 = scipy.stats.ttest_ind(ic_co, ic_ic)

    PVCResults = {'PVC':np.concatenate([co_co,co_ic,ic_co,ic_ic]), 
                  'Corr Type':np.repeat(['co-co', 'co-im', 'im-co', 'im-im'], [co_co.shape[0], co_ic.shape[0], ic_co.shape[0], ic_ic.shape[0]])}
    PVCResults['Training Day'] = np.repeat(training_day, PVCResults['PVC'].shape[0])

    # plot barplot.
    fig = plt.figure(figsize = (3,3))
    ax = Clear_Axes(plt.axes(), close_spines = ['top', 'right'], ifxticks=True, ifyticks=True, **kwargs)
    sns.violinplot(x = 'Corr Type', y = 'PVC', data = PVCResults, width = 0.6, ax = ax)
    #sns.barplot(x = 'Corr Type', y = 'PVC', data = PVCResults, width = 0.6, edgecolor = 'black', errwidth = 1, capsize=0.25, ax = ax, errcolor = 'black', alpha = 0.6, **kwargs)
    #sns.stripplot(x = 'Corr Type', y = 'PVC', data = PVCResults, palette = 'Set2', size = 2, jitter=0.8, ax = ax, legend = False, alpha = 0.6)
    ax.legend(facecolor = 'white', edgecolor = 'white', title = 'co - correct; im - impasse', loc = 'lower left')
    ax.set_yticks(np.linspace(-1,1,11))
    ax.axhline(0, ls = "--", color = 'black')
    ax.set_ylabel('Population Vector Correlation')
    ax.set_xlabel(training_day)
    ax.axis([-0.5,3.5,-1,1])
    plt.xticks([0,1,2,3])

    plot_star(left = np.array([0,0,0,1,1,2])+0.2, right = np.array([1,2,3,2,3,3])-0.2, height = [0.75, 0.83, 0.99, 0.75, 0.91, 0.75], delt_h=[-0.02,-0.02,-0.02,-0.02,-0.02,-0.02], 
              p = [pvalue_12,pvalue_13,pvalue_14,pvalue_23,pvalue_24,pvalue_34], barwidth = 1, fontsize = 8, ax = ax)
    plt.tight_layout()
    plt.savefig(save_loc+' [violinplot].png', dpi = 1200)
    plt.savefig(save_loc+' [violinplot].svg', dpi = 1200)
    plt.close()


    # Analysis spatial bins with high correlation:


    return PVCResults

def plot_CumulativePVC_comparison_figure(map1_area:list|np.ndarray, map2_area:list|np.ndarray, map1:int = 1, map2:int = 2,savefig:str = None, **kwargs):
    '''
    Author: YAO Shuyang
    Date: Jan 27th, 2023
    Note: there's some areas exhibit a robustly high correlattion between map1 and map2. To visualize two area that shared high correlation on two map simultaneously to make it direct.

    Parameter
    ---------
    - map1_area: NDarray[int]
    - map2_area: NDarray[int]
    '''
    assert np.nanmax(map1_area) < 144 and np.nanmax(map2_area) < 144
    assert savefig is not None

    # Generate simulated rate map
    ratemap1 = np.zeros(144, dtype = np.float64)
    ratemap2 = np.zeros(144, dtype = np.float64)
    x_order1, _ = Get_X_Order(1)
    x_order2, _ = Get_X_Order(2)
    ratemap1[x_order1[map1_area]] = 1
    ratemap2[x_order2[map2_area]] = 1

    # plot figure
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize = (8,4))
    ax[0] = Clear_Axes(ax[0])
    ax[1] = Clear_Axes(ax[1])   

    ax[0].imshow(np.reshape(ratemap1,[12,12]), **kwargs)
    DrawMazeProfile(maze_type = map1, axes = ax[0], nx = 12, linewidth=2, color = 'black')
    ax[0].set_title('Maze '+str(map1))

    ax[1].imshow(np.reshape(ratemap2,[12,12]), **kwargs)
    DrawMazeProfile(maze_type = map2, axes = ax[1], nx = 12, linewidth=2, color = 'black')
    ax[1].set_title('Maze '+str(map2)) 
    plt.show()

def plot_pvc_ratemap(f:pd.DataFrame = f_CellReg, row:int = None, map1:int = 1, map2:int = 2, save_loc:str = None, f_trace:pd.DataFrame = f1, map_type:str = 'old_map_clear',
                     cmap = 'summer', barplot_kwgs:dict = {}, **kwargs):
    '''
    Author: YAO Shuyang
    Date: Jan 27th, 2023
    Note: To plot a rate_map for population vector correlation.

    Parameters
    ----------
    - f: <class 'pandas.DataFrame'>, default value is f_CellReg which saves directories of all cross_session cellRegistered.mat file
    - row: int, the row id of the line want to read in f. Row should not be larger than the length of f. If f is default file(f_CellReg), row should not be bigger than 17.
    - map1: int, input a maze you want to choose. Only 0,1,2,3 are valid (0 and 3 represent Open Field 1 and Open Field 2, respectively, while 1 and 2 represent Maze 1/2 respectively)
    - map2: int, input another maze you want to choose. Only 0,1,2,3 are valid (0 and 3 represent Open Field 1 and Open Field 2, respectively, while 1 and 2 represent Maze 1/2 respectively). Map2 Should be different with Map1 (and are required to be bigger than map 1), or it will report an error!!!!!!!
    - save_loc: str, the location you want to save the diagonal figure.
    - f_trace: <class 'pandas.DataFrame'>, default value is f1 which saves basic information of cross_maze paradigm corresponding to f's default value f_CellReg.
    - map_type: str, the map type you want to use to calculation PVC. the map must belongs to a kind of old_map(which has a shape of (n, 144)) or it will raise an error.

    Returns
    -------
    - bool. If the function has successfully run, return True or it will stop the funciton by 'AssertError' or return a False.
    '''
    assert row is not None
    assert row < len(f) # row should not be bigger than the length of f, or it will overflow.
    assert map1 < map2 # map 1 should be bigger than map 2
    ValueErrorCheck(map1, [0,1,2])
    ValueErrorCheck(map2, [1,2,3])

    # Read and Sort Index Map
    print("Step 1 - Read And Sort Index Map")
    if os.path.exists(f['Cell Reg Path'][row]):
        index_map = Read_and_Sort_IndexMap(path = f_CellReg['Cell Reg Path'][row], occur_num = 2, align_type = 'cross_session')
    else:
        print(f_CellReg['Cell Reg Path'][row], 'is not exist!')
        return False, None
    
    # Select Cell Pairs that Both exist in index_map in map1 and map2
    is_cell_detected = np.where(index_map == 0, 0, 1)
    cellpair = np.where(np.nansum(is_cell_detected[[map1,map2],:], axis = 0) == 2)[0]
    index_map = index_map[:, cellpair]
    index_map = index_map.astype(np.int64)

    if index_map.shape[1] <= 10: # did not find cell pair that satisfy our requirement.
        return None

    # Get Trace File Set
    print("Step 2 - Get Trace File Set")
    idx = np.where((f_trace['MiceID'] == f['MiceID'][row])&(f_trace['date'] == f['date'][row]))[0]
    trace_set = TraceFileSet(idx = idx, file = f_trace, Behavior_Paradigm = 'Cross Maze')
    KeyWordErrorCheck(trace_set[0], __file__, keys = [map_type])

    # Generate Polulation vector correlation
    print("Step 3 - Generate Population Vector Correlation And Analyze It (By PVCMatrixAnalysis).")
    # Sort PVC by correction path
    PVC = calc_PVC(rate_map_all1 = trace_set[map1][map_type][index_map[map1,:]-1,:], rate_map_all2 = trace_set[map2][map_type][index_map[map2,:]-1, :])
    if map1 == 0 and map2 != 3:
        map1_order, sep1 = Get_X_Order(map2)
    else:
        map1_order, sep1 = Get_X_Order(trace_set[map1]['maze_type'])
    map2_order, sep2 = Get_X_Order(trace_set[map2]['maze_type'])
    PVC = PVC[map1_order, :]
    PVC = PVC[:, map2_order]
    PVCMatrixAnalysis(PVC, map1, map2, save_loc = save_loc, training_day = f_CellReg['training_day'][row], sep1 = int(sep1+0.5), sep2 = int(sep2+0.5), **barplot_kwgs)

    print("Step 4 - Plot PVC Rate Map")    
    fig = plt.figure(figsize=(8,6))
    ax = Clear_Axes(axes = plt.axes(), close_spines = ['top', 'right'], ifxticks = True, ifyticks = True)
    ax.set_aspect('equal')
    labels = ['Open Field 1', 'Maze 1', 'Maze 2', 'Open Field 2']
    ax.set_xlabel(labels[map2]) # Note that x axis is correlated to map2 while y axis is correlated to map 1
    ax.set_ylabel(labels[map1])

    # show the rate map
    im = ax.imshow(PVC, vmin = np.nanmin(PVC), vmax = np.nanmax(PVC), cmap = cmap, **kwargs)
    cbar = plt.colorbar(im, ax = ax)
    cbar.set_label('Population Vector Correlation')
    ax.invert_yaxis()
    ax.axvline(sep2, color = 'black')
    ax.axhline(sep1, color = 'black')
    plt.savefig(save_loc+'.png', dpi = 600)
    plt.savefig(save_loc+'.svg', dpi = 600)
    plt.close()
    print("Done.",end='\n\n')
    
    return PVC


# Read PVC or Generate PVC
if 1 == False:#os.path.exists(os.path.join(figdata, code_id+' - Cumulative PVC Matrix.pkl')):
    with open(os.path.join(figdata, code_id+' - Cumulative PVC Matrix.pkl'), 'rb') as handle:
        PVCT = pickle.load(handle)
else:
    PVCT = np.zeros((18,144,144), dtype = np.float64)

    for i in range(len(f_CellReg)):
        loc = os.path.join(p, str(int(f_CellReg['MiceID'][i]))+' - '+str(int(f_CellReg['date'][i])))
        mkdir(loc)
        PVC = plot_pvc_ratemap(row = i, save_loc = os.path.join(loc, 'Maze1-Maze2'), map1 = 1, map2 = 2, cmap = 'Spectral')
        if PVC is None:
            PVC = np.zeros((144,144))*np.nan
        PVCT[i,:,:] = PVC
        plot_pvc_ratemap(row = i, save_loc = os.path.join(loc, 'OpenF-Maze1'), map1 = 0, map2 = 1, cmap = 'Spectral')
        plot_pvc_ratemap(row = i, save_loc = os.path.join(loc, 'OpenF-Maze2'), map1 = 0, map2 = 2, cmap = 'Spectral')
        plot_pvc_ratemap(row = i, save_loc = os.path.join(loc, 'OpenF-OpenF'), map1 = 0, map2 = 3, cmap = 'Spectral')

    with open(os.path.join(figdata, code_id+' - Cumulative PVC Matrix.pkl'), 'wb') as f:
        pickle.dump(PVCT,f)

print("Step 4 - Plot PVC Rate Map")  
mice = 11092
idx = np.where(f_CellReg['MiceID'] == mice)[0]
CPVC = np.nanmean(PVCT[idx,:,:], axis = 0)
fig = plt.figure(figsize=(8,6))
ax = Clear_Axes(axes = plt.axes(), close_spines = ['top', 'right'], ifxticks = True, ifyticks = True)
ax.set_aspect('equal')
labels = ['Open Field 1', 'Maze 1', 'Maze 2', 'Open Field 2']
ax.set_xlabel(labels[2]) # Note that x axis is correlated to map2 while y axis is correlated to map 1
ax.set_ylabel(labels[1])

# show the rate map
_, sep1 = Get_X_Order(1)
_, sep2 = Get_X_Order(2)
im = ax.imshow(CPVC, cmap = 'Spectral', vmin = np.min(CPVC), vmax = np.max(CPVC))
cbar = plt.colorbar(im, ax = ax)
cbar.set_label('Population Vector Correlation')
ax.invert_yaxis()
ax.axvline(sep2, color = 'black')
ax.axhline(sep1, color = 'black')
plt.savefig(os.path.join(p, str(mice)+'-Cumulative_PVC.png'), dpi = 600)
plt.savefig(os.path.join(p,str(mice)+'-Cumulative_PVC.svg'), dpi = 600)
plt.close()
print("Done.",end='\n\n')

