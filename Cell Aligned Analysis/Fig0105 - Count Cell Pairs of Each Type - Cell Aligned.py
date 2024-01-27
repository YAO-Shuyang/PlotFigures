from mylib.statistic_test import *

code_id = '0105 - Count Cell Pairs of Each Type'
p = os.path.join(figpath, 'Cell Aligned', code_id)
mkdir(p)

def count_cell_pairs(row:int, map1:int, map2:int, f_trace:pd.DataFrame = f1, f:pd.DataFrame = f_CellReg, 
                     **kwargs):
    '''
    Author: YAO Shuyang
    Date: Jan 29th, 2023
    Note: To count the number of 'pc-pc', 'pc-npc', 'npc-pc', 'npc-npc' cell pairs in each day. Plot a bar figure.

    Parameters
    ----------
    - f: <class 'pandas.DataFrame'>, default value is f_CellReg which saves directories of all cross_session cellRegistered.mat file
    - row: int, required
        The row id of the line want to read in f. Row should not be larger than the length of f. If f is default file(f_CellReg), row should not be bigger than 17.
    - map1: int, required
        Input a maze you want to choose. Only 0,1,2,3 are valid (0 and 3 represent Open Field 1 and Open Field 2, respectively, while 1 and 2 represent Maze 1/2 respectively)
    - map2: int, required
        Input another maze you want to choose. Only 0,1,2,3 are valid (0 and 3 represent Open Field 1 and Open Field 2, respectively, while 1 and 2 represent Maze 1/2 respectively). Map2 Should be different with Map1 (and are required to be bigger than map 1), or it will report an error!!!!!!!
    - f_trace: <class 'pandas.DataFrame'>, default value is f1 which saves basic information of cross_maze paradigm corresponding to f's default value f_CellReg.

    Returns
    -------
    - (numpy.NDarray[int], numpy.NDarray[U])
        the first array contains the number of each kind of pair, while the second array contains the name of each pair.
    '''
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

    # Get Trace File Set
    print("Step 2 - Get Trace File Set")
    idx = np.where((f_trace['MiceID'] == f['MiceID'][row])&(f_trace['date'] == f['date'][row]))[0]
    trace_set = TraceFileSet(idx = idx, file = f_trace, Behavior_Paradigm = 'Cross Maze')
    KeyWordErrorCheck(trace_set[0], __file__, keys = ['is_placecell'])

    is_placecell1 = trace_set[map1]['is_placecell'][index_map[map1,:]-1]
    is_placecell2 = trace_set[map2]['is_placecell'][index_map[map2,:]-1]

    counts = np.zeros(4, dtype = np.int64)
    pair_types = np.array(['pc-pc', 'pc-npc', 'npc-pc', 'npc-npc'])

    counts[0] = len(np.where((is_placecell1 == 1)&(is_placecell2 == 1))[0])
    counts[1] = len(np.where((is_placecell1 == 1)&(is_placecell2 == 0))[0])
    counts[2] = len(np.where((is_placecell1 == 0)&(is_placecell2 == 1))[0])
    counts[3] = len(np.where((is_placecell1 == 0)&(is_placecell2 == 0))[0])
    print("Done.", end='\n\n')

    return counts, pair_types, counts/np.nansum(counts)

def plot_bar_figure(Data:dict, save_loc:str, training_day:str, MiceID:int|str, figsize = (4,3), **kwargs):
    '''
    Author: YAO Shuyang
    Date: Jan 29th, 2023
    Note: to plot barplot for each training day

    Parameters
    ----------
    Data: dict
    '''
    KeyWordErrorCheck(Data, __file__, ['Env Pair', 'Cell Pair', 'Counts'])
    fig = plt.figure(figsize = figsize)
    ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
    sns.barplot(x = 'Env Pair', y = 'Counts', data = Data, hue = 'Cell Pair', palette='Set2', ax = ax, errcolor = 'black', capsize = 0.1, errwidth = 1)
    ax.legend(edgecolor = 'white', facecolor = 'white', loc = 'upper center', title = 'Cell Pair Type [pc: place cell; npc: non-place cell]', ncol = 4, 
              fontsize = 7, title_fontsize = 8, markerscale = 0.8)
    ax.set_xlabel("Environment Pair")
    ax.set_ylabel("Number of Cell Pairs")
    ax.set_title(str(MiceID)+' - '+training_day)
    MAX = np.nanmax(Data['Counts'])
    ax.axis([-0.5, 3.5, 0, int(MAX*1.3)])
    ax.set_yticks(ColorBarsTicks(peak_rate = int(MAX*1.3), is_auto = True))
    plt.xticks(ticks = [0,1,2,3], labels = ['OF-M1','OF-M2','OF-OF','M1-M2'])
    plt.tight_layout()
    plt.savefig(save_loc+'.png', dpi = 1200)
    plt.savefig(save_loc+'.svg', dpi = 1200)
    plt.close()

def plot_line_graph(Data:dict, save_loc:str, MiceID:int = 11095, env_type:str = 'Maze1-Maze2', **kwargs):
    '''
    Author: YAO Shuyang
    Date: Jan 29th, 2023
    Note: to plot barplot for each training day

    Parameters
    ----------
    Data: dict
    
    env_type: str, optional, only 'OpenF-Maze1','OpenF-Maze2','OpenF-OpenF','Maze1-Maze2' are valid.

    save_loc: str, the direcotry to save your figure.
    '''
    KeyWordErrorCheck(Data, __file__, ['Env Pair', 'Cell Pair', 'Counts', 'Ratio', 'Training Day'])
    ValueErrorCheck(env_type, ['OpenF-Maze1','OpenF-Maze2','OpenF-OpenF','Maze1-Maze2'])

    idx = np.where((Data['MiceID'] == MiceID)&(Data['Env Pair'] == env_type))[0]
    SubData = DivideData(Data, index = idx, keys = ['Env Pair', 'Cell Pair', 'Ratio', 'Training Day'])

    fig = plt.figure(figsize = (4,3))
    
    ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
    sns.lineplot(x = 'Training Day', y = 'Ratio', data = SubData, hue = 'Cell Pair', palette='Set2', ax = ax, err_style = 'bars',
                 err_kws = {'elinewidth':2, 'capsize':3, 'capthick':2})
    ax.legend(edgecolor = 'white', facecolor = 'white', loc = 'upper center', title = 'Cell Pair Type [pc: place cell; npc: non-place cell]', ncol = 4, 
              fontsize = 7, title_fontsize = 8, markerscale = 0.8)
    ax.set_xlabel("Training Day")
    ax.set_ylabel("Ratio of Cell Pairs")
    ax.set_title(str(MiceID)+' '+env_type)
    MAX = np.nanmax(Data['Counts'])
    ax.axis([-0.5, 8.5, 0, 1.3])
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0,1.3])
    plt.xticks([0,1,2,3,4,5,6,7,8], ['D'+str(i) for i in range(1,10)])
    plt.tight_layout()
    plt.savefig(save_loc+str(MiceID)+' - '+env_type+' [lineplot].png', dpi = 1200)
    plt.savefig(save_loc+str(MiceID)+' - '+env_type+' [lineplot].svg', dpi = 1200)
    plt.close()




if os.path.exists(os.path.join(figdata, code_id+'.pkl')):
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    # Initiate data struct.
    Data = {'Training Day':np.array([]), 'MiceID':np.array([], np.int64), 'Counts':np.array([], np.int64), 'Cell Pair':np.array([]), 
            'Env Pair':np.array([]), 'Ratio':np.array([], dtype = np.float64)}

    for i in range(len(f_CellReg)):
        print(i, '--------------------------')
        loc = os.path.join(p, str(int(f_CellReg['MiceID'][i]))+' - '+str(int(f_CellReg['date'][i])))
        mkdir(loc)
        Data['Training Day'] = np.concatenate([Data['Training Day'], np.repeat(f_CellReg['training_day'][i], 16)])
        Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(f_CellReg['MiceID'][i], 16)])

        counts1, _, ratio1 = count_cell_pairs(i, 0, 1) # OpenF-Maze1
        counts2, _, ratio2 = count_cell_pairs(i, 0, 2) # OpenF-Maze2
        counts3, _, ratio3 = count_cell_pairs(i, 0, 3) # OpenF-OpenF
        counts4, _, ratio4 = count_cell_pairs(i, 1, 2) # Maze1-Maze2

        Data['Cell Pair'] = np.concatenate([Data['Cell Pair'], ['pc-pc', 'pc-npc', 'npc-pc', 'npc-npc','pc-pc', 'pc-npc', 'npc-pc', 'npc-npc',
                                                          'pc-pc', 'pc-npc', 'npc-pc', 'npc-npc','pc-pc', 'pc-npc', 'npc-pc', 'npc-npc']])
        Data['Env Pair'] = np.concatenate([Data['Env Pair'], np.repeat(['OpenF-Maze1','OpenF-Maze2','OpenF-OpenF','Maze1-Maze2'], 4)])
        Data['Counts'] = np.concatenate([Data['Counts'], counts1, counts2, counts3, counts4])
        Data['Ratio'] = np.concatenate([Data['Ratio'], ratio1, ratio2, ratio3, ratio4])

        plot_bar_figure(Data = {'Cell Pair':['pc-pc', 'pc-npc', 'npc-pc', 'npc-npc','pc-pc', 'pc-npc', 'npc-pc', 'npc-npc',
                                                          'pc-pc', 'pc-npc', 'npc-pc', 'npc-npc','pc-pc', 'pc-npc', 'npc-pc', 'npc-npc'], 
                                'Env Pair':np.repeat(['OpenF-Maze1','OpenF-Maze2','OpenF-OpenF','Maze1-Maze2'], 4),
                                'Counts':np.concatenate([counts1, counts2, counts3, counts4])}, 
                        save_loc = os.path.join(loc, 'Counts Number [barplot]'), 
                        training_day = f_CellReg['training_day'][i],
                        MiceID = int(f_CellReg['MiceID'][i]))


    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as f:
        pickle.dump(Data, f)

    fData = pd.DataFrame(Data)
    fData.to_excel(os.path.join(figdata, code_id+'.xlsx'), sheet_name = 'CountCellPairs', index = False)


# plot lineplot
loc = os.path.join(p, 'Lineplot')
mkdir(loc)
plot_line_graph(Data, save_loc = os.path.join(loc,''), MiceID = 11095, env_type = 'OpenF-OpenF')
plot_line_graph(Data, save_loc = os.path.join(loc,''), MiceID = 11095, env_type = 'OpenF-Maze1')
plot_line_graph(Data, save_loc = os.path.join(loc,''), MiceID = 11095, env_type = 'OpenF-Maze2')
plot_line_graph(Data, save_loc = os.path.join(loc,''), MiceID = 11095, env_type = 'Maze1-Maze2')

plot_line_graph(Data, save_loc = os.path.join(loc,''), MiceID = 11092, env_type = 'OpenF-OpenF')
plot_line_graph(Data, save_loc = os.path.join(loc,''), MiceID = 11092, env_type = 'OpenF-Maze1')
plot_line_graph(Data, save_loc = os.path.join(loc,''), MiceID = 11092, env_type = 'OpenF-Maze2')
plot_line_graph(Data, save_loc = os.path.join(loc,''), MiceID = 11092, env_type = 'Maze1-Maze2')