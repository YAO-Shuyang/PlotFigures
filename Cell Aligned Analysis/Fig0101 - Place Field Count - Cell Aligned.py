from mylib.statistic_test import *

code_id = '0101 - Field Number'
p = os.path.join(figpath, 'Cell Aligned', code_id)
mkdir(p)

def place_field_cellAligned(trace_set:list = [], index_map:np.ndarray = None, map1:int = 1, map2:int = 2):
    '''
    Author: YAO Shuyang
    Date: Jan 26th, 2023
    Note: To plot a diagonal figure to show whether cells that encode map1 with more fields will tend to encode map2 with more fields.   
    '''
    print("    Generate Place Field Number...")
    field_numbers = np.zeros((2, index_map.shape[1]), dtype = np.int64)
    for i in tqdm(range(index_map.shape[1])):
        field_numbers[0,i] = len(trace_set[map1]['place_field_all'][int(index_map[map1,i])-1].keys())
        field_numbers[1,i] = len(trace_set[map2]['place_field_all'][int(index_map[map2,i])-1].keys())

    return field_numbers

for i in range(len(f_CellReg)):
    # plot op - op as control
    mkdir(os.path.join(p, str(int(f_CellReg['MiceID'][i]))+' - '+str(int(f_CellReg['date'][i]))))
    plot_diagonal_figure(row = i, map1 = 0, map2 = 3, function = place_field_cellAligned, markersize = 2, residents = 5, add_noise = True, 
                         save_loc = os.path.join(p, str(int(f_CellReg['MiceID'][i]))+' - '+str(int(f_CellReg['date'][i])), 'control - OpenF1-OpenF2'))
    plot_diagonal_figure(row = i, map1 = 0, map2 = 1, function = place_field_cellAligned, markersize = 2, residents = 5, add_noise = True,
                         save_loc = os.path.join(p, str(int(f_CellReg['MiceID'][i]))+' - '+str(int(f_CellReg['date'][i])), 'OpenF-Maze1'))
    plot_diagonal_figure(row = i, map1 = 0, map2 = 2, function = place_field_cellAligned, markersize = 2, residents = 5, add_noise = True,
                         save_loc = os.path.join(p, str(int(f_CellReg['MiceID'][i]))+' - '+str(int(f_CellReg['date'][i])), 'OpenF-Maze2'))
    plot_diagonal_figure(row = i, map1 = 1, map2 = 2, function = place_field_cellAligned, markersize = 2, residents = 5, add_noise = True,
                         save_loc = os.path.join(p, str(int(f_CellReg['MiceID'][i]))+' - '+str(int(f_CellReg['date'][i])), 'Maze1-Maze2'))

    print("--------------------------------------------------------------------------------------------", end = '\n\n\n\n')                    
    