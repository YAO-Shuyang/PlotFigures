from mylib.statistic_test import *

code_id = "0046 - Footprint & New recruit cells"
loc = join(figpath, code_id)
datap = join(figdata, code_id)
mkdir(loc)
mkdir(datap)


dates = ['20220820', '20220822', '20220824', '20220826',
         '20220828', '20220830']
foot_print_maze1 = [join(r"E:\Data\FigData\0046 - Footprint & New recruit cells\Maze1-footprint", 'SFP'+d+'.mat') for d in dates]
foot_print_maze2 = [join(r"E:\Data\FigData\0046 - Footprint & New recruit cells\Maze2-footprint", 'SFP'+d+'.mat') for d in dates]

def field_locate_on_area(place_field: dict, check_area: list or np.ndarray) -> bool:
    total_fields = []
    for k in place_field.keys():
        total_fields += place_field[k]
    
    total_fields = np.array(total_fields, dtype=np.int64)
    
    old_nodes = np.unique(S2F[total_fields-1])
    for i in old_nodes:
        if i in check_area:
            return True
    
    return False

def get_cell_collections(place_field_all: list[dict], is_placecell: np.ndarray, check_area: list) -> np.ndarray:
    cells = []
    for i, field in enumerate(place_field_all):
        if is_placecell[i] == 0:
            continue
        if field_locate_on_area(field, check_area=check_area):
            cells.append(i)
    
    return np.array(cells, dtype=np.int64)

def get_footprint(SFP_loc:str = None):
    if os.path.exists(SFP_loc):
        with h5py.File(SFP_loc, 'r') as f:
            sfp = np.array(f['SFP'])
            
    return sfp


def plot_figure(trace_set: list[dict], SFP_list: list[str], save_loc: str = None, file_name: str = None, check_area = []):
    assert len(trace_set) == 6 and len(SFP_list) == 6
    
    fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(6*3.2, 5))
    
    for i in range(6):
        cells = get_cell_collections(trace_set[i]['place_field_all'], trace_set[i]['is_placecell'], check_area)
        sfp = get_footprint(SFP_list[i])
        footprint = np.nanmax(sfp,axis = 2)
            
        ax = Clear_Axes(axes[i])
        ax.imshow(footprint, cmap = 'hot')
        
        for j in cells:
            x, y = np.where(sfp[:, :, j] == np.nanmax(sfp[:, :, j]))
            ax.plot(y, x, 'o', color = 'yellow', markersize = 3)
        
    plt.show()
        
        


idx = np.where((f1['date'] >= 20220820)&(f1['MiceID'] == 11095)&(f1['maze_type'] == 1))[0]
trace_set = TraceFileSet(idx, file=f1, tp=r'E:\Data\Cross_maze')
plot_figure(trace_set, foot_print_maze1, check_area=CorrectPath_maze_1[2:3])

idx = np.where((f1['date'] >= 20220820)&(f1['MiceID'] == 11095)&(f1['maze_type'] == 2))[0]
trace_set = TraceFileSet(idx, file=f1, tp=r'E:\Data\Cross_maze')
plot_figure(trace_set, foot_print_maze2, check_area=CorrectPath_maze_2[31:32])
    
    
    