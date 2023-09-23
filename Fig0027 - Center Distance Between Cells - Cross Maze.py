# Fig0027-1, Barplot of Peak Curve Density

from mylib.statistic_test import *

code_id = '0027'

idx = np.where((f1['total_session'] >= 4)&((f1['MiceID'] == 11092)|(f1['MiceID'] == 11095))&(f1['session'] == 1))[0]
print(idx)

# In index_map, each cell has 4 value corresponding to cell id in 4 sessions, and if the cell is not detected at a certain session, the value is set as 0.
 
# Only cell pairs which satisfy these 3 criteria are used to analysis:
# 1. Number of place fields of both two cells (of the cell pairs) should be less than 3.
# 2. Both of the two cells must be detected in at least 3 sessions that correlate to 3 different environments, that means 2 open field sessions violates this rule and 
# thus should not take into concerns of analysis. (And naturally, only combination A {open field, maze 1, maze2} and combination B {maze 1, maze 2, open field} are valid
# circumstances.) 
# 3. A cell pair that can be selected for analysis must belong to same combination. (Both A or both B)

# Check if cells have place field more than 3 and thus violate criterion 1.
def check_field_number(cell_id:int, trace:dict):

    if cell_id >= trace['n_neuron']:
        print("OverflowError! cell_id is bigger than number of neuron.")
        return False
    
    # If number of place field is equal to or less than 3, return true.
    if len(trace['place_field_all'][int(cell_id)].keys()) <= 5:
        return True
    else:
        return False

# Divide cells into 2 groups (combination), and return their index.
def DivideCombination(index_map = None, trace_set = None):
    A = [] # Combination A
    B = [] # Combination B

    index_map = index_map.astype(np.int64)
    # Cells that are detected in 3 sessions at least:
    index_tmp = np.where(index_map == 0, 0, 1)
    num_of_seq = np.nansum(index_tmp, axis = 0)
    # value >= 3 are selected
    idx = np.where(num_of_seq >= 3)[0]

    # Ergodic of cells that satisfy criterion 2. Details see above ↑.
    for i in idx:
        # if maze 1 data loss, continue
        if index_map[1,i] == 0:
            continue
        # if maze 2 data loss, continue
        if index_map[2,i] == 0:
            continue
        
        # if the cell is detected in open field 2 session, put it into list B.
        if index_map[0,i] == 0 and check_field_number(index_map[1,i], trace_set[1]) and check_field_number(index_map[2,i], trace_set[2]) and check_field_number(index_map[3,i], trace_set[3]):
            B.append(i)
            continue

        # if the cell is detected in open field 1 session, put it into list A.
        if check_field_number(index_map[0,i], trace_set[0]) and check_field_number(index_map[1,i], trace_set[1]) and check_field_number(index_map[2,i], trace_set[2]):
            A.append(i)
        
    return np.array(A, dtype = np.int64), np.array(B, dtype = np.int64)

# Get field center of a cell.
def GetFieldCenterIDX(cell_id:int, trace:dict):
    old_map = trace['old_map_clear'][int(cell_id)]
    return np.argmax(old_map)

# Select 2 cells (that are belong to the same combination) and calculate inter field center distance(IFCD)
def calc_IFCD_for_4session(index_map = None, trace_set = None):
    # Divide cells into 2 combination.
    A, B = DivideCombination(index_map = index_map, trace_set = trace_set)
    print("Combination A,B:",A,B)
    # If there're a cells in A and b cells in B, than the total number of 3 dimension coordinate is a(a-1)/2 + b(b-1)/2.
    C = np.concatenate([A,B])
    label = np.concatenate([np.repeat('A', A.shape[0]), np.repeat('B',B.shape[0])])
    IFCD = np.zeros((A.shape[0]+B.shape[0],A.shape[0]+B.shape[0],3), dtype = np.float64)

    # Select cells in combination A or B to make cell pairs and calculate IFCD value:
    for i in range(IFCD.shape[0]-1):
        for j in range(i+1, IFCD.shape[0]):
            k1 = 0 if label[i] == 'A' else 3
            k2 = 0 if label[j] == 'A' else 3

            center_list1 = np.array([GetFieldCenterIDX(index_map[k1,C[i]], trace_set[k1]), GetFieldCenterIDX(index_map[1,C[i]], trace_set[1]), GetFieldCenterIDX(index_map[2,C[i]], trace_set[2])], dtype = np.int64)+1
            center_list2 = np.array([GetFieldCenterIDX(index_map[k2,C[j]], trace_set[k2]), GetFieldCenterIDX(index_map[1,C[j]], trace_set[1]), GetFieldCenterIDX(index_map[2,C[j]], trace_set[2])], dtype = np.int64)+1
            IFCD[i,j,:] = InterFieldCenterDistance(center_list1 = center_list1, center_list2 = center_list2, maze_list = np.array([0,1,2], dtype = np.int64))
            IFCD[j,i,:] = cp.deepcopy(IFCD[i,j,:])

    return IFCD

# calculate chance level as control.
def calc_IFCD_for_4session_chance_level(cells = 0):
    if cells == 0:
        # no cells return None
        return

    IFCD_cl = np.zeros((cells,cells,3), dtype = np.float64)
    for i in range(cells-1):
        for j in range(i+1, cells):
            center_rand1 = np.random.choice(np.arange(1,145), size = 3, replace = True)
            center_rand2 = np.random.choice(np.arange(1,145), size = 3, replace = True)
            IFCD_cl[i,j,:] = InterFieldCenterDistance(center_list1 = center_rand1, center_list2 = center_rand2, maze_list = np.array([0,1,2], dtype = np.int64))
            IFCD_cl[j,i,:] = cp.deepcopy(IFCD_cl[i,j,:]) 

    return IFCD_cl        

# Visualization IFCD ================================================================================================================================================
# project 3D coordinate on a 2D plane. You should select 2 axis to span the 2D space for projection.
def IFCDmatrix_to_vector(IFCD):
    cells = IFCD.shape[0]
    vec = np.zeros((int(cells*(cells-1)/2),3), dtype = np.float64)
    k = 0
    for i in range(0,cells-1):
        for j in range(i+1, cells):
            vec[k,:] = IFCD[i,j,:]  
            k += 1

    return vec

def plot_2D_projection(IFCD:np.ndarray, axis1 = 0, axis2 = 1, IFCD_cl = None):
    maze_list = ['Open Field', 'Maze 1', 'Maze 2']
    MAX = [11*np.sqrt(2), FastDistance(start=1, goal=144, nx = 12, maze_type=1), FastDistance(start=1, goal=144, nx = 12, maze_type=2)]
    plt.figure(figsize = (6,6))
    ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifyticks = True, ifxticks = True)
    ax.set_xlabel(maze_list[axis1])
    ax.set_ylabel(maze_list[axis2])
    x = IFCDmatrix_to_vector(IFCD)[:,axis1] / MAX[axis1]
    y = IFCDmatrix_to_vector(IFCD)[:,axis2] / MAX[axis2]
    ax.set_aspect('equal')
    ax.set_xticks(np.linspace(0,1,6))
    ax.set_yticks(np.linspace(0,1,6))
    ax.axis([0,1,0,1])

    print(x,y)
    ax.plot(x,y,'o', label = 'Experimental value')
    if IFCD_cl is not None:
        x_cl = IFCDmatrix_to_vector(IFCD_cl)[:, axis1]
        y_cl = IFCDmatrix_to_vector(IFCD_cl)[:, axis2]
        ax.plot(x_cl, y_cl,'^',label = 'Chance Level')

    plt.show()


# =================================================================================================================================================================
# =================================================================================================================================================================
# =================================================================================================================================================================
# =================================================================================================================================================================
# Divide cells into 2 groups (combination), and return their index.
def DivideCombination2(index_map = None, trace_set = None):
    C = []

    index_map = index_map.astype(np.int64)
    # Cells that are detected in 3 sessions at least:
    index_tmp = np.where(index_map == 0, 0, 1)
    num_of_seq = np.nansum(index_tmp, axis = 0)
    # value >= 3 are selected
    idx = np.where(num_of_seq >= 3)[0]

    # Ergodic of cells that satisfy criterion 2. Details see above ↑.
    for i in idx:
        # if maze 1 data loss, continue
        if index_map[1,i] == 0:
            continue
        # if maze 2 data loss, continue
        if index_map[2,i] == 0:
            continue
        
        # if maze 1, maze 2 data are existed, put it into list A.
        if check_field_number(index_map[1,i], trace_set[1]) and check_field_number(index_map[2,i], trace_set[2]):
            C.append(i)
        
    return np.array(C, dtype = np.int64)

# Select 2 cells (that are belong to the same combination) and calculate inter field center distance(IFCD)
def calc_IFCD_for_4session2(index_map = None, trace_set = None):
    # Divide cells into 2 combination.
    C = DivideCombination2(index_map = index_map, trace_set = trace_set)
    IFCD = np.zeros((C.shape[0], C.shape[0],2), dtype = np.float64)

    # Select cells in combination C to make cell pairs and calculate IFCD value:
    for i in range(IFCD.shape[0]-1):
        for j in range(i+1, IFCD.shape[0]):
            center_list1 = np.array([GetFieldCenterIDX(index_map[1,C[i]], trace_set[1]), GetFieldCenterIDX(index_map[2,C[i]], trace_set[2])], dtype = np.int64)+1
            center_list2 = np.array([GetFieldCenterIDX(index_map[1,C[j]], trace_set[1]), GetFieldCenterIDX(index_map[2,C[j]], trace_set[2])], dtype = np.int64)+1
            IFCD[i,j,:] = InterFieldCenterDistance(center_list1 = center_list1, center_list2 = center_list2, maze_list = np.array([1,2], dtype = np.int64))
            IFCD[j,i,:] = cp.deepcopy(IFCD[i,j,:])

    return IFCD

# calculate chance level as control.
def calc_IFCD_for_4session_chance_level2(cells = 0):
    if cells == 0:
        # no cells return None
        return

    IFCD_cl = np.zeros((cells,cells,2), dtype = np.float64)
    for i in range(cells-1):
        for j in range(i+1, cells):
            center_rand1 = np.random.choice(np.arange(1,145), size = 2, replace = True)
            center_rand2 = np.random.choice(np.arange(1,145), size = 2, replace = True)
            IFCD_cl[i,j,:] = InterFieldCenterDistance(center_list1 = center_rand1, center_list2 = center_rand2, maze_list = np.array([1,2], dtype = np.int64))
            IFCD_cl[j,i,:] = cp.deepcopy(IFCD_cl[i,j,:]) 

    return IFCD_cl        

# Visualization IFCD ================================================================================================================================================
# project 3D coordinate on a 2D plane. You should select 2 axis to span the 2D space for projection.
def IFCDmatrix_to_vector2(IFCD):
    cells = IFCD.shape[0]
    vec = np.zeros((int(cells*(cells-1)/2),2), dtype = np.float64)
    k = 0
    for i in range(0,cells-1):
        for j in range(i+1, cells):
            vec[k,:] = IFCD[i,j,:]  
            k += 1

    return vec

def plot_2D_projection2(IFCD:np.ndarray, axis1 = 0, axis2 = 1, IFCD_cl = None, save_loc = None, **kwargs):
    maze_list = ['Maze 1', 'Maze 2']
    MAX = [11*np.sqrt(2),11*np.sqrt(2)]#[FastDistance(start=1, goal=144, nx = 12, maze_type=1), FastDistance(start=1, goal=144, nx = 12, maze_type=2)]
    plt.figure(figsize = (6,8))
    ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifyticks = True, ifxticks = True)
    ax.set_xlabel(maze_list[axis1])
    ax.set_ylabel(maze_list[axis2])
    x = IFCDmatrix_to_vector2(IFCD)[:,axis1] / MAX[axis1]
    y = IFCDmatrix_to_vector2(IFCD)[:,axis2] / MAX[axis2]
    ax.set_aspect('equal')
    ax.set_xticks(np.linspace(0,1,6))
    ax.set_yticks(np.linspace(0,1,6))
    ax.axis([0,1,0,1])

    ax.plot(x,y,'o', label = 'Experimental value', **kwargs)
    if IFCD_cl is not None:
        x_cl = IFCDmatrix_to_vector2(IFCD_cl)[:, axis1] / MAX[axis1]
        y_cl = IFCDmatrix_to_vector2(IFCD_cl)[:, axis2] / MAX[axis2]
        ax.plot(x_cl, y_cl,'^',label = 'Chance Level', **kwargs)

    ax.legend(facecolor = 'white', edgecolor = 'white', title = 'Data Type', loc = 'upper left', framealpha = 0.6, bbox_to_anchor = (0.03,1.15))
    if save_loc is not None:
        plt.savefig(save_loc+'.png', dpi = 600)
        plt.savefig(save_loc+'.svg', dpi = 600)
        plt.close()
    else:
        plt.show()

mkdir(os.path.join(figpath, code_id))
for i in idx:
    CellRegPath = os.path.join(CM_path, str(int(f1['MiceID'][i])), str(int(f1['date'][i])), 'cross_session', 'AlignedResults', 'cellRegistered.mat')
    print(str(int(f1['MiceID'][i])), str(int(f1['date'][i])),":")
    index_map = Read_and_Sort_IndexMap(path = CellRegPath, occur_num = 3, align_type = 'cross_session', name_label = 'SFP2022')

    # Combine trace file corresponding to the 4 sessions together.
    sessions_idx = np.where((f1['MiceID'] == f1['MiceID'][i])&(f1['date'] == f1['date'][i]))[0]
    trace_set = []
    for j in sessions_idx:
        with open(f1['Trace File'][j], 'rb') as handle:
            trace = pickle.load(handle)
            trace_set.append(trace)

    # calculating IFCD
    IFCD = calc_IFCD_for_4session2(index_map = index_map, trace_set = trace_set)
    if len(IFCD) == 0:
        continue
    IFCD_cl = calc_IFCD_for_4session_chance_level2(cells = IFCD.shape[0])
    plot_2D_projection2(IFCD, markersize = 3, IFCD_cl = IFCD_cl, save_loc=os.path.join(figpath, code_id, str(int(f1['MiceID'][i]))+'-'+str(int(f1['date'][i]))+'Maze1 vs Maze2'), alpha = 0.8)