# Does place field clustered in decision making point?

from mylib.statistic_test import *

D = [GetDMatrices(1,12),GetDMatrices(2,12)]

# The distance of field center to the nearest decision point
def field_dis_to_decipt(field_center:int, maze_type = None, D:np.ndarray = None, nx:int = 12):
    '''
    This function is to calculate the distance from field center to the nearest decision making point.

    Parameters
    ----------
    field_center: int, the maze id of center bin.
    maze_type: int, only 1 or 2. (Open field does not have decision making point)
    D: np.ndarray, the distance matrix.
    '''
    ValueErrorCheck(nx, [12,24,48])
    if nx != 12:
        field_center = Son2FatherGraph[field_center+1] if nx == 48 else Quarter2FatherGraph[field_center+1]
    else:
        field_center = field_center+1

    ValueErrorCheck(maze_type, [1,2])
    assert field_center in np.arange(1, 145)

    DP = DecisionPoint1 if maze_type == 1 else DecisionPoint2
    dis = np.array(DP.shape[0], d)



code_id = '0031 - Place Field Cluster'

with open(f1['Trace File'][99], 'rb') as handle:
    trace = pickle.load(handle)

print(trace['place_field_all'][0])