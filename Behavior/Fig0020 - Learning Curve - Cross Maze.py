# Fig0020-1, Plotting learning curve in behavioral pattern, and explore time of each lap are plotted
# Fig0020-2, Learning curve ploting the mean of laps on each training day.

from mylib.statistic_test import *
from matplotlib.gridspec import GridSpec

code_id = '0020 - Learning Curve'
loc = os.path.join(figpath, code_id)
mkdir(loc)

maze_indices = np.where(f_pure_behav['maze_type'] != 0)[0]
if os.path.exists(join(figdata, code_id+' .pkl')) == False:
    Data = {
        "MiceID",
        "Stage",
        "Maze Type",
        "Lap",
        "Lap Time",
        "Explore Time",
    }
else:
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
            Data = pickle.load(handle)