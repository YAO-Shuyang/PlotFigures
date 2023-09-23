# Fig0016-1 Spatial Information Line Graph, for Simple maze paradigm.

# data ara saved in 0017data.pkl, data is a dict type with 5 members: 
#                                     'MiceID': numpy.ndarray, shape(33301,), dtype(str) {e.g. 10031}
#                                     'Training Day': numpy.ndarray, shape(33301,), dtype(str) {e.g. Day 1}
#                                     'SI': numpy.ndarray, shape(33301,), dtype(np.float64)
#                                     'Cell': numpy.ndarray, shape(33301,), dtype(np.int64) {e.g. Cell 1}
#                                     'Maze Type': numpy.ndarray, shape(33301,), dtype(str) {e.g. Maze 1}
# All Cells (33301 cells), spike number > 30

from mylib.statistic_test import *

code_id = '0016'

if os.path.exists(os.path.join(figdata, code_id+'data.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Cell','SI'], f = f2, function = SpatialInformation_Interface, 
                              file_name = code_id, behavior_paradigm = 'SimpleMaze')   
else:
    with open(os.path.join(figdata, code_id+'data.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

cell_number = Data['SI'].shape[0]
fs = 14

# Spatial Information-------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = 'Training Day',y = 'SI', data = Data, hue = 'Maze Type', ax = ax, legend = True)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze Type', loc = 'upper left')
#ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Spatial Information', fontsize = fs)
ax.set_title('Simple ('+str(cell_number)+' cells)', fontsize = fs)
ax.axis([-0.5,8.5,0,1])
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8], labels = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.set_yticks(np.linspace(0,1,6))
plt.tight_layout()
plt.savefig(os.path.join(figpath,code_id+'-SI-CrossMaze.svg'), dpi = 600)
plt.savefig(os.path.join(figpath,code_id+'-SI-CrossMaze.png'), dpi = 600)
plt.show()