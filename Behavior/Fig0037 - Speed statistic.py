from mylib.statistic_test import *

code_id = '0037 - Speed Statistic'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Speed', 'Maze Bin'], f = f1, 
                              function = Speed_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        

def plot_figure_maze(Data, env, mice, maze_type, file_name):
    idx = np.where((Data['MiceID'] == mice)&(Data['Maze Type'] == env))[0]
    SubData = SubDict(Data, keys=Data.keys(), idx=idx)
    
    image = ImageBase(maze_type)
    MAX = np.nanmax(SubData['Speed'])
    image.add_main_cp_top_band(MAX)
    image.add_main_ip_top_band(MAX)
      
    sns.lineplot(x='Maze Bin', y='Speed', data=SubData, hue = 'Training Day', palette='rainbow',
                 markeredgecolor = None, ax = image.ax2)
    image.ax2.legend(bbox_to_anchor = (1,1), loc = 'upper left', title='Training Day', 
              title_fontsize = 8, fontsize = 8, ncol = 2, facecolor = 'white', edgecolor = 'white')
    image.ax2.set_xlabel("Maze Track (bin)\n(Linearized)")
    image.ax2.set_ylabel("Speed (cm/s)")
    image.ax2.set_yticks(ColorBarsTicks(MAX, is_auto=True, tick_number=5))
    image.ax2.axis([0, np.nanmax(SubData['Maze Bin']+1), 0, MAX*1.1])
    image.savefig(loc, file_name=file_name)

plot_figure_maze(Data, 'Maze 1', '11095', 1, '11095-Maze 1')
plot_figure_maze(Data, 'Maze 2', '11095', 2, '11095-Maze 2')
plot_figure_maze(Data, 'Maze 1', '11092', 1, '11092-Maze 1')
plot_figure_maze(Data, 'Maze 2', '11092', 2, '11092-Maze 2')
