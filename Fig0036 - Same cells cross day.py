from mylib.statistic_test import *

code_id = '0036 - Same cells cross day'
loc = os.path.join(figpath, code_id)
mkdir(loc)

def plot_figure(data, save_loc, file_name):
    cell_num = np.where(data!=0, 1, 0)
    num_vec = np.nansum(cell_num, axis = 0)
    num_dis = np.zeros(data.shape[0])

    for d in range(data.shape[0]):
        num_dis[d] = len(np.where(num_vec >= d+1)[0])

    print(num_dis)
    fig = plt.figure(figsize=(3,2))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifyticks=True)
    cmap = matplotlib.colormaps['rainbow']
    colors = cmap(np.linspace(0, 1, data.shape[0]))
    for d in range(data.shape[0]):
        ax.bar(d+1, num_dis[d], color = colors[d], label = str(d+1), width=0.6)
    ax.legend(facecolor = 'white', bbox_to_anchor=(1, 1), fontsize = 8, 
              title_fontsize = 8, loc='upper left',
              edgecolor = 'white', title = 'number of\nsession(s)')
    ax.set_xticks(np.arange(1, data.shape[0]+1))
    ax.axis([0.5, data.shape[0]+1, 0, np.nanmax(num_dis)*1.2])
    ax.set_xlabel("Cells were detected in at least\nhow many Session(s)")
    ax.set_ylabel("Number of Cells")
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, file_name+'.png'), dpi = 2400)
    plt.savefig(os.path.join(save_loc, file_name+'.svg'), dpi = 2400)
    plt.close()


data = ReadCellReg(cellReg_95_maze1)
plot_figure(data, loc, '11095-Maze1')

data = ReadCellReg(cellReg_95_maze2)
plot_figure(data, loc, '11095-Maze2')