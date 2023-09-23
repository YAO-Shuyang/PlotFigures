# Fig0014 is about firing rate, either peak rate or mean rate.
# Simple Maze

# Firing rate of rate map 12 x 12.
# Fig0014-1: Peak rate vs training day.
# Fig0014-2: Mean rate vs training day.
# Fig0014-3: Peak rate divided by path type, maze 1.
# Fig0014-4: Peak rate divided by path type, maze 2.
# Fig0014-5: Mean rate divided by path type, maze 1.
# Fig0014-6: Mean rate divided by path type, maze 2.

from mylib.statistic_test import *

code_id = '0014'

if os.path.exists(os.path.join(figdata,code_id+'data.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['peak_rate','mean_rate','peak_rate_on_path','mean_rate_on_path','path_type'], 
                                f = f2, function = FiringRateProcess_Interface, file_name = code_id+'', behavior_paradigm = 'SimpleMaze')
else:
    with open(os.path.join(figdata,code_id+'data.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

x = MultipleLocator(1)
cell_number = Data['peak_rate'].shape[0]
fs = 14


# Fig0015-1: Peak rate vs training day. ---------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = 'Training Day',y = 'peak_rate', data = Data, hue = 'Maze Type', ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze Type', loc = 'lower right')
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Peak Rate / Hz', fontsize = fs)
ax.set_title('Simple Maze ('+str(cell_number)+")", fontsize = fs)
ax.axis([-0.5,8.5,0,12])
ax.set_yticks([0,2,4,6,8,10,12])
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8], labels=['D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(figpath, code_id+'-1-Peak Rate-SimpleMaze.svg'), dpi = 600)
plt.savefig(os.path.join(figpath, code_id+'-1-Peak Rate-SimpleMaze.png'), dpi = 600)
plt.close()

# Fig0015-2: Mean rate vs training day. ---------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = 'Training Day',y = 'mean_rate', data = Data, hue = 'Maze Type', ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze Type', loc = 'upper left')
ax.axis([-0.5,8.5,0,1])
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8], labels = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.set_yticks(np.linspace(0,1,6))
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Mean Rate / Hz', fontsize = fs)
ax.set_title('Simple Maze ('+str(cell_number)+")", fontsize = fs)
plt.tight_layout()
plt.savefig(os.path.join(figpath,code_id+'-2-Mean Rate-SimpleMaze.svg'), dpi = 600)
plt.savefig(os.path.join(figpath,code_id+'-2-Mean Rate-SimpleMaze.png'), dpi = 600)
plt.close()



# Fig0015-3: Peak rate divided by path type, maze 1 ----------------------------------------------------------------------------------
idx = np.where(Data['Maze Type'] == 'Maze 1')[0]
a = len(idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = np.concatenate([Data['Training Day'][idx], Data['Training Day'][idx]]),
             y = np.concatenate([Data['peak_rate'][idx], Data['peak_rate_on_path'][idx]]),
             hue = np.concatenate([np.repeat('Entire Path', len(Data['peak_rate'][idx])),
                                   Data['path_type'][idx]]), ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze 1', loc = 'upper left')
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Peak Rate / Hz', fontsize = fs)
#ax.set_title('Simple Maze('+str(a)+' cells), Maze 1', fontsize = fs)
ax.axis([-0.5,8.5,0,12])
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8], labels = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.set_yticks(np.linspace(0,12,7))
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(figpath,code_id+'-3-Peak Rate divided by path-SimpleMaze-Maze1.svg'), dpi = 600)
plt.savefig(os.path.join(figpath,code_id+'-3-Peak Rate divided by path-SimpleMaze-Maze1.png'), dpi = 600)
plt.close()




# Fig0015-4: Peak rate divided by path type, maze 2. ------------------------------------------------------------------------------------
idx = np.where(Data['Maze Type'] == 'Maze 2')[0]
a = len(idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = np.concatenate([Data['Training Day'][idx], Data['Training Day'][idx]]),
             y = np.concatenate([Data['peak_rate'][idx], Data['peak_rate_on_path'][idx]]),
             hue = np.concatenate([np.repeat('Entire Path', len(Data['peak_rate'][idx])),
                                   Data['path_type'][idx]]), ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze 2', loc = 'upper left')
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
plt.xticks(ticks = [0,1,2,3,4,5,6,7], labels = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8'])
ax.set_ylabel('Peak Rate / Hz', fontsize = fs)
#ax.set_title('Simple Maze('+str(a)+' cells), Maze 2', fontsize = fs)
ax.axis([-0.5,7.5,0,16])
ax.set_yticks(np.linspace(0,16,9))
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(figpath,code_id+'-4-Peak Rate divided by path-SimpleMaze-Maze2.svg'), dpi = 600)
plt.savefig(os.path.join(figpath,code_id+'-4-Peak Rate divided by path-SimpleMaze-Maze2.png'), dpi = 600)
plt.close()




# Fig0015-5: Mean rate divided by path type, maze 1. ----------------------------------------------------------------------------------
idx = np.where(Data['Maze Type'] == 'Maze 1')[0]
a = len(idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = np.concatenate([Data['Training Day'][idx], Data['Training Day'][idx]]),
             y = np.concatenate([Data['mean_rate'][idx], Data['mean_rate_on_path'][idx]]),
             hue = np.concatenate([np.repeat('Entire Path', len(Data['mean_rate'][idx])),
                                   Data['path_type'][idx]]), ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze 1', loc = 'lower left')
ax.xaxis.set_major_locator(x)
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8], labels = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Mean Rate / Hz', fontsize = fs)
#ax.set_title('Simple Maze('+str(a)+' cells), Maze 1', fontsize = fs)
ax.axis([-0.5,8.5,0,1])
ax.set_yticks(np.linspace(0,1,6))
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(figpath,code_id+'-5-Mean Rate divided by path-SimpleMaze-Maze1.svg'), dpi = 600)
plt.savefig(os.path.join(figpath,code_id+'-5-Mean Rate divided by path-SimpleMaze-Maze1.png'), dpi = 600)
plt.close()




# Fig0015-6: Mean rate divided by path type, maze 2. ------------------------------------------------------------------------------------
idx = np.where(Data['Maze Type'] == 'Maze 2')[0]
a = len(idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = np.concatenate([Data['Training Day'][idx], Data['Training Day'][idx]]),
             y = np.concatenate([Data['mean_rate'][idx], Data['mean_rate_on_path'][idx]]),
             hue = np.concatenate([np.repeat('Entire Path', len(Data['mean_rate'][idx])),
                                   Data['path_type'][idx]]), ax = ax)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze 2', loc = 'lower left')
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Meak Rate / Hz', fontsize = fs)
#ax.set_title('Simple Maze('+str(a)+' cells), Maze 2', fontsize = fs)
ax.axis([-0.5,7.5,0,1])
plt.xticks(ticks = [0,1,2,3,4,5,6,7], labels = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8'])
ax.set_yticks(np.linspace(0,1,6))
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(figpath,code_id+'-6-Mean Rate divided by path-SimpleMaze-Maze2.svg'), dpi = 600)
plt.savefig(os.path.join(figpath,code_id+'-6-Mean Rate divided by path-SimpleMaze-Maze2.png'), dpi = 600)
plt.close()