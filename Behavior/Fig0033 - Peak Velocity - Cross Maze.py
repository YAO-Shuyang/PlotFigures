from mylib.statistic_test import *
from brokenaxes import brokenaxes

code_id = '0033 - Peak Velocity'


if os.path.exists(os.path.join(figdata,code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Cell', 'velocity'], f = f1, function = PeakVelocity_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze', func_kwgs = {'is_placecell':False})
else:
    with open(os.path.join(figdata,code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

p = os.path.join(figpath, code_id)
mkdir(p)

x = MultipleLocator(1)
cell_number = Data['Maze Type'].shape[0]
fs = 12

# Fig0033-1: mice's velocity at where the peak rate locate vs training day. ---------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = 'Training Day',y = 'velocity', data = Data, hue = 'Maze Type', ax = ax, err_style = 'bars', err_kws = {'elinewidth':1, 'capsize':3, 'capthick':1})
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze Type', loc = 'upper center', fontsize = 8, title_fontsize = 9, ncol = 3)
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Velocity / (cm/s)', fontsize = fs)
ax.axhline(6, ls = ':', color = "orange")
ax.axhline(4, ls = ':', color = "limegreen")
ax.set_yticks(np.linspace(6,16,6))
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9,10], labels = ['P1','P2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.axis([-0.5,10.5,6,16])
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(p,code_id+'-1-Velocity at Peak Rate-CrossMaze.svg'), dpi = 600)
plt.savefig(os.path.join(p,code_id+'-1-Velocity at Peak Rate-CrossMaze.png'), dpi = 600)
plt.close()

# Mouse #11095
# Fig0033-1: mice's velocity at where the peak rate locate vs training day. ---------------------------------------------------------------------------------------------------------
idx = np.where(Data['MiceID'] == '11095')[0]
SubData = DivideData(Data, idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = 'Training Day',y = 'velocity', data = SubData, hue = 'Maze Type', ax = ax, err_style = 'bars', err_kws = {'elinewidth':1, 'capsize':3, 'capthick':1})
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze Type', loc = 'upper center', fontsize = 8, title_fontsize = 9, ncol = 3)
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Velocity / (cm/s)', fontsize = fs)
ax.axhline(6, ls = ':', color = "orange")
ax.axhline(4, ls = ':', color = "limegreen")
ax.set_yticks(np.linspace(6,16,6))
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9,10], labels = ['P1','P2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.axis([-0.5,10.5,6,16])
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(p,code_id+'-1-Velocity at Peak Rate-CrossMaze.svg'), dpi = 600)
plt.savefig(os.path.join(p,code_id+'-1-Velocity at Peak Rate-CrossMaze.png'), dpi = 600)
plt.close()

# Mouse #11092
# Fig0033-3: mice's velocity at where the peak rate locate vs training day. ---------------------------------------------------------------------------------------------------------
idx = np.where(Data['MiceID'] == '11092')[0]
SubData = DivideData(Data, idx)
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = 'Training Day',y = 'velocity', data = SubData, hue = 'Maze Type', ax = ax, err_style = 'bars', err_kws = {'elinewidth':1, 'capsize':3, 'capthick':1})
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze Type', loc = 'upper center', fontsize = 8, title_fontsize = 9, ncol = 3)
ax.xaxis.set_major_locator(x)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Velocity / (cm/s)', fontsize = fs)
ax.axhline(6, ls = ':', color = "orange")
ax.axhline(4, ls = ':', color = "limegreen")
ax.set_yticks(np.linspace(6,16,6))
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9,10], labels = ['P1','P2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
ax.axis([-0.5,10.5,6,16])
ax.xaxis.set_major_locator(x)
plt.tight_layout()
plt.savefig(os.path.join(p,code_id+'-1-Velocity at Peak Rate-CrossMaze.svg'), dpi = 600)
plt.savefig(os.path.join(p,code_id+'-1-Velocity at Peak Rate-CrossMaze.png'), dpi = 600)
plt.close()