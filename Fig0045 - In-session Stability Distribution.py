from mylib.statistic_test import *

code_id = '0045 - In-session Stability Distribution'
loc = join(figpath, code_id)
mkdir(loc)

lines = np.where((f1['date'] >= 20220820))[0]
if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['OEC', 'FSC'], 
                              f = f1, function = Fig0045_Interface, file_idx=lines,
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
        
fig = plt.figure(figsize=(4,3))
colors = sns.color_palette("rocket", 1)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
a = ax.hist(Data['OEC'], bins = 50, color=colors)[0]
y_max = np.nanmax(a)
ax.set_xticks(np.linspace(-0.2,1,7))
ax.set_yticks(ColorBarsTicks(y_max, is_auto=True, tick_number=5))
plt.savefig(join(loc, '[OEC] Place Cell In-session stability.png'), dpi = 2400)
plt.savefig(join(loc, '[OEC] Place Cell In-session stability.svg'), dpi = 2400)
plt.close()

fig = plt.figure(figsize=(4,3))
colors = sns.color_palette("rocket", 1)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
a = ax.hist(Data['FSC'], bins = 50, color=colors)[0]
y_max = np.nanmax(a)
ax.set_xticks(np.linspace(-0.2,1,7))
ax.set_yticks(ColorBarsTicks(y_max, is_auto=True, tick_number=5))
plt.savefig(join(loc, '[FSC] Place Cell In-session stability.png'), dpi = 2400)
plt.savefig(join(loc, '[FSC] Place Cell In-session stability.svg'), dpi = 2400)
plt.close()
    