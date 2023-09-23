from mylib.statistic_test import *

code_id = '0044 - In-session Population Vector'
loc = join(figpath, code_id)
mkdir(loc)

lines = np.where((f1['date'] >= 20220813)&(f1['date']!=20220814))[0]
if os.path.exists(os.path.join(figdata, code_id+'2.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['OEC', 'FSC'], 
                              f = f1, function = PVCorrelations2_Interface, file_idx=lines,
                              file_name = code_id+'2', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'2.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
        
fig = plt.figure(figsize=(4,3))
colors = sns.color_palette("rocket", 3)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(x='Training Day', y='OEC', data=Data, hue = 'Maze Type', palette=colors)
ax.set_yticks(np.linspace(0,0.7,8))
plt.savefig(join(loc, "Change of mean of Population vector.png"), dpi = 2400)
plt.savefig(join(loc, "Change of mean of Population vector.svg"), dpi = 2400)
plt.close()

Dates = ['Day '+str(i) for i in range(1,10)]
for d in Dates:
    idx1 = np.where((Data['Maze Type'] == 'Maze 1')&(Data['Training Day'] == d)&(np.isnan(Data['OEC']) == False))[0]
    idx2 = np.where((Data['Maze Type'] == 'Maze 2')&(Data['Training Day'] == d)&(np.isnan(Data['OEC']) == False))[0]
    idx0 = np.where((Data['Maze Type'] == 'Open Field')&(Data['Training Day'] == d)&(np.isnan(Data['OEC']) == False))[0]

    print(d)
    print("Open - Maze 2", ttest_ind(Data['OEC'][idx0], Data['OEC'][idx2]))
    print("Maze 1 - Maze 2", ttest_ind(Data['OEC'][idx1], Data['OEC'][idx2]))
    print("Open - Maze 1", ttest_ind(Data['OEC'][idx0], Data['OEC'][idx1]))
    print("Done.", end='\n\n')