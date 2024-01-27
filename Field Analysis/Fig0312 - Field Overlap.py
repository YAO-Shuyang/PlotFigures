from mylib.statistic_test import *

code_id = '0312 - Field Overlap'
loc = os.path.join(figpath, code_id)
mkdir(loc)

idx = np.where((f_CellReg_day['maze_type'] != 0))[0]
if os.path.exists(os.path.join(figdata,code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Start Session', 'Interval', 'Cell Pair Number',
                                             'Turn-On Proportion', 'Turn-Off Proportion', 
                                             'Kept-On Proportion', 'Kept-Off Proportion', 
                                             'Prev Field Number', 'Next Field Number', 
                                             'Field-On Proportion', 'Field-Off Proportion', 
                                             'Field-Kept Proportion', 'Data Type'], f = f_CellReg_day, 
                              function = PlaceFieldOverlapProportion_Interface,
                              file_name = code_id, behavior_paradigm = 'CellReg CrossMaze')
else:
    with open(os.path.join(figdata,code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

#(Data['Stage'] == 'Stage 1')&
idx = np.where((Data['MiceID'] == 10227)&(Data['Maze Type'] == 'Maze 1')&(Data['Data Type'] == 'Data'))[0]
SubData = SubDict(Data, Data.keys(), idx)
print(SubData)
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x='Start Session', 
    y='Field-Kept Proportion',
    data=SubData,
    hue='Interval',
    ax=ax,
)
ax.set_ylim(0, 100)
plt.show()