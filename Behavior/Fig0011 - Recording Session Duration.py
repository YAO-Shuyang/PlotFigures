from mylib.statistic_test import *

code_id = "0011 - Recording Duration"
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata,code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Duration'], 
                              f = f1, function = SessionDuration_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata,code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

idx = np.concatenate([np.where(Data['MiceID'] == m)[0] for m in ['11095', '11092', '10209', '10212']])
SubData = SubDict(Data, Data.keys(), idx)
colors = sns.color_palette("rocket", 3)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x='MiceID',
    y='Duration',
    data=SubData,
    hue='Maze Type',
    palette=colors,
    capsize=0.1,
    width=0.7,
    errcolor='black',
    errwidth=0.8
)
ax.set_ylim([0,45])
ax.set_yticks(np.linspace(0,40,5))
plt.savefig(join(loc, 'Recording Duration.png'), dpi=600)
plt.savefig(join(loc, 'Recording Duration.svg'), dpi=600)
plt.close()