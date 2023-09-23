# Because we find that place cells exhibit a multifield spatial code, to further identify the distribution of fields is really necessary.
# PeakCurve has partially demonstated the distribution of place fields, though it only takes main field into concerns. However Just like what have been mentioned above,
#   the existence of multifield spatial code indicate that main fields are only small part of the whole collection of place fields, and this provides a justification to 
#   further analysis the subfields.

# Besides, we have observed this phenomenone from the cross day recorded data: Some place cells with multifields are recorded in the successive 6 sessions (fr. Aug 20
#   to Aug 30, 2022; 11095; Maze 1; See in Fig0001, line 4 and 6, for example), and they fired with similar firing pattern in the 6 days, that is similar number and
#   location of place fields. However, the peak firing rate of each subfield(for the convience to discuss, it includes main field) of certain cell may switch between each
#   other, that is, the original main field recorded in Day A may loss its superiority in Day B while the subfield recorded in Day A may become a main field in Day B.
# This phenomenone inplies that there may not be a significant differences between the properties, at least the peak firing rate, of main fields and subfields. So we
#   have not reason to analysis main field only.

# Now we combine all the fields together and have a look at what will happen.

from mylib.statistic_test import *

code_id = '0029 - Field Centers To Start - Distribution'
loc = os.path.join(figpath, code_id)
mkdir(loc)


if os.path.exists(os.path.join(figdata, code_id+'.pkl')):
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    indices = np.where(f1['maze_type'] != 0)[0]
    Gs = [Graph(12, 12, cp.deepcopy(maze_graphs[(1, 12)])), Graph(12, 12, cp.deepcopy(maze_graphs[(2, 12)]))]
    Data = DataFrameEstablish(variable_names = ['Distance To Start', 'Cell'],func_kwgs={'Gs':Gs},
                              f = f1, function = FieldCentersToStart_Interface, file_idx=indices,
                              file_name = code_id, behavior_paradigm = 'CrossMaze')

Data['Distance To Start'] = Data['Distance To Start'] * 8
x_ticks = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']   
familiar_ticks = ['Day '+str(i) for i in range(7, 10)] + ['>=Day 10']
def plot_field_distribution(ax, Data, mice, maze_type, stage):
    idx = np.where((Data['MiceID'] == str(mice))&(Data['Maze Type'] == f"Maze {maze_type}")&(Data['Stage'] == stage))[0]
    SubData = SubDict(Data, keys=Data.keys(), idx=idx)
    idx = np.concatenate([np.where(SubData['Training Day'] == day)[0] for day in x_ticks])
    SubData = SubDict(SubData, keys=SubData.keys(), idx=idx)
    
    ax.hist(
        SubData['Distance To Start'],
        bins = 60,
        range=(0, 600),
        density = True,
        alpha = 0.6,
        rwidth = 0.8,
    )
    ax.axis([0, 600, 0, 0.005])
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=2)

fig, axes = plt.subplots(ncols=1, nrows=4, figsize=(8,12))
ax1, ax2, ax3 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True), Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True), Clear_Axes(axes[2], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
plot_field_distribution(ax1, Data, 10209, 1, 'Stage 2')
plot_field_distribution(ax2, Data, 10212, 1, 'Stage 2')
plot_field_distribution(ax3, Data, 11095, 1, 'Stage 2')
plt.show()
