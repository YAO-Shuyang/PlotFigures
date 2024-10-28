from mylib.statistic_test import *
from mylib.field.field_tracker import indept_test_for_evolution_events, compute_joint_probability_matrix

code_id = "0323 - Coordinatedly Drift - Analysis of Joint Prob Matrix"
loc = join(figpath, code_id)
mkdir(loc)


if __name__ == '__main__':
    from tqdm import tqdm

idx = np.where((f_CellReg_modi['Type'] == 'Real')&(f_CellReg_modi['maze_type'] != 0))[0]
if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(
        variable_names = ['Training Session', 'delta-P', 'Dimension', 'Axis', 'Pair Type',
                          'Paradigm', 'X'],
        f = f_CellReg_modi, f_member=['Type'],
        function = CoordinatedDrift_Interface, func_kwgs={"dis_thre": 0.5},
        file_name = code_id, file_idx=idx,
        behavior_paradigm = 'CrossMaze'
    )
    
Data['hue'] = np.array([Data['Maze Type'][i]+Data['Paradigm'][i]+Data['Pair Type'][i] for i in range(Data['Type'].shape[0])])    

idx = np.where((((Data['Maze Type'] != 'Open Field')&(Data['Paradigm'] == 'CrossMaze'))|(Data['Paradigm'] != 'CrossMaze'))&
               (np.isnan(Data['delta-P']) == False)&(Data['Dimension'] == 2)&(Data['Axis'] == 'IP axis')&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

colors = sns.color_palette("Reds", 2) + sns.color_palette("Oranges", 2) + sns.color_palette("Wistia", 2) + sns.color_palette("Greens", 2) + sns.color_palette("Blues", 2) + sns.color_palette("Purples", 2)

fig = plt.figure(figsize=(10, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
box = sns.boxplot(
    x="X", 
    y="delta-P",
    hue="hue",
    data=SubData,
    palette=colors,
    ax = ax,
    linecolor='black',
    linewidth=0.5,
    gap=0.2,
    flierprops={'markersize': 0.5},
)
for line in box.patches:
    line.set_linewidth(0)   # Remove the outer box line
ax.set_xlim([-0.5, 2.5])
ax.set_ylim([-0.06, 0.06])
ax.set_yticks(np.linspace(-0.06, 0.06, 7))
plt.savefig(join(loc, '[dim=2] delt_P All Paradigm.png'), dpi = 600)
plt.savefig(join(loc, '[dim=2] delt_P All Paradigm.svg'), dpi = 600)
plt.show()