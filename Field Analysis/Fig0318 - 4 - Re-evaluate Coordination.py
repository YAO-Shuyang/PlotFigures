from mylib.statistic_test import *

code_id = "0318 - Independence test for field evolution events"
loc = join(figpath, code_id)
mkdir(loc)


idx = np.where((f_CellReg_modi['Type'] == 'Real')&(f_CellReg_modi['maze_type'] != 0))[0]
if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(
        variable_names = ['Training Session', 'Chi-Square Statistic', 'MI', 'Dimension', 'Pair Type', 'Pair Num',
                          'Paradigm'],
        f = f_CellReg_modi, f_member=['Type'], func_kwgs={"if_consider_distance": True, "dis_thre": 0.5},
        function = IndependentEvolution_Interface, 
        file_name = code_id, file_idx = idx,
        behavior_paradigm = 'CrossMaze'
    )

Data['hue'] = np.array([Data['Maze Type'][i]+Data['Paradigm'][i] for i in range(Data['Type'].shape[0])])    
print(np.unique(Data['Pair Type']))
print("Coordination")
for i in np.unique(Data['hue']):
    idx1 = np.where((Data['Pair Type'] == 'Sibling') & (Data['hue'] == i))[0]
    idx2 = np.where((Data['Pair Type'] == 'Non-sibling') & (Data['hue'] == i))[0]
    print(i, ttest_rel(Data['Chi-Square Statistic'][idx1], Data['Chi-Square Statistic'][idx2]))
print()

idx = np.where((((Data['Maze Type'] != 'Open Field')&(Data['Paradigm'] == 'CrossMaze'))|(Data['Paradigm'] != 'CrossMaze'))&
               (np.isnan(Data['Chi-Square Statistic']) == False)&(Data['Dimension'] == 2)&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.boxplot(
    x="hue", 
    y="Chi-Square Statistic",
    hue="Pair Type",
    data=SubData,
    palette=['#457B9D', '#A8DADC'],
    ax = ax,
    linecolor='black',
    linewidth=0.5,
    gap=0.2,
    flierprops={'markersize': 0.5},
)
ax.semilogy()
ax.set_ylim(0.1, 2000)
plt.savefig(join(loc, '[dim=2] Chi-Square Statistic.png'), dpi = 600)
plt.savefig(join(loc, '[dim=2] Chi-Square Statistic.svg'), dpi = 600)
plt.close()
