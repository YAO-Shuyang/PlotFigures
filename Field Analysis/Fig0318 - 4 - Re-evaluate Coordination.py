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

idx = np.where((((Data['Maze Type'] != 'Open Field')&(Data['Paradigm'] == 'CrossMaze'))|(Data['Paradigm'] != 'CrossMaze'))&
               (np.isnan(Data['Chi-Square Statistic']) == False)&(Data['Dimension'] == 2)&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

fig = plt.figure(figsize=(2, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'hue',
    y = 'Chi-Square Statistic',
    data = SubData,
    hue = 'Pair Type',
    palette=['#003366', '#0099CC'],
    width=0.8,
    capsize=0.3,
    err_kws={"linewidth": 0.5, 'color': 'black'},
    ax=ax
)
sns.stripplot(
    x='hue',
    y='Chi-Square Statistic',
    data=SubData,
    hue='Pair Type',
    palette=['#F2E8D4', '#D4C9A8'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    jitter=0.2,
    dodge=True,
    ax=ax
)
ax.semilogy()
ax.set_ylim(0.1, 2000)
plt.savefig(join(loc, '[dim=2] Chi-Square Statistic.png'), dpi = 600)
plt.savefig(join(loc, '[dim=2] Chi-Square Statistic.svg'), dpi = 600)
plt.close()
