from mylib.statistic_test import *
from mylib.field.field_tracker import indept_test_for_evolution_events

code_id = "0325 - Coordination Index"
loc = join(figpath, code_id)
mkdir(loc)

idx = np.where((f_CellReg_modi['Type'] == 'Real')&(f_CellReg_modi['maze_type'] != 0))[0]
if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(
        variable_names = ['Training Session', 'Coordinate Index', 'Dimension', 'Pair Type', 'Pair Num',
                          'Paradigm'],
        f = f_CellReg_modi, f_member=['Type'],
        function = CoordinateIndex_Interface, 
        file_name = code_id, file_idx = idx,
        behavior_paradigm = 'CrossMaze'
    )
    
Data['hue'] = np.array([Data['Type'][i]+Data['Pair Type'][i]+str(Data['Dimension'][i])+Data['Maze Type'][i]+Data['Paradigm'][i] for i in range(Data['Type'].shape[0])])   
Data['Coordinate Index'] = Data['Coordinate Index'] * 100 

# Maze A and B
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
for i in range(4):    
    ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((Data['Paradigm'] == 'CrossMaze')&
            (np.isnan(Data['Coordinate Index']) == False)&
            (Data['Maze Type'] != 'Open Field')&
            (Data['Type'] == 'Real')&
            (Data['Dimension'] == i+2)&
            (Data["Training Session"]+Data['Dimension'] <= 14))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    sns.lineplot(
        x=SubData["Training Session"]-1, 
        y=SubData["Coordinate Index"], 
        hue="hue", 
        palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
        data=SubData, 
        ax=ax,
        linewidth = 0.5,
        err_kws={"edgecolor":None},
        legend=False
    )
    sns.stripplot(
        x='Training Session',
        y='Coordinate Index',
        data=SubData,
        hue = 'hue',
        palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
        edgecolor='black',
        size=3,
        linewidth=0.15,
        jitter=0.2,
        ax=ax,
        legend=False
    )
    ax.set_ylim(0, 30)
plt.savefig(join(loc, f"Coordinated Index - Maze A&B Real.png"), dpi = 600)
plt.savefig(join(loc, f"Coordinated Index - Maze A&B Real.svg"), dpi = 600)
plt.close()


print("Maze A & B CI Test Across Days:")
for dt in range(2, 6):
    for i in range(1, 15-dt):
        sib = np.where((Data['Maze Type'] == 'Maze 1')&
                       (Data['Type'] == 'Real')&
                       (Data['Dimension'] == dt)&
                       (Data["Paradigm"] == 'CrossMaze')&
                       (Data["Pair Type"] == 'Sibling')&
                       (Data["Training Session"] == i))[0]
        
        non = np.where((Data['Maze Type'] == 'Maze 1')&
                       (Data['Type'] == 'Real')&
                       (Data['Dimension'] == dt)&
                       (Data["Paradigm"] == 'CrossMaze')&
                       (Data["Pair Type"] == 'Non-sibling')&
                       (Data["Training Session"] == i))[0]
        print(f"    Maze 1, Dim {dt} - Day {i}", ttest_rel(Data['Coordinate Index'][sib], Data['Coordinate Index'][non]))
        sib = np.where((Data['Maze Type'] == 'Maze 2')&
                       (Data['Type'] == 'Real')&
                       (Data['Dimension'] == dt)&
                       (Data["Paradigm"] == 'CrossMaze')&
                       (Data["Pair Type"] == 'Sibling')&
                       (Data["Training Session"] == i))[0]
        
        non = np.where((Data['Maze Type'] == 'Maze 2')&
                       (Data['Type'] == 'Real')&
                       (Data['Dimension'] == dt)&
                       (Data["Paradigm"] == 'CrossMaze')&
                       (Data["Pair Type"] == 'Non-sibling')&
                       (Data["Training Session"] == i))[0]
        print(f"    Maze 2, Dim {dt} - Day {i}", ttest_rel(Data['Coordinate Index'][sib], Data['Coordinate Index'][non]), end='\n\n')     
print()


# Other Paradigm
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
for i in range(4):    
    ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((Data['Paradigm'] != 'CrossMaze')&
               (np.isnan(Data['Coordinate Index']) == False)&
               (Data['Type'] == 'Real')&
               (Data['Dimension'] == i+2)&
               (Data["Training Session"]+Data['Dimension'] <= 8))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    sns.lineplot(
            x=SubData["Training Session"]-1, 
            y=SubData["Coordinate Index"], 
            hue="hue", 
            palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', 
                     '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
            data=SubData, 
            ax=ax,
            linewidth = 0.5,
            err_kws={"edgecolor":None},
        legend=False
    )
    sns.stripplot(
            x='Training Session',
            y='Coordinate Index',
            data=SubData,
            hue='hue',
            palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A', 
                     '#C3AED6', '#66C7B4', '#A7D8DE', '#F67280'],
            edgecolor='black',
            size=3,
            linewidth=0.15,
            jitter=0.2,
            ax=ax,
        legend=False
    )
plt.savefig(join(loc, f"[Reverse&Hairpin] Coordinate Index - Real.png"), dpi = 600)
plt.savefig(join(loc, f"[Reverse&Hairpin] Coordinate Index - Real.svg"), dpi = 600)
plt.close()