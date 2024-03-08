from mylib.statistic_test import *
from mylib.field.field_tracker import indept_test_for_evolution_events

code_id = "0318 - Independence test for field evolution events"
loc = join(figpath, code_id)
mkdir(loc)

Num = {
    ('CrossMaze', 'Real', 1): 20998,
    ('CrossMaze', 'Real', 2): 25704,
    ('CrossMaze', 'Real', 0): 3861,
    ('CrossMaze', 'Shuffle', 1): 178269,
    ('CrossMaze', 'Shuffle', 2): 226816,
    ('CrossMaze', 'Shuffle', 0): 12853,
    ('HairpinMaze cis', 'Real', 3): 7713,
    ('HairpinMaze trs', 'Real', 3): 6324,
    ('HairpinMaze cis', 'Shuffle', 3): 38507,
    ('HairpinMaze trs', 'Shuffle', 3): 23103,
    ('ReverseMaze cis', 'Real', 1): 1622,
    ('ReverseMaze trs', 'Real', 1): 2876,
    ('ReverseMaze cis', 'Shuffle', 1): 11185,
    ('ReverseMaze trs', 'Shuffle', 1): 14526
}

"""
idx = np.where(f_CellReg_modi['Type'] == 'Real')[0]
if os.path.exists(join(figdata, code_id+' [over days].pkl')):
    with open(join(figdata, code_id+' [over days].pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(
        variable_names = ['Training Session', 'Chi-Square Statistic', 'MI', 'Dimension', 'Pair Type', 'Pair Num',
                          'Paradigm'],
        f = f_CellReg_modi, f_member=['Type'], file_idx=idx,
        function = IndependentEvolution_Interface, 
        func_kwgs={"N":Num},
        file_name = code_id+' [over days]', 
        behavior_paradigm = 'CrossMaze'
    )
"""
if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(
        variable_names = ['Training Session', 'Chi-Square Statistic', 'MI', 'Dimension', 'Pair Type', 'Pair Num',
                          'Paradigm'],
        f = f_CellReg_modi, f_member=['Type'],
        function = IndependentEvolution_Interface, 
        file_name = code_id, 
        behavior_paradigm = 'CrossMaze'
    )

Data['hue'] = np.array([Data['Type'][i]+Data['Pair Type'][i]+str(Data['Dimension'][i]) for i in range(Data['Type'].shape[0])])    




for maze in ['Open Field', 'Maze 1', 'Maze 2']:
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
    for i in range(4):    
        ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Chi-Square Statistic']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Type'] == 'Real')&
               (Data['Dimension'] == i+2)&
               (Data["Training Session"]+Data['Dimension'] <= 14))[0]
        SubData = SubDict(Data, Data.keys(), idx=idx)
        sns.lineplot(
            x=SubData["Training Session"]-1, 
            y=SubData["Chi-Square Statistic"], 
            hue="hue", 
            palette=['#003366', '#0099CC'],
            data=SubData, 
            ax=ax
        )
        sns.stripplot(
            x='Training Session',
            y='Chi-Square Statistic',
            data=SubData,
            hue = 'hue',
            palette=['#F2E8D4', '#8E9F85'],
            hue_order=['RealSibling', 'RealNon-sibling'],
            edgecolor='black',
            size=1,
            linewidth=0.10,
            jitter=0.2,
            ax=ax
        )
    plt.savefig(join(loc, f"Chi2 - {maze} Real.png"), dpi = 600)
    plt.savefig(join(loc, f"Chi2 - {maze} Real.svg"), dpi = 600)
    plt.close()
    

for maze in ['Open Field', 'Maze 1', 'Maze 2']:
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
    for i in range(4):    
        ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Chi-Square Statistic']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Type'] == 'Real')&
               (Data['Dimension'] == i+2))[0]
        SubData = SubDict(Data, Data.keys(), idx=idx)
        sns.lineplot(
            x=SubData["Training Session"]-1, 
            y=SubData["Pair Num"], 
            palette=['#003366', '#0099CC'],
            data=SubData, 
            ax=ax
        )
        sns.stripplot(
            x='Training Session',
            y='Pair Num',
            data=SubData,
            palette=['#F2E8D4', '#8E9F85'],
            hue_order=['RealSibling', 'RealNon-sibling'],
            edgecolor='black',
            size=1,
            linewidth=0.10,
            jitter=0.2,
            ax=ax
        )
    plt.savefig(join(loc, f"Pair Num - {maze} Real.png"), dpi = 600)
    plt.savefig(join(loc, f"Pair Num - {maze} Real.svg"), dpi = 600)
    plt.close()
    
for maze in ['Open Field', 'Maze 1', 'Maze 2']:
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
    for i in range(4):    
        ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['MI']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Type'] == 'Real')&
               (Data['Dimension'] == i+2))[0]
        SubData = SubDict(Data, Data.keys(), idx=idx)
        sns.lineplot(
            x=SubData["Training Session"]-1, 
            y=SubData["MI"], 
            hue=SubData["hue"], 
            palette=['#003366', '#0099CC'],
            ax=ax
        )
        sns.stripplot(
            x='Training Session',
            y='MI',
            data=SubData,
            hue = 'hue',
            palette=['#F2E8D4', '#8E9F85'],
            hue_order=['RealSibling', 'RealNon-sibling'],
            edgecolor='black',
            size=1,
            linewidth=0.10,
            jitter=0.2,
            ax=ax
        )
    plt.savefig(join(loc, f"MI - {maze} Real.png"), dpi = 600)
    plt.savefig(join(loc, f"MI - {maze} Real.svg"), dpi = 600)
    plt.close()
    
    
# Other Paradigm.
for maze in ['Open Field', 'Maze 1', 'Maze 2']:
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
    for i in range(4):    
        ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Chi-Square Statistic']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Type'] == 'Real')&
               (Data['Dimension'] == i+2)&
               (Data["Training Session"]+Data['Dimension'] <= 14))[0]
        SubData = SubDict(Data, Data.keys(), idx=idx)
        sns.lineplot(
            x=SubData["Training Session"]-1, 
            y=SubData["Chi-Square Statistic"], 
            hue="hue", 
            palette=['#003366', '#0099CC'],
            data=SubData, 
            ax=ax
        )
        sns.stripplot(
            x='Training Session',
            y='Chi-Square Statistic',
            data=SubData,
            hue = 'hue',
            palette=['#F2E8D4', '#8E9F85'],
            hue_order=['RealSibling', 'RealNon-sibling'],
            edgecolor='black',
            size=1,
            linewidth=0.10,
            jitter=0.2,
            ax=ax
        )
    plt.savefig(join(loc, f"Chi2 - {maze} Real.png"), dpi = 600)
    plt.savefig(join(loc, f"Chi2 - {maze} Real.svg"), dpi = 600)
    plt.close()
    

for maze in ['Open Field', 'Maze 1', 'Maze 2']:
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
    for i in range(4):    
        ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Chi-Square Statistic']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Type'] == 'Real')&
               (Data['Dimension'] == i+2))[0]
        SubData = SubDict(Data, Data.keys(), idx=idx)
        sns.lineplot(
            x=SubData["Training Session"]-1, 
            y=SubData["Pair Num"], 
            palette=['#003366', '#0099CC'],
            data=SubData, 
            ax=ax
        )
        sns.stripplot(
            x='Training Session',
            y='Pair Num',
            data=SubData,
            palette=['#F2E8D4', '#8E9F85'],
            hue_order=['RealSibling', 'RealNon-sibling'],
            edgecolor='black',
            size=1,
            linewidth=0.10,
            jitter=0.2,
            ax=ax
        )
    plt.savefig(join(loc, f"Pair Num - {maze} Real.png"), dpi = 600)
    plt.savefig(join(loc, f"Pair Num - {maze} Real.svg"), dpi = 600)
    plt.close()
    
for maze in ['Open Field', 'Maze 1', 'Maze 2']:
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
    for i in range(4):    
        ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['MI']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Type'] == 'Real')&
               (Data['Dimension'] == i+2))[0]
        SubData = SubDict(Data, Data.keys(), idx=idx)
        sns.lineplot(
            x=SubData["Training Session"]-1, 
            y=SubData["MI"], 
            hue=SubData["hue"], 
            palette=['#003366', '#0099CC'],
            ax=ax
        )
        sns.stripplot(
            x='Training Session',
            y='MI',
            data=SubData,
            hue = 'hue',
            palette=['#F2E8D4', '#8E9F85'],
            hue_order=['RealSibling', 'RealNon-sibling'],
            edgecolor='black',
            size=1,
            linewidth=0.10,
            jitter=0.2,
            ax=ax
        )
    plt.savefig(join(loc, f"MI - {maze} Real.png"), dpi = 600)
    plt.savefig(join(loc, f"MI - {maze} Real.svg"), dpi = 600)
    plt.close()