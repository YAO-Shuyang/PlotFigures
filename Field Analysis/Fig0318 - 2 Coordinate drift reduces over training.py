from mylib.statistic_test import *
from mylib.field.field_tracker import indept_test_for_evolution_events
from scipy.stats import wilcoxon

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
        f = f_CellReg_modi, f_member=['Type'], func_kwgs={"if_consider_distance": True, "dis_thre": 0.5},
        function = IndependentEvolution_Interface, 
        file_name = code_id, 
        behavior_paradigm = 'CrossMaze'
    )

Data['hue'] = np.array([Data['Type'][i]+Data['Pair Type'][i]+str(Data['Dimension'][i])+Data['Maze Type'][i]+Data['Paradigm'][i] for i in range(Data['Type'].shape[0])])



idx = np.where((Data['Paradigm'] == 'CrossMaze')&
            (np.isnan(Data['Chi-Square Statistic']) == False)&
            (Data['Maze Type'] != 'Open Field')&
            (Data['Type'] == 'Real')&
            (Data['Pair Type'] == 'Sibling')&
            ((Data["Training Session"]+Data['Dimension'] == 13)|
             (Data['Training Session'] == 1)))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

for i in np.where(SubData['Training Session'] == 1)[0]:
    SubData['hue'][i] = 'First Session' + SubData['Maze Type'][i]
for i in np.where(SubData['Training Session'] != 1)[0]:
    SubData['hue'][i] = 'Last Session' + SubData['Maze Type'][i]
    
print(np.unique(SubData['hue']))

print("Cross day comparison of Chi2, Maze A and B:")
for i in range(4):
    print("    Maze A, Dim ", i+2, 
          ttest_rel(SubData['Chi-Square Statistic'][np.where((SubData['hue'] == 'First SessionMaze 1')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['Chi-Square Statistic'][np.where((SubData['hue'] == 'Last SessionMaze 1')&(SubData['Dimension'] == i+2))[0]]))
    print("    Maze B, Dim ", i+2, 
          ttest_rel(SubData['Chi-Square Statistic'][np.where((SubData['hue'] == 'First SessionMaze 2')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['Chi-Square Statistic'][np.where((SubData['hue'] == 'Last SessionMaze 2')&(SubData['Dimension'] == i+2))[0]]), end='\n\n')
    
fig = plt.figure(figsize=(5,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Dimension',
    y = 'Chi-Square Statistic',
    data = SubData,
    hue = 'hue',
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    width=0.8,
    capsize=0.1,
    errcolor='black',
    errwidth=0.5,
)
sns.stripplot(
    x='Dimension',
    y='Chi-Square Statistic',
    data=SubData,
    hue='hue',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    jitter=0.2,
    dodge=True,
    ax=ax
)
plt.savefig(join(loc, '[Crossday Comparison] Chi-Square Statistic.png'), dpi = 600)
plt.savefig(join(loc, '[Crossday Comparison] Chi-Square Statistic.svg'), dpi = 600)
plt.close()

print("Cross day comparison of Mutual information, Maze A and B:")
for i in range(4):
    print("    Maze A, Dim ", i+2, 
          ttest_rel(SubData['MI'][np.where((SubData['hue'] == 'First SessionMaze 1')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['MI'][np.where((SubData['hue'] == 'Last SessionMaze 1')&(SubData['Dimension'] == i+2))[0]]))
    print("    Maze B, Dim ", i+2, 
          ttest_rel(SubData['MI'][np.where((SubData['hue'] == 'First SessionMaze 2')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['MI'][np.where((SubData['hue'] == 'Last SessionMaze 2')&(SubData['Dimension'] == i+2))[0]]), end='\n\n')
    
fig = plt.figure(figsize=(5,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Dimension',
    y = 'MI',
    data = SubData,
    hue = 'hue',
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    width=0.8,
    capsize=0.1,
    errcolor='black',
    errwidth=0.5,
)
sns.stripplot(
    x='Dimension',
    y='MI',
    data=SubData,
    hue='hue',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    jitter=0.2,
    dodge=True,
    ax=ax
)
plt.savefig(join(loc, '[Crossday Comparison] MI.png'), dpi = 600)
plt.savefig(join(loc, '[Crossday Comparison] MI.svg'), dpi = 600)
plt.close()

idx = np.where((Data['Paradigm'] != 'CrossMaze')&
            (np.isnan(Data['Chi-Square Statistic']) == False)&
            (Data['Type'] == 'Real')&
            (Data['Pair Type'] == 'Sibling')&
            ((Data["Training Session"]+Data['Dimension'] == 8)|
             (Data['Training Session'] == 1)))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

for i in np.where(SubData['Training Session'] == 1)[0]:
    SubData['hue'][i] = 'First Session' + SubData['Paradigm'][i]
for i in np.where(SubData['Training Session'] != 1)[0]:
    SubData['hue'][i] = 'Last Session' + SubData['Paradigm'][i]
    
print(np.unique(SubData['hue']))

print("Cross day comparison of Chi2, Maze A and B:")
for i in range(4):
    print("    MAf, Dim ", i+2, 
          ttest_rel(SubData['Chi-Square Statistic'][np.where((SubData['hue'] == 'First SessionReverseMaze cis')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['Chi-Square Statistic'][np.where((SubData['hue'] == 'Last SessionReverseMaze cis')&(SubData['Dimension'] == i+2))[0]]))
    
    print("    MAb, Dim ", i+2, 
          ttest_rel(SubData['Chi-Square Statistic'][np.where((SubData['hue'] == 'First SessionReverseMaze trs')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['Chi-Square Statistic'][np.where((SubData['hue'] == 'Last SessionReverseMaze trs')&(SubData['Dimension'] == i+2))[0]]))
    
    print("    HPf, Dim ", i+2, 
          ttest_rel(SubData['Chi-Square Statistic'][np.where((SubData['hue'] == 'First SessionHairpinMaze cis')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['Chi-Square Statistic'][np.where((SubData['hue'] == 'Last SessionHairpinMaze cis')&(SubData['Dimension'] == i+2))[0]]))
    
    print("    HPb, Dim ", i+2, 
          ttest_rel(SubData['Chi-Square Statistic'][np.where((SubData['hue'] == 'First SessionHairpinMaze trs')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['Chi-Square Statistic'][np.where((SubData['hue'] == 'Last SessionHairpinMaze trs')&(SubData['Dimension'] == i+2))[0]]), end='\n\n')
     
fig = plt.figure(figsize=(5,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Dimension',
    y = 'Chi-Square Statistic',
    data = SubData,
    hue = 'hue',
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', 
             '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    width=0.8,
    capsize=0.05,
    errcolor='black',
    errwidth=0.5,
)
sns.stripplot(
    x='Dimension',
    y='Chi-Square Statistic',
    data=SubData,
    hue='hue',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A', 
             '#C3AED6', '#66C7B4', '#A7D8DE', '#F67280'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    jitter=0.2,
    dodge=True,
    ax=ax
)
plt.savefig(join(loc, '[Crossday Comparison] [Hairpin & Reverse] Chi-Square Statistic.png'), dpi = 600)
plt.savefig(join(loc, '[Crossday Comparison] [Hairpin & Reverse] Chi-Square Statistic.svg'), dpi = 600)
plt.close()

print("Cross day comparison of Mutual information, Hairpin & Reverse:")
for i in range(4):
    print("    MAf, Dim ", i+2, 
          ttest_rel(SubData['MI'][np.where((SubData['hue'] == 'First SessionReverseMaze cis')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['MI'][np.where((SubData['hue'] == 'Last SessionReverseMaze cis')&(SubData['Dimension'] == i+2))[0]]))
    
    print("    MAb, Dim ", i+2, 
          ttest_rel(SubData['MI'][np.where((SubData['hue'] == 'First SessionReverseMaze trs')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['MI'][np.where((SubData['hue'] == 'Last SessionReverseMaze trs')&(SubData['Dimension'] == i+2))[0]]))
    
    print("    HPf, Dim ", i+2, 
          ttest_rel(SubData['MI'][np.where((SubData['hue'] == 'First SessionHairpinMaze cis')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['MI'][np.where((SubData['hue'] == 'Last SessionHairpinMaze cis')&(SubData['Dimension'] == i+2))[0]]))
    
    print("    HPb, Dim ", i+2, 
          ttest_rel(SubData['MI'][np.where((SubData['hue'] == 'First SessionHairpinMaze trs')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['MI'][np.where((SubData['hue'] == 'Last SessionHairpinMaze trs')&(SubData['Dimension'] == i+2))[0]]), end='\n\n')
    
fig = plt.figure(figsize=(5,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Dimension',
    y = 'MI',
    data = SubData,
    hue = 'hue',
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', 
             '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    width=0.8,
    capsize=0.05,
    errcolor='black',
    errwidth=0.5,
)
sns.stripplot(
    x='Dimension',
    y='MI',
    data=SubData,
    hue='hue',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A', 
             '#C3AED6', '#66C7B4', '#A7D8DE', '#F67280'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    jitter=0.2,
    dodge=True,
    ax=ax
)
plt.savefig(join(loc, '[Crossday Comparison] [Hairpin & Reverse] MI.png'), dpi = 600)
plt.savefig(join(loc, '[Crossday Comparison] [Hairpin & Reverse] MI.svg'), dpi = 600)
plt.close()

# Maze A and B
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
ylims = [6000, 8000, 12000, 17000]
for i in range(4):    
    ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

    SubData = SubDict(Data, Data.keys(), idx=idx)
    sns.lineplot(
        x=SubData["Training Session"]-1, 
        y=SubData["Chi-Square Statistic"], 
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
        y='Chi-Square Statistic',
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
    ax.set_ylim(0, ylims[i])
plt.savefig(join(loc, f"Chi2 - Maze A&B Real.png"), dpi = 600)
plt.savefig(join(loc, f"Chi2 - Maze A&B Real.svg"), dpi = 600)
plt.close()


print("Maze A & B & Open Field Chi2 Statistic Test Across Days:")
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
        print(f"    Maze 1, Dim {dt} - Day {i}", ttest_rel(Data['Chi-Square Statistic'][sib], Data['Chi-Square Statistic'][non]))
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
        print(f"    Maze 2, Dim {dt} - Day {i}", ttest_rel(Data['Chi-Square Statistic'][sib], Data['Chi-Square Statistic'][non]), end='\n\n')
        """
        sib = np.where((Data['Maze Type'] == 'Open Field')&
                       (Data['Type'] == 'Real')&
                       (Data['Dimension'] == dt)&
                       (Data["Paradigm"] == 'CrossMaze')&
                       (Data["Pair Type"] == 'Sibling')&
                       (Data["Training Session"] == i))[0]
        
        non = np.where((Data['Maze Type'] == 'Open Field')&
                       (Data['Type'] == 'Real')&
                       (Data['Dimension'] == dt)&
                       (Data["Paradigm"] == 'CrossMaze')&
                       (Data["Pair Type"] == 'Non-sibling')&
                       (Data["Training Session"] == i))[0]
        print(f"    Open Field, Dim {dt} - Day {i}", ttest_rel(Data['Chi-Square Statistic'][sib], Data['Chi-Square Statistic'][non]), end='\n\n')
        """        
print()
   
print("Maze A & B & Open Field Mutual Information Test Across Days:")
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
        print(f"    Maze 1, Dim {dt} - Day {i}", ttest_rel(Data['MI'][sib], Data['MI'][non]))
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
        print(f"    Maze 2, Dim {dt} - Day {i}", ttest_rel(Data['MI'][sib], Data['MI'][non]), end='\n\n')
        """
        sib = np.where((Data['Maze Type'] == 'Open Field')&
                       (Data['Type'] == 'Real')&
                       (Data['Dimension'] == dt)&
                       (Data["Paradigm"] == 'CrossMaze')&
                       (Data["Pair Type"] == 'Sibling')&
                       (Data["Training Session"] == i))[0]
        
        non = np.where((Data['Maze Type'] == 'Open Field')&
                       (Data['Type'] == 'Real')&
                       (Data['Dimension'] == dt)&
                       (Data["Paradigm"] == 'CrossMaze')&
                       (Data["Pair Type"] == 'Non-sibling')&
                       (Data["Training Session"] == i))[0]
        print(f"    Open Field, Dim {dt} - Day {i}", ttest_rel(Data['MI'][sib], Data['MI'][non]), end='\n\n')
        """
print()

ylims = [0.17, 0.2, 0.22, 0.25]
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
for i in range(4):    
    ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((Data['Paradigm'] == 'CrossMaze')&
           (np.isnan(Data['Chi-Square Statistic']) == False)&
           (Data['Maze Type'] != 'Open Field')&
           (Data['Type'] == 'Real')&
           (Data['Dimension'] == i+2)&
           (Data["Training Session"]+Data['Dimension'] <= 14))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    sns.lineplot(
        x=SubData["Training Session"]-1, 
        y=SubData["Pair Num"], 
        palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
        data=SubData, 
        hue=SubData['hue'],
        ax=ax,
        linewidth = 0.5,
        err_kws={"edgecolor":None},
        legend=False
    )
    sns.stripplot(
        x='Training Session',
        y='Pair Num',
        data=SubData,
        hue='hue',
        palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
        edgecolor='black',
        size=3,
        linewidth=0.15,
        jitter=0.2,
        ax=ax,
        legend=False
    )
    ax.set_ylim(0, int(np.max(SubData['Pair Num'])//10000 + 1)*10000)

plt.savefig(join(loc, f"Pair Num - Maze A&B Real.png"), dpi = 600)
plt.savefig(join(loc, f"Pair Num - Maze A&B Real.svg"), dpi = 600)
plt.close()
    
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
for i in range(4):    
    ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['MI']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Type'] == 'Real')&
               (Data['Dimension'] == i+2)&
               (Data["Training Session"]+Data['Dimension'] <= 14))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    sns.lineplot(
            x=SubData["Training Session"]-1, 
            y=SubData["MI"], 
            hue=SubData["hue"], 
            palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
            ax=ax,
            linewidth = 0.5,
            err_kws={"edgecolor":None},
        legend=False
    )
    sns.stripplot(
            x='Training Session',
            y='MI',
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
    ax.set_ylim(0, ylims[i])
    plt.savefig(join(loc, f"MI - Maze A&B Real.png"), dpi = 600)
plt.savefig(join(loc, f"MI - Maze A&B Real.svg"), dpi = 600)
plt.close()
    
"""
# Open Field
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
for i in range(4):    
    ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((Data['Paradigm'] == 'CrossMaze')&
            (np.isnan(Data['Chi-Square Statistic']) == False)&
            (Data['Maze Type'] == 'Open Field')&
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
        ax=ax,
        linewidth = 0.5,
        err_kws={"edgecolor":None}
    )
    sns.stripplot(
        x='Training Session',
        y='Chi-Square Statistic',
        data=SubData,
        hue = 'hue',
        palette=['#F2E8D4', '#D4C9A8'],
        edgecolor='black',
        size=3,
        linewidth=0.15,
        jitter=0.2,
        ax=ax
    )
plt.savefig(join(loc, f"[Open Field] Chi2 - Real.png"), dpi = 600)
plt.savefig(join(loc, f"[Open Field] Chi2 - Real.svg"), dpi = 600)
plt.close()
    

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
for i in range(4):    
    ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((Data['Paradigm'] == 'CrossMaze')&
           (np.isnan(Data['Chi-Square Statistic']) == False)&
           (Data['Maze Type'] == 'Open Field')&
           (Data['Type'] == 'Real')&
           (Data['Dimension'] == i+2)&
           (Data["Training Session"]+Data['Dimension'] <= 14))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    sns.lineplot(
        x=SubData["Training Session"]-1, 
        y=SubData["Pair Num"], 
        palette=['#003366', '#0099CC'],
        data=SubData, 
        hue=SubData['hue'],
        ax=ax,
        linewidth = 0.5,
        err_kws={"edgecolor":None}
    )
    sns.stripplot(
        x='Training Session',
        y='Pair Num',
        data=SubData,
        hue='hue',
        palette=['#F2E8D4', '#D4C9A8'],
        edgecolor='black',
        size=3,
        linewidth=0.15,
        jitter=0.2,
        ax=ax
    )
    ax.set_ylim(0, int(np.max(SubData['Pair Num'])//10000 + 1)*10000)
plt.savefig(join(loc, f"[Open Field] Pair Num - Real.png"), dpi = 600)
plt.savefig(join(loc, f"[Open Field] Pair Num - Real.svg"), dpi = 600)
plt.close()
    
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
for i in range(4):    
    ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['MI']) == False)&
               (Data['Maze Type'] == 'Open Field')&
               (Data['Type'] == 'Real')&
               (Data['Dimension'] == i+2)&
               (Data["Training Session"]+Data['Dimension'] <= 14))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    sns.lineplot(
            x=SubData["Training Session"]-1, 
            y=SubData["MI"], 
            hue=SubData["hue"], 
            palette=['#003366', '#0099CC'],
            ax=ax,
            linewidth = 0.5,
            err_kws={"edgecolor":None}
    )
    sns.stripplot(
            x='Training Session',
            y='MI',
            data=SubData,
            hue = 'hue',
            palette=['#F2E8D4', '#D4C9A8'],

            edgecolor='black',
            size=3,
            linewidth=0.15,
            jitter=0.2,
            ax=ax
    )
plt.savefig(join(loc, f"[Open Field] MI - Real.png"), dpi = 600)
plt.savefig(join(loc, f"[Open Field] MI - Real.svg"), dpi = 600)
plt.close()
"""

# Other Paradigm
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
for i in range(4):    
    ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((Data['Paradigm'] != 'CrossMaze')&
               (np.isnan(Data['Chi-Square Statistic']) == False)&
               (Data['Type'] == 'Real')&
               (Data['Dimension'] == i+2)&
               (Data["Training Session"]+Data['Dimension'] <= 8))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    sns.lineplot(
            x=SubData["Training Session"]-1, 
            y=SubData["Chi-Square Statistic"], 
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
            y='Chi-Square Statistic',
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
plt.savefig(join(loc, f"[Reverse&Hairpin] Chi2 - Real.png"), dpi = 600)
plt.savefig(join(loc, f"[Reverse&Hairpin] Chi2 - Real.svg"), dpi = 600)
plt.close()
    

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
for i in range(4):    
    ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((Data['Paradigm'] != 'CrossMaze')&
               (np.isnan(Data['Chi-Square Statistic']) == False)&
               (Data['Type'] == 'Real')&
               (Data['Dimension'] == i+2)&
               (Data["Training Session"]+Data['Dimension'] <= 8))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    sns.lineplot(
            x=SubData["Training Session"]-1, 
            y=SubData["Pair Num"], 
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
            y='Pair Num',
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
    ax.set_ylim(0, int(np.max(SubData['Pair Num'])//10000 + 1)*10000)

plt.savefig(join(loc, f"[Reverse&Hairpin] Pair Num - Real.png"), dpi = 600)
plt.savefig(join(loc, f"[Reverse&Hairpin] Pair Num - Real.svg"), dpi = 600)
plt.close()
    
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8))
for i in range(4):    
    ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((Data['Paradigm'] != 'CrossMaze')&
           (np.isnan(Data['Chi-Square Statistic']) == False)&
           (Data['Type'] == 'Real')&
           (Data['Dimension'] == i+2)&
           (Data["Training Session"]+Data['Dimension'] <= 8))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    sns.lineplot(
            x=SubData["Training Session"]-1, 
            y=SubData["MI"], 
            hue=SubData["hue"], 
            palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', 
                     '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
            ax=ax,
            linewidth = 0.5,
            err_kws={"edgecolor":None},
        legend=False
    )
    sns.stripplot(
            x='Training Session',
            y='MI',
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
plt.savefig(join(loc, f"[Reverse&Hairpin] MI - Real.png"), dpi = 600)
plt.savefig(join(loc, f"[Reverse&Hairpin] MI - Real.svg"), dpi = 600)
plt.close()