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
    
Data['hue'] = np.array([Data['Maze Type'][i]+Data['Pair Type'][i]+Data['Paradigm'][i] for i in range(Data['Type'].shape[0])])      
Data['Coordinate Index'] = Data['Coordinate Index'] * 100 

# Maze A and B
idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] != 'Open Field')&
               (np.isnan(Data['Coordinate Index']) == False)&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Dimension',
    y = 'Coordinate Index',
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
    y='Coordinate Index',
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
ax.set_yticks(np.linspace(0, 20, 5))
ax.set_ylim(0, 20)
plt.savefig(join(loc, 'Coordinate Index.png'), dpi = 600)
plt.savefig(join(loc, 'Coordinate Index.svg'), dpi = 600)
plt.close()


print("Maze A and B Chi2 Statistic -----------------------------------------")
for dim in range(2, 6):
    print(f"Maze A, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Maze 1')&
                   (np.isnan(Data['Coordinate Index']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Maze 1')&
                   (np.isnan(Data['Coordinate Index']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print_estimator(Data['Coordinate Index'][sib_idx])
    is_equal_var = levene(Data['Coordinate Index'][sib_idx], Data['Coordinate Index'][non_idx])[1] >= 0.05
    print(dim, is_equal_var, ttest_ind(Data['Coordinate Index'][sib_idx], Data['Coordinate Index'][non_idx], equal_var = is_equal_var))
    print(f"Maze B dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Maze 2')&
                   (np.isnan(Data['Coordinate Index']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Maze 2')&
                   (np.isnan(Data['Coordinate Index']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    print_estimator(Data['Coordinate Index'][sib_idx])
    is_equal_var = levene(Data['Coordinate Index'][sib_idx], Data['Coordinate Index'][non_idx])[1] >= 0.05
    print(dim, is_equal_var, ttest_ind(Data['Coordinate Index'][sib_idx], Data['Coordinate Index'][non_idx], equal_var = is_equal_var), end='\n\n')
print()


idx = np.where((Data['Paradigm'] != 'CrossMaze')&
               (np.isnan(Data['Coordinate Index']) == False)&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Dimension',
    y = 'Coordinate Index',
    data = SubData,
    hue = 'hue',
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    width=0.8,
    capsize=0.05,
    errcolor='black',
    errwidth=0.5,
)
sns.stripplot(
    x='Dimension',
    y='Coordinate Index',
    data=SubData,
    hue='hue',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A', 
             '#C3AED6', '#66C7B4', '#A7D8DE', '#F67280'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    jitter=0.1,
    dodge=True,
    ax=ax
)
ax.set_yticks(np.linspace(0, 20, 5))
ax.set_ylim(0, 20)
plt.savefig(join(loc, '[Hairpin&Reverse] Coordinate Index.png'), dpi = 600)
plt.savefig(join(loc, '[Hairpin&Reverse] Coordinate Index.svg'), dpi = 600)
plt.close()

print("Hairpin&Reverse Chi2 Statistic -----------------------------------------")
for dim in range(2, 6):
    print(f"HairpinMaze cis, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'HairpinMaze cis')&
                   (np.isnan(Data['Coordinate Index']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'HairpinMaze cis')&
                   (np.isnan(Data['Coordinate Index']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    print_estimator(Data['Coordinate Index'][sib_idx])
    is_equal_var = levene(Data['Coordinate Index'][sib_idx], Data['Coordinate Index'][non_idx])[1] >= 0.05
    print(dim, is_equal_var, ttest_ind(Data['Coordinate Index'][sib_idx], Data['Coordinate Index'][non_idx], equal_var = is_equal_var))
    
    print(f"HairpinMaze trs, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'HairpinMaze trs')&
                   (np.isnan(Data['Coordinate Index']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'HairpinMaze trs')&
                   (np.isnan(Data['Coordinate Index']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print_estimator(Data['Coordinate Index'][sib_idx])
    is_equal_var = levene(Data['Coordinate Index'][sib_idx], Data['Coordinate Index'][non_idx])[1] >= 0.05
    print(dim, is_equal_var, ttest_ind(Data['Coordinate Index'][sib_idx], Data['Coordinate Index'][non_idx], equal_var = is_equal_var))
    

    print(f"ReverseMaze cis, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'ReverseMaze cis')&
                   (np.isnan(Data['Coordinate Index']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'ReverseMaze cis')&
                   (np.isnan(Data['Coordinate Index']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print_estimator(Data['Coordinate Index'][sib_idx])
    is_equal_var = levene(Data['Coordinate Index'][sib_idx], Data['Coordinate Index'][non_idx])[1] >= 0.05
    print(dim, is_equal_var, ttest_ind(Data['Coordinate Index'][sib_idx], Data['Coordinate Index'][non_idx], equal_var = is_equal_var))
    
    print(f"ReverseMaze trs, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'ReverseMaze trs')&
                   (np.isnan(Data['Coordinate Index']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'ReverseMaze trs')&
                   (np.isnan(Data['Coordinate Index']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print_estimator(Data['Coordinate Index'][sib_idx])
    is_equal_var = levene(Data['Coordinate Index'][sib_idx], Data['Coordinate Index'][non_idx])[1] >= 0.05
    print(dim, is_equal_var, ttest_ind(Data['Coordinate Index'][sib_idx], Data['Coordinate Index'][non_idx], equal_var = is_equal_var), end='\n\n')
print()


idx = np.where((Data['Paradigm'] == 'CrossMaze')&
            (np.isnan(Data['Coordinate Index']) == False)&
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

print("Cross day comparison of Coordinate Index, Maze A and B:")
for i in range(4):
    print("    Maze A, Dim ", i+2, 
          ttest_rel(SubData['Coordinate Index'][np.where((SubData['hue'] == 'First SessionMaze 1')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['Coordinate Index'][np.where((SubData['hue'] == 'Last SessionMaze 1')&(SubData['Dimension'] == i+2))[0]]))
    print("    Maze B, Dim ", i+2, 
          ttest_rel(SubData['Coordinate Index'][np.where((SubData['hue'] == 'First SessionMaze 2')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['Coordinate Index'][np.where((SubData['hue'] == 'Last SessionMaze 2')&(SubData['Dimension'] == i+2))[0]]), end='\n\n')
    
fig = plt.figure(figsize=(5,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Dimension',
    y = 'Coordinate Index',
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
    y='Coordinate Index',
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
plt.savefig(join(loc, '[Crossday Comparison] Coordinate Index.png'), dpi = 600)
plt.savefig(join(loc, '[Crossday Comparison] Coordinate Index.svg'), dpi = 600)
plt.close()

idx = np.where((Data['Paradigm'] != 'CrossMaze')&
            (np.isnan(Data['Coordinate Index']) == False)&
            (Data['Type'] == 'Real')&
            (Data['Pair Type'] == 'Sibling')&
            ((Data["Training Session"]+Data['Dimension'] == 8)|
             (Data['Training Session'] == 1)))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

for i in np.where(SubData['Training Session'] == 1)[0]:
    SubData['hue'][i] = 'First Session' + SubData['Paradigm'][i]
for i in np.where(SubData['Training Session'] != 1)[0]:
    SubData['hue'][i] = 'Last Session' + SubData['Paradigm'][i]

print("Cross day comparison of Mutual information, Hairpin & Reverse:")
for i in range(4):
    print("    MAf, Dim ", i+2, 
          ttest_rel(SubData['Coordinate Index'][np.where((SubData['hue'] == 'First SessionReverseMaze cis')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['Coordinate Index'][np.where((SubData['hue'] == 'Last SessionReverseMaze cis')&(SubData['Dimension'] == i+2))[0]]))
    
    print("    MAb, Dim ", i+2, 
          ttest_rel(SubData['Coordinate Index'][np.where((SubData['hue'] == 'First SessionReverseMaze trs')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['Coordinate Index'][np.where((SubData['hue'] == 'Last SessionReverseMaze trs')&(SubData['Dimension'] == i+2))[0]]))
    
    print("    HPf, Dim ", i+2, 
          ttest_rel(SubData['Coordinate Index'][np.where((SubData['hue'] == 'First SessionHairpinMaze cis')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['Coordinate Index'][np.where((SubData['hue'] == 'Last SessionHairpinMaze cis')&(SubData['Dimension'] == i+2))[0]]))
    
    print("    HPb, Dim ", i+2, 
          ttest_rel(SubData['Coordinate Index'][np.where((SubData['hue'] == 'First SessionHairpinMaze trs')&(SubData['Dimension'] == i+2))[0]], 
                    SubData['Coordinate Index'][np.where((SubData['hue'] == 'Last SessionHairpinMaze trs')&(SubData['Dimension'] == i+2))[0]]), end='\n\n')
    
fig = plt.figure(figsize=(5,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Dimension',
    y = 'Coordinate Index',
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
    y='Coordinate Index',
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
plt.savefig(join(loc, '[Crossday Comparison] [Hairpin & Reverse] Coordinate Index.png'), dpi = 600)
plt.savefig(join(loc, '[Crossday Comparison] [Hairpin & Reverse] Coordinate Index.svg'), dpi = 600)
plt.close()