from mylib.statistic_test import *
from mylib.field.field_tracker import indept_test_for_evolution_events

code_id = "0318 - Independence test for field evolution events"
loc = join(figpath, code_id)
mkdir(loc)


idx = np.where(f_CellReg_modi['Type'] == 'Real')[0]
if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(
        variable_names = ['Training Session', 'Chi-Square Statistic', 'MI', 'Dimension', 'Pair Type', 'Pair Num',
                          'Paradigm'],
        f = f_CellReg_modi, f_member=['Type'],
        function = IndependentEvolution_Interface, 
        file_name = code_id, file_idx = idx,
        behavior_paradigm = 'CrossMaze'
    )

Data['hue'] = np.array([Data['Maze Type'][i]+Data['Pair Type'][i]+Data['Paradigm'][i] for i in range(Data['Type'].shape[0])])    

idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] != 'Open Field')&
               (np.isnan(Data['Chi-Square Statistic']) == False)&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Dimension',
    y = 'Chi-Square Statistic',
    data = SubData,
    hue = 'hue',
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', '#D4C9A8', '#8E9F85'],
    width=0.8,
    capsize=0.1,
    errcolor='black',
    errwidth=0.5,
)
ax.set_yticks(np.linspace(0, 6000, 7))
ax.set_ylim(0, 6000)
plt.savefig(join(loc, 'Chi-Square Statistic.png'), dpi = 600)
plt.savefig(join(loc, 'Chi-Square Statistic.svg'), dpi = 600)
plt.close()

print("Maze A and B Chi2 Statistic -----------------------------------------")
for dim in range(2, 6):
    print(f"Maze A, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Maze 1')&
                   (np.isnan(Data['Chi-Square Statistic']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Maze 1')&
                   (np.isnan(Data['Chi-Square Statistic']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print(dim, ttest_ind(Data['Chi-Square Statistic'][sib_idx], Data['Chi-Square Statistic'][non_idx]))
    print(f"Maze B dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Maze 2')&
                   (np.isnan(Data['Chi-Square Statistic']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Maze 2')&
                   (np.isnan(Data['Chi-Square Statistic']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print(dim, ttest_ind(Data['Chi-Square Statistic'][sib_idx], Data['Chi-Square Statistic'][non_idx]), end='\n\n')
print()

# Maze A&B Mutual Information
fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Dimension',
    y = 'MI',
    data = SubData,
    hue = 'hue',
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', '#D4C9A8', '#8E9F85'],
    width=0.8,
    capsize=0.1,
    errcolor='black',
    errwidth=0.5,
)
ax.set_ylim(0, 0.1)
ax.set_yticks(np.linspace(0, 0.1, 6))
plt.savefig(join(loc, 'MI.png'), dpi = 600)
plt.savefig(join(loc, 'MI.svg'), dpi = 600)
plt.close()

print("Maze A and B Mutual Information -----------------------------------------")
for dim in range(2, 6):
    print(f"Maze A, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Maze 1')&
                   (np.isnan(Data['MI']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Maze 1')&
                   (np.isnan(Data['MI']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print(dim, ttest_ind(Data['MI'][sib_idx], Data['MI'][non_idx]))
    print(f"Maze B dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Maze 2')&
                   (np.isnan(Data['MI']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Maze 2')&
                   (np.isnan(Data['MI']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print(dim, ttest_ind(Data['MI'][sib_idx], Data['MI'][non_idx]), end='\n\n')
print()


# Other behavior paradigm.
idx = np.where((Data['Paradigm'] != 'CrossMaze')&
               (np.isnan(Data['Chi-Square Statistic']) == False)&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Dimension',
    y = 'Chi-Square Statistic',
    data = SubData,
    hue = 'hue',
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    width=0.8,
    capsize=0.05,
    errcolor='black',
    errwidth=0.5,
)
ax.set_yticks(np.linspace(0, 6000, 7))
ax.set_ylim(0, 6000)
plt.savefig(join(loc, '[Hairpin&Reverse] Chi-Square Statistic.png'), dpi = 600)
plt.savefig(join(loc, '[Hairpin&Reverse] Chi-Square Statistic.svg'), dpi = 600)
plt.close()

print("Hairpin&Reverse Chi2 Statistic -----------------------------------------")
for dim in range(2, 6):
    print(f"HairpinMaze cis, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'HairpinMaze cis')&
                   (np.isnan(Data['Chi-Square Statistic']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'HairpinMaze cis')&
                   (np.isnan(Data['Chi-Square Statistic']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print(dim, ttest_ind(Data['Chi-Square Statistic'][sib_idx], Data['Chi-Square Statistic'][non_idx]))
    
    print(f"HairpinMaze trs, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'HairpinMaze trs')&
                   (np.isnan(Data['Chi-Square Statistic']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'HairpinMaze trs')&
                   (np.isnan(Data['Chi-Square Statistic']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print(dim, ttest_ind(Data['Chi-Square Statistic'][sib_idx], Data['Chi-Square Statistic'][non_idx]), end='\n\n')
    

    print(f"ReverseMaze cis, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'ReverseMaze cis')&
                   (np.isnan(Data['Chi-Square Statistic']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'ReverseMaze cis')&
                   (np.isnan(Data['Chi-Square Statistic']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print(dim, ttest_ind(Data['Chi-Square Statistic'][sib_idx], Data['Chi-Square Statistic'][non_idx]))
    
    print(f"ReverseMaze trs, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'ReverseMaze trs')&
                   (np.isnan(Data['Chi-Square Statistic']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'ReverseMaze trs')&
                   (np.isnan(Data['Chi-Square Statistic']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print(dim, ttest_ind(Data['Chi-Square Statistic'][sib_idx], Data['Chi-Square Statistic'][non_idx]), end='\n\n')
print()

# Maze A&B Mutual Information
fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Dimension',
    y = 'MI',
    data = SubData,
    hue = 'hue',
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    width=0.8,
    capsize=0.05,
    errcolor='black',
    errwidth=0.5,
)
ax.set_ylim(0, 0.30)
ax.set_yticks(np.linspace(0, 0.30, 7))
plt.savefig(join(loc, '[Hairpin&Reverse] MI.png'), dpi = 600)
plt.savefig(join(loc, '[Hairpin&Reverse] MI.svg'), dpi = 600)
plt.close()

print("Hairpin&Reverse Mutual Information -----------------------------------------")
for dim in range(2, 6):
    print(f"HairpinMaze cis, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'HairpinMaze cis')&
                   (np.isnan(Data['MI']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'HairpinMaze cis')&
                   (np.isnan(Data['MI']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print(dim, ttest_ind(Data['MI'][sib_idx], Data['MI'][non_idx]))
    
    print(f"HairpinMaze trs, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'HairpinMaze trs')&
                   (np.isnan(Data['MI']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'HairpinMaze trs')&
                   (np.isnan(Data['MI']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print(dim, ttest_ind(Data['MI'][sib_idx], Data['MI'][non_idx]), end='\n\n')
    

    print(f"ReverseMaze cis, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'ReverseMaze cis')&
                   (np.isnan(Data['MI']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'ReverseMaze cis')&
                   (np.isnan(Data['MI']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print(dim, ttest_ind(Data['MI'][sib_idx], Data['MI'][non_idx]))
    
    print(f"ReverseMaze trs, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'ReverseMaze trs')&
                   (np.isnan(Data['MI']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'ReverseMaze trs')&
                   (np.isnan(Data['MI']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print(dim, ttest_ind(Data['MI'][sib_idx], Data['MI'][non_idx]), end='\n\n')
print()




idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Open Field')&
               (np.isnan(Data['Chi-Square Statistic']) == False)&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Dimension',
    y = 'Chi-Square Statistic',
    data = SubData,
    hue = 'hue',
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', '#D4C9A8', '#8E9F85'],
    width=0.8,
    capsize=0.2,
    errcolor='black',
    errwidth=0.5,
)
ax.set_yticks(np.linspace(0, 20000, 11))
ax.set_ylim(0, 20000)
plt.savefig(join(loc, '[Open Field] Chi-Square Statistic.png'), dpi = 600)
plt.savefig(join(loc, '[Open Field] Chi-Square Statistic.svg'), dpi = 600)
plt.close()

print("Open Field Chi2 Statistic -----------------------------------------")
for dim in range(2, 6):
    print(f"Open Field, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Open Field')&
                   (np.isnan(Data['Chi-Square Statistic']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Open Field')&
                   (np.isnan(Data['Chi-Square Statistic']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print(dim, ttest_ind(Data['Chi-Square Statistic'][sib_idx], Data['Chi-Square Statistic'][non_idx]))
print()

# Maze A&B Mutual Information
fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Dimension',
    y = 'MI',
    data = SubData,
    hue = 'hue',
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', '#D4C9A8', '#8E9F85'],
    width=0.8,
    capsize=0.2,
    errcolor='black',
    errwidth=0.5,
)
ax.set_ylim(0, 0.5)
ax.set_yticks(np.linspace(0, 0.5, 6))
plt.savefig(join(loc, '[Open Field] MI.png'), dpi = 600)
plt.savefig(join(loc, '[Open Field] MI.svg'), dpi = 600)
plt.close()

print("Open Field Mutual Information -----------------------------------------")
for dim in range(2, 6):
    print(f"Open Field, dim {dim}")
    sib_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Open Field')&
                   (np.isnan(Data['MI']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Sibling'))[0]
    
    non_idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == 'Open Field')&
                   (np.isnan(Data['MI']) == False)&
                   (Data['Type'] == 'Real')&
                   (Data['Dimension'] == dim)&
                   (Data['Pair Type'] == 'Non-sibling'))[0]
    
    print(dim, ttest_ind(Data['MI'][sib_idx], Data['MI'][non_idx]))
print()