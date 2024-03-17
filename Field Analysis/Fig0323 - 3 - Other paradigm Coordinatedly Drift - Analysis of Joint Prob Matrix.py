from mylib.statistic_test import *
from mylib.field.field_tracker import indept_test_for_evolution_events, compute_joint_probability_matrix

code_id = "0323 - Coordinatedly Drift - Analysis of Joint Prob Matrix"
loc = join(figpath, code_id, "other paradigm")
mkdir(loc)

from tqdm import tqdm

idx = np.where(f_CellReg_modi['Type'] == 'Real')[0]
if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(
        variable_names = ['Training Session', 'delta-P', 'Dimension', 'Axis', 'Pair Type',
                          'Paradigm', 'X'],
        f = f_CellReg_modi, f_member=['Type'],
        function = CoordinatedDrift_Interface, 
        file_name = code_id, file_idx=idx,
        behavior_paradigm = 'CrossMaze'
    )
    
Data['hue'] = np.array([Data['Paradigm'][i] + Data['Maze Type'][i] + Data['Pair Type'][i] for i in range(Data['Paradigm'].shape[0])])

# Dim = 5 =======================================================================
idx = np.where((Data['Paradigm'] != 'CrossMaze') & 
               (Data['Dimension'] == 5) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(3, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
sns.barplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', 
             '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    ax = ax, 
    capsize = 0.05,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5
)
sns.stripplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    edgecolor='black',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A', 
             '#C3AED6', '#66C7B4', '#A7D8DE', '#F67280'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax
)
ax.set_ylim(-0.2, 0.2)
ax.set_yticks(np.linspace(-0.2, 0.2, 9))
plt.savefig(join(loc, f"[Hairpin & Reverse] Dim = 5, deltaP - IP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Hairpin & Reverse] Dim = 5, deltaP - IP Axis.svg"), dpi = 600)
plt.close()

print("Hairpin & Reverse: Dim = 5")
for x in range(1, 32):
    for paradigm in ['HairpinMaze cis', 'HairpinMaze trs', 'ReverseMaze cis', 'ReverseMaze trs']:
        print(f"  {paradigm} IP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 5) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 5) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()


idx = np.where((Data['Paradigm'] != 'CrossMaze') & 
               (Data['Dimension'] == 5) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(2, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
sns.barplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', 
             '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    ax = ax, 
    capsize = 0.05,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5
)
sns.stripplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    edgecolor='black',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A', 
             '#C3AED6', '#66C7B4', '#A7D8DE', '#F67280'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax
)
ax.set_ylim(-0.1, 0.1)
ax.set_yticks(np.linspace(-0.1, 0.1, 11))
plt.savefig(join(loc, f"[Hairpin & Reverse] Dim = 5, deltaP - CP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Hairpin & Reverse] Dim = 5, deltaP - CP Axis.svg"), dpi = 600)
plt.close()


print("Hairpin & Reverse Maze: Dim = 5")
for x in range(1, 31):
    for paradigm in ['HairpinMaze cis', 'HairpinMaze trs', 
                 'ReverseMaze cis', 'ReverseMaze trs']:
        print(f"  {paradigm} CP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 5) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 5) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()

# Dim = 4 =================================================================================
idx = np.where((Data['Paradigm'] != 'CrossMaze') & 
               (Data['Dimension'] == 4) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(3, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
sns.barplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', 
             '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    ax = ax, 
    capsize = 0.05,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5
)
sns.stripplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    edgecolor='black',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A', 
             '#C3AED6', '#66C7B4', '#A7D8DE', '#F67280'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax
)
ax.set_ylim(-0.2, 0.2)
ax.set_yticks(np.linspace(-0.2, 0.2, 9))
plt.savefig(join(loc, f"[Hairpin & Reverse] Dim = 4, deltaP - IP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Hairpin & Reverse] Dim = 4, deltaP - IP Axis.svg"), dpi = 600)
plt.close()

print("Hairpin & Reverse: Dim = 4")
for x in range(1, 16):
    for paradigm in ['HairpinMaze cis', 'HairpinMaze trs', 'ReverseMaze cis', 'ReverseMaze trs']:
        print(f"  {paradigm} IP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 4) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 4) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()


idx = np.where((Data['Paradigm'] != 'CrossMaze') & 
               (Data['Dimension'] == 4) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(2, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
sns.barplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', 
             '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    ax = ax, 
    capsize = 0.05,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5
)
sns.stripplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    edgecolor='black',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A', 
             '#C3AED6', '#66C7B4', '#A7D8DE', '#F67280'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax
)
ax.set_ylim(-0.1, 0.1)
ax.set_yticks(np.linspace(-0.1, 0.1, 11))
plt.savefig(join(loc, f"[Hairpin & Reverse] Dim = 4, deltaP - CP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Hairpin & Reverse] Dim = 4, deltaP - CP Axis.svg"), dpi = 600)
plt.close()


print("Hairpin & Reverse Maze: Dim = 4")
for x in range(1, 15):
    for paradigm in ['HairpinMaze cis', 'HairpinMaze trs', 
                 'ReverseMaze cis', 'ReverseMaze trs']:
        print(f"  {paradigm} CP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 4) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 4) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()

# Dim = 3 =================================================================================
idx = np.where((Data['Paradigm'] != 'CrossMaze') & 
               (Data['Dimension'] == 3) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(3, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
sns.barplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', 
             '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    ax = ax, 
    capsize = 0.05,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5
)
sns.stripplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    edgecolor='black',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A', 
             '#C3AED6', '#66C7B4', '#A7D8DE', '#F67280'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax
)
ax.set_ylim(-0.2, 0.2)
ax.set_yticks(np.linspace(-0.2, 0.2, 9))
plt.savefig(join(loc, f"[Hairpin & Reverse] Dim = 3, deltaP - IP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Hairpin & Reverse] Dim = 3, deltaP - IP Axis.svg"), dpi = 600)
plt.close()

print("Hairpin & Reverse: Dim = 3")
for x in range(1, 8):
    for paradigm in ['HairpinMaze cis', 'HairpinMaze trs', 'ReverseMaze cis', 'ReverseMaze trs']:
        print(f"  {paradigm} IP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 3) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 3) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()


idx = np.where((Data['Paradigm'] != 'CrossMaze') & 
               (Data['Dimension'] == 3) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(2, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
sns.barplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', 
             '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    ax = ax, 
    capsize = 0.05,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5
)
sns.stripplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    edgecolor='black',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A', 
             '#C3AED6', '#66C7B4', '#A7D8DE', '#F67280'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax
)
ax.set_ylim(-0.1, 0.1)
ax.set_yticks(np.linspace(-0.1, 0.1, 11))
plt.savefig(join(loc, f"[Hairpin & Reverse] Dim = 3, deltaP - CP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Hairpin & Reverse] Dim = 3, deltaP - CP Axis.svg"), dpi = 600)
plt.close()


print("Hairpin & Reverse Maze: Dim = 3")
for x in range(1, 7):
    for paradigm in ['HairpinMaze cis', 'HairpinMaze trs', 
                 'ReverseMaze cis', 'ReverseMaze trs']:
        print(f"  {paradigm} CP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 3) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 3) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()

# Dim = 1 ===========================================================================
idx = np.where((Data['Paradigm'] != 'CrossMaze') & 
               (Data['Dimension'] == 2) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(3, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
sns.barplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', 
             '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    ax = ax, 
    capsize = 0.05,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5
)
sns.stripplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    edgecolor='black',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A', 
             '#C3AED6', '#66C7B4', '#A7D8DE', '#F67280'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax
)
ax.set_ylim(-0.2, 0.2)
ax.set_yticks(np.linspace(-0.2, 0.2, 9))
plt.savefig(join(loc, f"[Hairpin & Reverse] deltaP - IP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Hairpin & Reverse] deltaP - IP Axis.svg"), dpi = 600)
plt.close()

print("Hairpin & Reverse: Dim = 2")
for x in range(1, 4):
    for paradigm in ['HairpinMaze cis', 'HairpinMaze trs', 'ReverseMaze cis', 'ReverseMaze trs']:
        print(f"  {paradigm} IP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 2) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 2) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()


idx = np.where((Data['Paradigm'] != 'CrossMaze') & 
               (Data['Dimension'] == 2) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(2, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
sns.barplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', 
             '#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    ax = ax, 
    capsize = 0.05,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5
)
sns.stripplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    edgecolor='black',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A', 
             '#C3AED6', '#66C7B4', '#A7D8DE', '#F67280'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax
)
ax.set_ylim(-0.1, 0.1)
ax.set_yticks(np.linspace(-0.1, 0.1, 11))
plt.savefig(join(loc, f"[Hairpin & Reverse] deltaP - CP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Hairpin & Reverse] deltaP - CP Axis.svg"), dpi = 600)
plt.close()


print("Hairpin & Reverse Maze: Dim = 2")
for x in range(1, 3):
    for paradigm in ['HairpinMaze cis', 'HairpinMaze trs', 
                 'ReverseMaze cis', 'ReverseMaze trs']:
        print(f"  {paradigm} CP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 2) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == paradigm) & 
               (Data['Dimension'] == 2) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()