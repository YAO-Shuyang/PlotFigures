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
        function = CoordinatedDrift_Interface, 
        file_name = code_id, file_idx=idx,
        behavior_paradigm = 'CrossMaze'
    )

Data['hue'] = np.array([Data['Paradigm'][i] + Data['Maze Type'][i] + Data['Pair Type'][i] for i in range(Data['Paradigm'].shape[0])])

idx = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 5) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(15, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
sns.barplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    ax = ax, 
    capsize = 0.1,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5,
    zorder=2
)
sns.stripplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    edgecolor='black',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax,
    zorder = 1
)
ax.set_ylim(-0.02, 0.02)
ax.set_yticks(np.linspace(-0.02, 0.02, 9))
plt.savefig(join(loc, f"[Maze A & B] Dim = 5, deltaP - IP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Maze A & B] Dim = 5, deltaP - IP Axis.svg"), dpi = 600)
plt.show()

idx = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 5) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(15, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
sns.barplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    ax = ax, 
    capsize = 0.1,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5,
    zorder=2
)
sns.stripplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    edgecolor='black',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax,
    zorder = 1
)
plt.savefig(join(loc, f"[Maze A & B] Dim = 5, deltaP - CP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Maze A & B] Dim = 5, deltaP - CP Axis.svg"), dpi = 600)
plt.show()

print("Maze A & B: Dim = 5")
for x in range(1, 32):
    for maze in ['Maze 1', 'Maze 2']:
        print(f"  {maze} IP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 5) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 5) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()

print("Maze A & B: Dim = 5")
for x in range(1, 31):
    for maze in ['Maze 1', 'Maze 2']:
        print(f"  {maze} CP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 5) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 5) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()

idx = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 4) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(7, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
sns.barplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    ax = ax, 
    capsize = 0.1,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5,
    zorder=2
)
sns.stripplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    edgecolor='black',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax,
    zorder = 1
)
ax.set_ylim(-0.04, 0.04)
ax.set_yticks(np.linspace(-0.04, 0.04, 9))
plt.savefig(join(loc, f"[Maze A & B] Dim = 4, deltaP - IP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Maze A & B] Dim = 4, deltaP - IP Axis.svg"), dpi = 600)
plt.show()

idx = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 4) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(7, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
sns.barplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    ax = ax, 
    capsize = 0.1,
    errcolor='black',
    errwidth=0.5,
    linewidth=0.5,
    zorder=2
)
sns.stripplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    edgecolor='black',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax,
    zorder = 1
)
ax.set_ylim(-0.01, 0.01)
ax.set_yticks(np.linspace(-0.01, 0.01, 5))
plt.savefig(join(loc, f"[Maze A & B] Dim = 4, deltaP - CP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Maze A & B] Dim = 4, deltaP - CP Axis.svg"), dpi = 600)
plt.show()

print("Maze A & B: Dim = 4")
for x in range(1, 16):
    for maze in ['Maze 1', 'Maze 2']:
        print(f"  {maze} IP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 4) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 4) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()

print("Maze A & B: Dim = 4")
for x in range(1, 15):
    for maze in ['Maze 1', 'Maze 2']:
        print(f"  {maze} CP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 4) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 4) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()


idx = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 3) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(5, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
sns.barplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    ax = ax, 
    capsize = 0.1,
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
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax
)
ax.set_ylim(-0.1, 0.1)
ax.set_yticks(np.linspace(-0.1, 0.1, 11))
plt.savefig(join(loc, f"[Maze A & B] Dim = 3, deltaP - IP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Maze A & B] Dim = 3, deltaP - IP Axis.svg"), dpi = 600)
plt.close()

idx = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 3) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(5, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
sns.barplot(
    x = 'X',
    y = 'delta-P',
    hue = 'hue',
    data = SubData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    ax = ax, 
    capsize = 0.1,
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
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax
)
ax.set_ylim(-0.02, 0.02)
ax.set_yticks(np.linspace(-0.02, 0.02, 9))
plt.savefig(join(loc, f"[Maze A & B] Dim = 3, deltaP - CP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Maze A & B] Dim = 3, deltaP - CP Axis.svg"), dpi = 600)
plt.close()

print("Maze A & B: Dim = 3")
for x in range(1, 8):
    for maze in ['Maze 1', 'Maze 2']:
        print(f"  {maze} IP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 3) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 3) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()

print("Maze A & B: Dim = 3")
for x in range(1, 7):
    for maze in ['Maze 1', 'Maze 2']:
        print(f"  {maze} CP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 3) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 3) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()

idx = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 2) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] != 'Open Field')&
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
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    ax = ax, 
    capsize = 0.1,
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
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax
)
ax.set_ylim(-0.2, 0.2)
ax.set_yticks(np.linspace(-0.2, 0.2, 9))
plt.savefig(join(loc, f"[Maze A & B] deltaP - IP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Maze A & B] deltaP - IP Axis.svg"), dpi = 600)
plt.close()

print("Maze A & B: Dim = 2")
for x in range(1, 4):
    for maze in ['Maze 1', 'Maze 2']:
        print(f"  {maze} IP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 2) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 2) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'IP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()


idx = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 2) & 
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] != 'Open Field')&
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
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    ax = ax, 
    capsize = 0.1,
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
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A'],
    size=2,
    linewidth=0.10,
    jitter=0.2,
    dodge=True,
    ax = ax
)
ax.set_ylim(-0.1, 0.1)
ax.set_yticks(np.linspace(-0.1, 0.1, 11))
plt.savefig(join(loc, f"[Maze A & B] deltaP - CP Axis.png"), dpi = 600)
plt.savefig(join(loc, f"[Maze A & B] deltaP - CP Axis.svg"), dpi = 600)
plt.close()


print("Maze A & B: Dim = 2")
for x in range(1, 3):
    for maze in ['Maze 1', 'Maze 2']:
        print(f"  {maze} CP axis X = {x}:")
        sib = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 2) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Sibling'))[0]
        non = np.where((Data['Paradigm'] == 'CrossMaze') & 
               (Data['Dimension'] == 2) & 
               (Data['X'] == x) &
               (np.isnan(Data['delta-P']) == False)&
               (Data['Maze Type'] == maze)&
               (Data['Axis'] == 'CP axis')&
               (Data['Training Session'] + Data['Dimension'] <= 14) &
               (Data['Pair Type'] == 'Non-sibling'))[0]
        print("    ", ttest_ind(Data['delta-P'][sib], Data['delta-P'][non]))
        print("      with 0:", ttest_1samp(Data['delta-P'][sib], 0))
        print("      with 0:", ttest_1samp(Data['delta-P'][non], 0))
    print()