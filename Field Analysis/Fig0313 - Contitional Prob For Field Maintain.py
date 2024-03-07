from mylib.statistic_test import *
from mylib.statistic_test import GetMultidayIndexmap, ReadCellReg
from mylib.multiday.core import MultiDayCore

code_id = "0313 - Conditional Probability for Field Maintain"
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = [
                             'Duration', 'Conditional Prob.', 'Conditional Recover Prob.',
                             'Global Recover Prob.', 'Cumulative Prob.', 
                             'Paradigm', 'On-Next Num', 'Off-Next Num'], 
                             f_member=['Type'],
                             f = f_CellReg_modi, function = ConditionalProb_Interface, 
                             file_name = code_id, behavior_paradigm = 'CrossMaze'
           )
"""
# Get Chance Level
if os.path.exists(join(figdata, code_id+' [Chance Level].pkl')):
    with open(join(figdata, code_id+' [Chance Level].pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    # Only Cross Maze Paradigm
    Data = {"MiceID": [], "Maze Type": [], "Conditional Prob.": [], 
            "Conditional Recover Prob.": [], "Global Recover Prob.": [], 
            "Cumulative Prob.": [], "Paradigm": [], "On-Next Num": [], 
            "Off-Next Num": []}
    
    for i in range(len(f_CellReg_modi)):
        if f['paradigm'][i] == 'CrossMaze':
            if f['maze_type'][i] == 0:
                index_map = GetMultidayIndexmap(
                    mouse=f['MiceID'][i],
                    stage=f['Stage'][i],
                    session=f['session'][i],
                    occu_num=2
                )
            else:
                with open(f['cellreg_folder'][i], 'rb') as handle:
                    index_map = pickle.load(handle)
        else:
            index_map = ReadCellReg(f['cellreg_folder'][i])
            
        index_map[np.where((index_map < 0)|np.isnan(index_map))] = 0
        mat = np.where(index_map>0, 1, 0)
        num = np.sum(mat, axis = 0)
        index_map = index_map[:, np.where(num >= 2)[0]]  
        print(index_map.shape)
        
        mouse = f_CellReg_modi['MiceID'][i]
        maze_type = f_CellReg_modi['maze_type'][i]
        paradigm = f_CellReg_modi['paradigm'][i]
        session = f_CellReg_modi['session'][i]
        stage = f_CellReg_modi['Stage'][i]
        
        # Initial basic elements
        n_neurons = index_map.shape[1]
        n_sessions = index_map.shape[0]    

        # Get information from daily trace.pkl
        core = MultiDayCore(
            keys = ['is_placecell', 'place_field_all_multiday'],
            paradigm=paradigm,
            direction=None
        )
        file_indices = np.where((f1['MiceID'] == mouse) & 
                                (f1['Stage'] == stage) & 
                                (f1['session'] == session))[0]
        
        if stage == 'Stage 1+2':
            file_indices = np.where((f1['MiceID'] == mouse) & 
                                    (f1['session'] == session) & 
                                    ((f1['Stage'] == 'Stage 1') | 
                                     (f1['Stage'] == 'Stage 2')))[0]
        
        if stage == 'Stage 1' and mouse in [10212] and session == 2:
            file_indices = np.where((f1['MiceID'] == mouse) & 
                                    (f1['session'] == session) & 
                                    (f1['Stage'] == 'Stage 1') & 
                                    (f1['date'] != 20230506))[0]
            
        res = core.get_trace_set(
            f=f1, 
            file_indices=file_indices, 
            keys=['is_placecell', 'place_field_all_multiday']
        )
"""

    
from scipy.optimize import curve_fit
def exp_func(x, k, b):
    return 1 - np.exp(-k * (x-b))

def polynomial_converge2(x, k, b, c):
    return c - 1 / (k*x + b)

def kww_decay(x, a, b, c):
    return a*np.exp(-np.power(x/b, c))

def report_para(Data, name: str = 'KWW', key: str = 'Conditional Prob.'):
    mazes = ['Maze 1', 'Maze 2']
    mice = [10209, 10212, 10224, 10227]
    
    if name == 'KWW':
        params = [[], [], []]
    elif name == 'exp':
        params = [[], []]
    elif name == 'poly':
        params = [[], [], []]

    for maze in mazes:
        idx = np.where((Data['Paradigm'] == 'CrossMaze')&
                       (Data['Maze Type'] == maze)&
                       (np.isnan(Data[key]) == False))[0]
        if len(idx) == 0:
            continue
        if name == 'KWW':
            params0, _ = curve_fit(kww_decay, 
                                   Data['Duration'][idx], 
                                   Data[key][idx]/100)
            params[0].append(params0[0])
            params[1].append(params0[1])
        elif name == 'exp':
            params0, _ = curve_fit(exp_func, 
                                   Data['Duration'][idx], 
                                   Data[key][idx]/100)
            params[0].append(params0[0])
            params[1].append(params0[1])
        elif name == 'poly':
            bounds = ([0, -np.inf, 0], [np.inf, np.inf, 1])
            params0, _ = curve_fit(polynomial_converge2, 
                                   Data['Duration'][idx], 
                                   Data[key][idx]/100, 
                                   bounds=bounds)
            params[0].append(params0[0])
            params[1].append(params0[1])
            params[2].append(params0[2])
                
    if name == 'KWW':
        print("KWW fitted parameters for all mice and both mazes -------------------------------")
        print("  b: ", np.mean(params[1]), np.std(params[1]))
        print("           ", params[1])
        print("  c: ", np.mean(params[2]), np.std(params[2]))
        print("           ", params[2], end='\n\n')
    elif name == 'exp':
        print("exp fitted parameters for all mice and both mazes -------------------------------")
        print("  k: ", np.mean(params[0]), np.std(params[0]))
        print("           ", params[0])
        print("  x0: ", np.mean(params[1]), np.std(params[1]))
        print("           ", params[1], end='\n\n')
    elif name == 'poly':
        print("poly fitted parameters for all mice and both mazes -------------------------------")
        print("  k: ", np.mean(params[0]), np.std(params[0]))
        print("           ", params[0])
        print("  b: ", np.mean(params[1]), np.std(params[1]))
        print("           ", params[1])
        print("  c: ", np.mean(params[2]), np.std(params[2]))
        print("           ", params[2], end='\n\n')


colors = sns.color_palette("rocket", 3)[1:]
markercolors = [sns.color_palette("Blues", 3)[1], sns.color_palette("Blues", 3)[2]]
chancecolors = ['#D4C9A8', '#8E9F85', '#C3AED6', '#FED7D7']

#Data['hue'] = np.array([Data['Papadigm'][i] + ' ' + Data['Maze Type'][i] for i in range(Data['Duration'].shape[0])])
idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Conditional Prob.']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

report_para(SubData, name='poly', key='Conditional Prob.')
report_para(SubData, name='exp', key='Conditional Prob.')
report_para(SubData, name='KWW', key='Conditional Recover Prob.')

idx1 = np.where(SubData['Maze Type'] == 'Maze 1')[0]
idx2 = np.where(SubData['Maze Type'] == 'Maze 2')[0]

initial_guesses = [0.9, 10, 0.5]  # Adjust these based on your data
# Bounds for the parameters: a between 0 and 1, b positive, c between 0 and 1
bounds = ([0, -np.inf, 0], [np.inf, np.inf, 1])
params1, _ = curve_fit(polynomial_converge2, 
                       SubData['Duration'][idx1], 
                       SubData['Conditional Prob.'][idx1]/100, 
                       bounds=bounds)
x1 = np.linspace(min(SubData['Duration'][idx1]), max(SubData['Duration'][idx1]), 10000)
print(params1)
params2, _ = curve_fit(polynomial_converge2, 
                       SubData['Duration'][idx2], 
                       SubData['Conditional Prob.'][idx2]/100, 
                       bounds=bounds)
x2 = np.linspace(min(SubData['Duration'][idx2]), max(SubData['Duration'][idx2]), 10000)
print(params2)

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
k1, b1, c1 = params1
k2, b2, c2 = params2

y1 = polynomial_converge2(x1, k1, b1, c1)
y2 = polynomial_converge2(x2, k2, b2, c2)
ax.plot(x1-1, y1*100, color=colors[0], label='Maze 1', linewidth = 0.5)
ax.plot(x2-1, y2*100, color=colors[1], label='Maze 2', linewidth = 0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Prob.',
    data=SubData,
    hue = "Maze Type",
    palette = markercolors,
    edgecolor='black',
    size=2,
    linewidth=0.1,
    alpha=0.8,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 103)
ax.set_yticks(np.linspace(0, 100, 6))

plt.savefig(join(loc, 'conditional prob.png'), dpi = 600)
plt.savefig(join(loc, 'conditional prob.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(5, 2.5))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Duration',
    y = 'On-Next Num',
    data=SubData,
    hue = "Maze Type",
    palette = colors,
    width=0.8,
    capsize=0.2,
    errcolor='black',
    errwidth=0.5,
    ax = ax
)
sns.stripplot(
    x = 'Duration',
    y = 'On-Next Num',
    data=SubData,
    hue = "Maze Type",
    palette = markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.1,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.semilogy()
ax.set_ylim(1, 50000)
plt.savefig(join(loc, 'on-next num.png'), dpi = 600)
plt.savefig(join(loc, 'on-next num.svg'), dpi = 600)
plt.close()


fig = plt.figure(figsize=(4,2))
idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Conditional Recover Prob.']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Conditional Recover Prob.']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Type'] == 'Shuffle'))[0]
ShufData = SubDict(Data, Data.keys(), idx=idx)
idx1 = np.where(SubData['Maze Type'] == 'Maze 1')[0]
idx2 = np.where(SubData['Maze Type'] == 'Maze 2')[0]
params1, _ = curve_fit(kww_decay, SubData['Duration'][idx1], SubData['Conditional Recover Prob.'][idx1]/100)
print("Conditional Recovery prob. -----------------------------------------")
x1 = np.linspace(min(SubData['Duration'][idx1]), max(SubData['Duration'][idx1]), 10000)
print("Maze 1", params1)
params2, _ = curve_fit(kww_decay, SubData['Duration'][idx2], SubData['Conditional Recover Prob.'][idx2]/100)
x2 = np.linspace(min(SubData['Duration'][idx2]), max(SubData['Duration'][idx2]), 10000)
print("Maze 2", params2)

y1 = kww_decay(x1, *params1)
y2 = kww_decay(x2, *params2)


fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x1-1, y1*100, color=colors[0], label='Maze 1', linewidth = 0.5)
ax.plot(x2-1, y2*100, color=colors[1], label='Maze 2', linewidth = 0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Recover Prob.',
    data=SubData,
    hue = "Maze Type",
    palette = markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
sns.lineplot(
    x = ShufData['Duration']-1,
    y = ShufData['Conditional Recover Prob.'],
    hue = ShufData['Maze Type'],
    palette = chancecolors,
    linewidth = 0.5,
    err_kws={"edgecolor": None},
    ax = ax
)
ax.set_ylim(-1, 40)
ax.set_yticks(np.linspace(0, 40, 5))

plt.savefig(join(loc, 'Conditional recover prob.png'), dpi = 600)
plt.savefig(join(loc, 'Conditional recover prob.svg'), dpi = 600)
plt.close()

print("Maze A&B Conditional Recovery Prob. -----------------------------------------")
for i in range(1, 25):
    print("Day ",i)
    print(ttest_ind(SubData['Conditional Recover Prob.'][SubData['Duration'] == i], 
                    ShufData['Conditional Recover Prob.'][ShufData['Duration'] == i]))
print()

fig = plt.figure(figsize=(5, 2.5))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Duration',
    y = 'Off-Next Num',
    data=SubData,
    hue = "Maze Type",
    palette = colors,
    width=0.8,
    capsize=0.2,
    errcolor='black',
    errwidth=0.5,
    ax = ax
)
sns.stripplot(
    x = 'Duration',
    y = 'Off-Next Num',
    data=SubData,
    hue = "Maze Type",
    palette = markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.1,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.semilogy()
ax.set_ylim(1, 40000)
plt.savefig(join(loc, 'off-next num.png'), dpi = 600)
plt.savefig(join(loc, 'off-next num.svg'), dpi = 600)
plt.close()


idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Conditional Prob.']) == False)&
               (Data['Maze Type'] == 'Open Field')&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Conditional Prob.']) == False)&
               (Data['Maze Type'] == 'Open Field')&
               (Data['Type'] == 'Shuffle'))[0]
ShufData = SubDict(Data, Data.keys(), idx=idx)

params2, _ = curve_fit(polynomial_converge2, SubData['Duration'], SubData['Conditional Prob.']/100)
x2 = np.linspace(min(SubData['Duration']), max(SubData['Duration']), 10000)
print("Open Field P1")
print(params2)

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

y2 = polynomial_converge2(x2, *params2)
ax.plot(x2-1, y2*100, color=colors[1], label='Maze 2', linewidth = 0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Prob.',
    data=SubData,
    hue = "Maze Type",
    palette = markercolors,
    edgecolor='black',
    size=2,
    linewidth=0.1,
    alpha=0.8,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 103)
ax.set_yticks(np.linspace(0, 100, 6))

plt.savefig(join(loc, '[Open Field] conditional prob.png'), dpi = 600)
plt.savefig(join(loc, '[Open Field] conditional prob.svg'), dpi = 600)
plt.close()



fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Duration',
    y = 'On-Next Num',
    data=SubData,
    hue = "Maze Type",
    palette = colors,
    width=0.8,
    capsize=0.2,
    errcolor='black',
    errwidth=0.5,
    ax = ax
)
sns.stripplot(
    x = 'Duration',
    y = 'On-Next Num',
    data=SubData,
    hue = "Maze Type",
    palette = markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.1,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(1, 40000)
ax.semilogy()

plt.savefig(join(loc, '[Open Field] on-next num.png'), dpi = 600)
plt.savefig(join(loc, '[Open Field] on-next num.svg'), dpi = 600)
plt.close()



fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Conditional Recover Prob.']) == False)&
               (Data['Maze Type'] == 'Open Field')&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
params2, _ = curve_fit(kww_decay, SubData['Duration'], SubData['Conditional Recover Prob.']/100)
x2 = np.linspace(min(SubData['Duration']), max(SubData['Duration']), 10000)
print(params2)
y2 = kww_decay(x2, *params2)
ax.plot(x2-1, y2*100, color=colors[1], label='Maze 2', linewidth = 0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Recover Prob.',
    data=SubData,
    hue = "Maze Type",
    palette = markercolors,
    edgecolor='black',
    size=2,
    linewidth=0.1,
    alpha=0.8,
    ax = ax,
    dodge=True,
    jitter=0.1
)
sns.lineplot(
    x = ShufData['Duration']-1,
    y = ShufData['Conditional Recover Prob.'],
    hue = ShufData['Maze Type'],
    palette = chancecolors,
    linewidth = 0.5,
    err_kws={"edgecolor": None},
    ax = ax
)
ax.set_ylim(-1, 60)
ax.set_yticks(np.linspace(0, 60, 7))

plt.savefig(join(loc, '[Open Field] conditional recovery prob.png'), dpi = 600)
plt.savefig(join(loc, '[Open Field] conditional recovery prob.svg'), dpi = 600)
plt.close()

print("Open Field Recovery Statistic Test")
for i in range(1, 12):
    print("Day ",i)
    print(ttest_ind(SubData['Conditional Recover Prob.'][SubData['Duration'] == i], 
                    ShufData['Conditional Recover Prob.'][ShufData['Duration'] == i]))
print()

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Duration',
    y = 'Off-Next Num',
    data=SubData,
    hue = "Maze Type",
    palette = colors,
    width=0.8,
    capsize=0.2,
    errcolor='black',
    errwidth=0.5,
    ax = ax
)
sns.stripplot(
    x = 'Duration',
    y = 'Off-Next Num',
    data=SubData,
    hue = "Maze Type",
    palette = markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.1,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(1, 10000)
ax.semilogy()

plt.savefig(join(loc, '[Open Field] off-next num.png'), dpi = 600)
plt.savefig(join(loc, '[Open Field] off-next num.svg'), dpi = 600)
plt.close()

# Hairpin & Reverse Maze

idx = np.where((Data['Paradigm'] != 'CrossMaze')&
               (np.isnan(Data['Conditional Prob.']) == False)&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

# Fit parameters
idx = np.where(SubData['Paradigm'] == 'HairpinMaze cis')[0]
params1, _ = curve_fit(polynomial_converge2, SubData['Duration'][idx], SubData['Conditional Prob.'][idx]/100, bounds=bounds)
x1 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print("Hairpin&Reverse Conditional Prob.")
print(params1)
y1 = polynomial_converge2(x1, *params1)

idx = np.where(SubData['Paradigm'] == 'HairpinMaze trs')[0]
params2, _ = curve_fit(polynomial_converge2, SubData['Duration'][idx], SubData['Conditional Prob.'][idx]/100, bounds=bounds)
x2 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params2)
y2 = polynomial_converge2(x2, *params2)

idx = np.where(SubData['Paradigm'] == 'ReverseMaze cis')[0]
params3, _ = curve_fit(polynomial_converge2, SubData['Duration'][idx], SubData['Conditional Prob.'][idx]/100, bounds=bounds)
x3 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params3)
y3 = polynomial_converge2(x3, *params3)

idx = np.where(SubData['Paradigm'] == 'ReverseMaze trs')[0]
params4, _ = curve_fit(polynomial_converge2, SubData['Duration'][idx], SubData['Conditional Prob.'][idx]/100, bounds=bounds)
x4 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params4)
    
y4 = polynomial_converge2(x4, *params4)
colors = ['#6D9BA8', '#A3CBB2', '#E9D985', '#D57A66']
ax.plot(x1-1, y1*100, color=colors[0], linewidth = 0.5)
ax.plot(x2-1, y2*100, color=colors[1], linewidth = 0.5)
ax.plot(x3-1, y3*100, color=colors[2], linewidth = 0.5)
ax.plot(x4-1, y4*100, color=colors[3], linewidth = 0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Prob.',
    data=SubData,
    hue = "Paradigm",
    hue_order=['HairpinMaze cis', 'HairpinMaze trs', 'ReverseMaze cis', 'ReverseMaze trs'],
    palette = colors,
    edgecolor='black',
    size=2,
    linewidth=0.1,
    alpha=0.9,
    ax = ax,
    jitter=0.2
)
ax.set_ylim(0, 103)
ax.set_yticks(np.linspace(0, 100, 6))

plt.savefig(join(loc, '[Hairpin&Reverse] conditional prob.png'), dpi = 600)
plt.savefig(join(loc, '[Hairpin&Reverse] conditional prob.svg'), dpi = 600)
plt.close()


idx = np.where((Data['Paradigm'] != 'CrossMaze')&
               (np.isnan(Data['Conditional Recover Prob.']) == False)&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

idx = np.where((Data['Paradigm'] != 'CrossMaze')&
               (np.isnan(Data['Conditional Recover Prob.']) == False)&
               (Data['Type'] == 'Shuffle'))[0]
ShufData = SubDict(Data, Data.keys(), idx=idx)

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

# Fit parameters
bounds = [[0, 0, 0], [np.inf, np.inf, 1]]
initial_guesses = [1, 1, 0.5]
idx = np.where(SubData['Paradigm'] == 'HairpinMaze cis')[0]
params1, _ = curve_fit(kww_decay, SubData['Duration'][idx], 
                       SubData['Conditional Recover Prob.'][idx]/100, 
                       bounds=bounds, 
                       p0=initial_guesses)
x1 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print("Hairpin&Reverse Conditional Recover Prob.")
print(params1)
y1 = kww_decay(x1, *params1)

idx = np.where(SubData['Paradigm'] == 'HairpinMaze trs')[0]
params2, _ = curve_fit(kww_decay, SubData['Duration'][idx], 
                       SubData['Conditional Recover Prob.'][idx]/100, 
                       bounds=bounds, 
                       p0=initial_guesses)
x2 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params2)
y2 = kww_decay(x2, *params2)

idx = np.where(SubData['Paradigm'] == 'ReverseMaze cis')[0]
params3, _ = curve_fit(kww_decay, SubData['Duration'][idx], 
                       SubData['Conditional Recover Prob.'][idx]/100, 
                       bounds=bounds, 
                       p0=initial_guesses)
x3 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params3)
y3 = kww_decay(x3, *params3)

try:
    idx = np.where(SubData['Paradigm'] == 'ReverseMaze trs')[0]
    params4, _ = curve_fit(lambda x, c, b: kww_decay(x, 1, b, c), SubData['Duration'][idx], 
                       SubData['Conditional Recover Prob.'][idx]/100)
    x4 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
    print(params4)
except:
    params4 = [np.nan, np.nan, np.nan]

y4 = kww_decay(x4, 1, *params4)
colors = ['#6D9BA8', '#A3CBB2', '#E9D985', '#D57A66']
ax.plot(x1-1, y1*100, color=colors[0], linewidth = 0.5)
ax.plot(x2-1, y2*100, color=colors[1], linewidth = 0.5)
ax.plot(x3-1, y3*100, color=colors[2], linewidth = 0.5)
ax.plot(x4-1, y4*100, color=colors[3], linewidth = 0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Recover Prob.',
    data=SubData,
    hue = "Paradigm",
    hue_order=['HairpinMaze cis', 'HairpinMaze trs', 'ReverseMaze cis', 'ReverseMaze trs'],
    palette = colors,
    edgecolor='black',
    size=2,
    linewidth=0.1,
    alpha=0.9,
    ax = ax,
    jitter=0.2
)
sns.lineplot(
    x = ShufData['Duration']-1,
    y = ShufData['Conditional Recover Prob.'],
    hue = ShufData['Paradigm'],
    hue_order=['HairpinMaze cis', 'HairpinMaze trs', 'ReverseMaze cis', 'ReverseMaze trs'],
    palette = chancecolors,
    linewidth = 0.5,
    err_kws={"edgecolor": None},
    ax = ax
)
ax.set_ylim(0, 40)
ax.set_yticks(np.linspace(0, 40, 5))

plt.savefig(join(loc, '[Hairpin&Reverse] conditional recover prob.png'), dpi = 600)
plt.savefig(join(loc, '[Hairpin&Reverse] conditional recover prob.svg'), dpi = 600)
plt.close()

print("HairpinMaze cis Recovery Statistic Test")
for i in range(1, 11):
    print("Day ",i)
    print(ttest_ind(SubData['Conditional Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'HairpinMaze cis'))[0]],
                    ShufData['Conditional Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'HairpinMaze cis'))[0]]))
print()
print("HairpinMaze trs Recovery Statistic Test")
for i in range(1, 11):
    print("Day ",i)
    print(ttest_ind(SubData['Conditional Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'HairpinMaze trs'))[0]],
                    ShufData['Conditional Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'HairpinMaze trs'))[0]]))
print()
print("Reverse cis Recovery Statistic Test")
for i in range(1, 11):
    print("Day ",i)
    print(ttest_ind(SubData['Conditional Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'ReverseMaze cis'))[0]],
                    ShufData['Conditional Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'ReverseMaze cis'))[0]]))
print()
print("Reverse trs Recovery Statistic Test")
for i in range(1, 11):
    print("Day ",i)
    print(ttest_ind(SubData['Conditional Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'ReverseMaze trs'))[0]],
                    ShufData['Conditional Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'ReverseMaze trs'))[0]]))
print()

# Global Recovery Prob.
idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Global Recover Prob.']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Global Recover Prob.']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Type'] == 'Shuffle'))[0]
ShufData = SubDict(Data, Data.keys(), idx=idx)

idx1 = np.where(SubData['Maze Type'] == 'Maze 1')[0]
idx2 = np.where(SubData['Maze Type'] == 'Maze 2')[0]

params1, _ = curve_fit(kww_decay, SubData['Duration'][idx1], SubData['Global Recover Prob.'][idx1]/100,
                       bounds=[[0, 0, 0], [np.inf, np.inf, 1]],
                       p0=[0.5, 0.5, 0.5])
params2, _ = curve_fit(kww_decay, SubData['Duration'][idx2], SubData['Global Recover Prob.'][idx2]/100,
                       bounds=[[0, 0, 0], [np.inf, np.inf, 1]],
                       p0=[0.5, 0.5, 0.5])
print("Maze A&B global recover prob.")
print(params1)
print(params2, end='\n\n')

x1 = np.linspace(min(SubData['Duration'][idx1]), max(SubData['Duration'][idx1]), 10000)
x2 = np.linspace(min(SubData['Duration'][idx2]), max(SubData['Duration'][idx2]), 10000)
y1 = kww_decay(x1, *params1)
y2 = kww_decay(x2, *params2)
colors = sns.color_palette("rocket", 3)[1:]

fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x1-1, y1*100, color=colors[0], linewidth = 0.5)
ax.plot(x2-1, y2*100, color=colors[1], linewidth = 0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Global Recover Prob.',
    data=SubData,
    hue = "Maze Type",
    hue_order=['Maze 1', 'Maze 2'],
    palette = markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    alpha=0.9,
    jitter=0.1,
    dodge=True,
    ax = ax
)
sns.lineplot(
    x = ShufData['Duration']-1,
    y = ShufData['Global Recover Prob.'],
    data=ShufData,
    hue=ShufData['Maze Type'],
    palette=chancecolors,
    err_kws={"edgecolor": None},
    linewidth = 0.5,
    ax = ax
)
ax.set_ylim(-2, 40)
ax.set_yticks(np.linspace(0, 40, 5))

plt.savefig(join(loc, 'Global recover prob.png'), dpi = 600)
plt.savefig(join(loc, 'Global recover prob.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x1-1, y1*100, color=colors[0], linewidth = 0.5)
ax.plot(x2-1, y2*100, color=colors[1], linewidth = 0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Global Recover Prob.',
    data=SubData,
    hue = "Maze Type",
    hue_order=['Maze 1', 'Maze 2'],
    palette = markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    alpha=0.9,
    jitter=0.1,
    dodge=True,
    ax = ax
)
sns.lineplot(
    x = ShufData['Duration']-1,
    y = ShufData['Global Recover Prob.'],
    data=ShufData,
    hue=ShufData['Maze Type'],
    palette=chancecolors,
    err_kws={"edgecolor": None},
    linewidth = 0.5,
    ax = ax
)
ax.semilogy()
ax.set_ylim(0.01, 40)

plt.savefig(join(loc, 'Global recover prob [semilogy].png'), dpi = 600)
plt.savefig(join(loc, 'Global recover prob [semilogy].svg'), dpi = 600)
plt.close()

print("Maze A&B statistic test global recovery.")
for i in range(1, 23):
    print("Day ", i)
    print(levene(SubData['Global Recover Prob.'][np.where(SubData['Duration'] == i)[0]], 
                ShufData['Global Recover Prob.'][np.where(ShufData['Duration'] == i)[0]]))
    print(ttest_ind(SubData['Global Recover Prob.'][np.where(SubData['Duration'] == i)[0]], 
                ShufData['Global Recover Prob.'][np.where(ShufData['Duration'] == i)[0]], equal_var=False))
    
    
# Global Recovery Prob. Open Field
idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Global Recover Prob.']) == False)&
               (Data['Maze Type'] == 'Open Field')&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Global Recover Prob.']) == False)&
               (Data['Maze Type'] == 'Open Field')&
               (Data['Type'] == 'Shuffle'))[0]
ShufData = SubDict(Data, Data.keys(), idx=idx)

params, _ = curve_fit(kww_decay, SubData['Duration'], SubData['Global Recover Prob.']/100,
                       bounds=[[0, 0, 0], [np.inf, np.inf, 1]],
                       p0=[0.5, 0.5, 0.5])
print("Open Field global recover prob.")
print(params, end='\n\n')

x = np.linspace(min(SubData['Duration']), max(SubData['Duration']), 10000)
y = kww_decay(x, *params)
colors = sns.color_palette("rocket", 3)[:1]

fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x-1, y*100, color=colors[0], linewidth = 0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Global Recover Prob.',
    data=SubData,
    hue = "Maze Type",
    palette = markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    alpha=0.9,
    jitter=0.1,
    dodge=True,
    ax = ax
)
sns.lineplot(
    x = ShufData['Duration']-1,
    y = ShufData['Global Recover Prob.'],
    data=ShufData,
    hue=ShufData['Maze Type'],
    palette=chancecolors,
    err_kws={"edgecolor": None},
    linewidth = 0.5,
    ax = ax
)
ax.set_ylim(-2, 40)
ax.set_yticks(np.linspace(0, 40, 5))

plt.savefig(join(loc, '[Open Field] Global recover prob.png'), dpi = 600)
plt.savefig(join(loc, '[Open Field] Global recover prob.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x-1, y*100, color=colors[0], linewidth = 0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Global Recover Prob.',
    data=SubData,
    hue = "Maze Type",
    palette = markercolors,
    edgecolor='black',
    size=3,
    linewidth=0.15,
    alpha=0.9,
    jitter=0.1,
    dodge=True,
    ax = ax
)
sns.lineplot(
    x = ShufData['Duration']-1,
    y = ShufData['Global Recover Prob.'],
    data=ShufData,
    hue=ShufData['Maze Type'],
    palette=chancecolors,
    err_kws={"edgecolor": None},
    linewidth = 0.5,
    ax = ax
)
ax.semilogy()
ax.set_ylim(0.01, 40)

plt.savefig(join(loc, '[Open Field] Global recover prob [semilogy].png'), dpi = 600)
plt.savefig(join(loc, '[Open Field] Global recover prob [semilogy].svg'), dpi = 600)
plt.close()

print("Open Field Global recovery statistic test")
for i in range(1, 12):
    print("Day ",i)
    print(ttest_ind(SubData['Global Recover Prob.'][np.where(SubData['Duration'] == i)[0]], 
                ShufData['Global Recover Prob.'][np.where(ShufData['Duration'] == i)[0]], alternative='greater'))
print()


# Global Recovery Prob. Hairpin
idx = np.where((Data['Paradigm'] != 'CrossMaze')&
               (np.isnan(Data['Global Recover Prob.']) == False)&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

idx = np.where((Data['Paradigm'] != 'CrossMaze')&
               (np.isnan(Data['Global Recover Prob.']) == False)&
               (Data['Type'] == 'Shuffle'))[0]
ShufData = SubDict(Data, Data.keys(), idx=idx)


# Fit parameters
bounds = [[0, 0, 0], [np.inf, np.inf, 1]]
initial_guesses = [1, 1, 0.5]
idx = np.where(SubData['Paradigm'] == 'HairpinMaze cis')[0]
params1, _ = curve_fit(kww_decay, SubData['Duration'][idx], 
                       SubData['Global Recover Prob.'][idx]/100, 
                       bounds=bounds, 
                       p0=initial_guesses)
x1 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print("Hairpin&Reverse Global Recover Prob.")
print(params1)
y1 = kww_decay(x1, *params1)

idx = np.where(SubData['Paradigm'] == 'HairpinMaze trs')[0]
params2, _ = curve_fit(kww_decay, SubData['Duration'][idx], 
                       SubData['Global Recover Prob.'][idx]/100, 
                       bounds=bounds, 
                       p0=initial_guesses)
x2 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params2)
y2 = kww_decay(x2, *params2)

idx = np.where(SubData['Paradigm'] == 'ReverseMaze cis')[0]
params3, _ = curve_fit(kww_decay, SubData['Duration'][idx], 
                       SubData['Global Recover Prob.'][idx]/100, 
                       bounds=bounds, 
                       p0=initial_guesses)
x3 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params3)
y3 = kww_decay(x3, *params3)

idx = np.where(SubData['Paradigm'] == 'ReverseMaze trs')[0]
params4, _ = curve_fit(lambda x, b, c: kww_decay(x, 36, b, c), SubData['Duration'][idx], 
                       SubData['Global Recover Prob.'][idx]/100)
x4 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params4)


y4 = kww_decay(x4, 36, *params4)
colors = ['#6D9BA8', '#A3CBB2', '#E9D985', '#D57A66']

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x1-1, y1*100, color=colors[0], linewidth = 0.5)
ax.plot(x2-1, y2*100, color=colors[1], linewidth = 0.5)
ax.plot(x3-1, y3*100, color=colors[2], linewidth = 0.5)
ax.plot(x4-1, y4*100, color=colors[3], linewidth = 0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Global Recover Prob.',
    data=SubData,
    hue = "Paradigm",
    hue_order=['HairpinMaze cis', 'HairpinMaze trs', 'ReverseMaze cis', 'ReverseMaze trs'],
    palette = colors,
    edgecolor='black',
    size=2,
    linewidth=0.1,
    alpha=0.9,
    ax = ax,
    jitter=0.2
)
sns.lineplot(
    x = ShufData['Duration']-1,
    y = ShufData['Global Recover Prob.'],
    hue = ShufData['Paradigm'],
    hue_order=['HairpinMaze cis', 'HairpinMaze trs', 'ReverseMaze cis', 'ReverseMaze trs'],
    palette = chancecolors,
    linewidth = 0.5,
    err_kws={"edgecolor": None},
    ax = ax
)
ax.set_ylim(-1, 40)
ax.set_yticks(np.linspace(0, 40, 5))

plt.savefig(join(loc, '[Hairpin&Reverse] global recover prob.png'), dpi = 600)
plt.savefig(join(loc, '[Hairpin&Reverse] global recover prob.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x1-1, y1*100, color=colors[0], linewidth = 0.5)
ax.plot(x2-1, y2*100, color=colors[1], linewidth = 0.5)
ax.plot(x3-1, y3*100, color=colors[2], linewidth = 0.5)
ax.plot(x4-1, y4*100, color=colors[3], linewidth = 0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Global Recover Prob.',
    data=SubData,
    hue = "Paradigm",
    hue_order=['HairpinMaze cis', 'HairpinMaze trs', 'ReverseMaze cis', 'ReverseMaze trs'],
    palette = colors,
    edgecolor='black',
    size=2,
    linewidth=0.1,
    alpha=0.9,
    ax = ax,
    jitter=0.2
)
sns.lineplot(
    x = ShufData['Duration']-1,
    y = ShufData['Global Recover Prob.'],
    hue = ShufData['Paradigm'],
    hue_order=['HairpinMaze cis', 'HairpinMaze trs', 'ReverseMaze cis', 'ReverseMaze trs'],
    palette = chancecolors,
    linewidth = 0.5,
    err_kws={"edgecolor": None},
    ax = ax
)
ax.semilogy()
ax.set_ylim(0.01, 40)

plt.savefig(join(loc, '[Hairpin&Reverse] global recover prob [semilogy].png'), dpi = 600)
plt.savefig(join(loc, '[Hairpin&Reverse] global recover prob [semilogy].svg'), dpi = 600)
plt.close()

print("HairpinMaze cis Global Recovery Statistic Test")
for i in range(1, 11):
    print("Day ",i)
    print(ttest_ind(SubData['Global Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'HairpinMaze cis'))[0]],
                    ShufData['Global Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'HairpinMaze cis'))[0]]))
print()
print("HairpinMaze trs Recovery Statistic Test")
for i in range(1, 11):
    print("Day ",i)
    print(ttest_ind(SubData['Global Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'HairpinMaze trs'))[0]],
                    ShufData['Global Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'HairpinMaze trs'))[0]]))
print()
print("Reverse cis Recovery Statistic Test")
for i in range(1, 11):
    print("Day ",i)
    print(ttest_ind(SubData['Global Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'ReverseMaze cis'))[0]],
                    ShufData['Global Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'ReverseMaze cis'))[0]]))
print()
print("Reverse trs Recovery Statistic Test")
for i in range(1, 11):
    print("Day ",i)
    print(ttest_ind(SubData['Global Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'ReverseMaze trs'))[0]],
                    ShufData['Global Recover Prob.'][np.where((SubData['Duration'] == i)&(SubData['Paradigm'] == 'ReverseMaze trs'))[0]]))
print()