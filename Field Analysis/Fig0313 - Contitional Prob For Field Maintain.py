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
            params[2].append(params0[2])
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
        print("  a: ", np.mean(params[0]), np.std(params[0]))
        print("           ", params[0])
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
idx1 = np.where(SubData['Maze Type'] == 'Maze 1')[0]
idx2 = np.where(SubData['Maze Type'] == 'Maze 2')[0]
params1, _ = curve_fit(kww_decay, SubData['Duration'][idx1], SubData['Conditional Recover Prob.'][idx1]/100)
print("Conditional Recovery prob. -----------------------------------------")
x1 = np.linspace(min(SubData['Duration'][idx1]), max(SubData['Duration'][idx1]), 10000)
print("Maze 1", params1)
params2, _ = curve_fit(kww_decay, SubData['Duration'][idx2], SubData['Conditional Recover Prob.'][idx2]/100)
x2 = np.linspace(min(SubData['Duration'][idx2]), max(SubData['Duration'][idx2]), 10000)
print("Maze 2", params2)

y1 = kww_decay(x1, params1[0], params1[1], params1[2])
y2 = kww_decay(x2, params2[0], params2[1], params2[2])


idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Conditional Recover Prob.']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Type'] == 'Shuffle'))[0]
ShufData = SubDict(Data, Data.keys(), idx=idx)
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

ax.set_ylim(-1, 45)
ax.set_yticks(np.linspace(0, 45, 10))

plt.savefig(join(loc, 'Conditional recover prob.png'), dpi = 600)
plt.savefig(join(loc, 'Conditional recover prob.svg'), dpi = 600)
plt.close()

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
params2, _ = curve_fit(polynomial_converge2, SubData['Duration'], SubData['Conditional Prob.']/100)
x2 = np.linspace(min(SubData['Duration']), max(SubData['Duration']), 10000)
print(params2)

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
k2, b2, c2 = params2

y2 = polynomial_converge2(x2, k2, b2, c2)
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
ax.set_ylim(1, 40000)
plt.savefig(join(loc, '[Open Field] on-next num.png'), dpi = 600)
plt.savefig(join(loc, '[Open Field] on-next num.svg'), dpi = 600)
plt.close()

# Global Recovery

idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Global Recover Prob.']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
"""
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.where((Data['Maze Type'] == "Maze 1")&(np.isnan(Data['Conditional Prob.']) == False))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
a1, b1 = SigmoidFit(SubData['Duration'], SubData['Conditional Prob.']/100)
x1 = np.linspace(1, 23, 26001)
y1 = Sigmoid(x1, a1, b1)
print(f"Maze 1: {a1:.3f}, {b1:.3f}")
ax.plot(x1, y1*100, color=sns.color_palette("rocket", 3)[1], linewidth=0.5)
idx = np.where((Data['Maze Type'] == "Maze 2")&(np.isnan(Data['Conditional Prob.']) == False))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
a2, b2 = SigmoidFit(SubData['Duration'], SubData['Conditional Prob.']/100)
x2 = np.linspace(1, 12, 26001)
y2 = Sigmoid(x2, a2, b2)
print(f"Maze 2: {a2:.3f}, {b2:.3f}")
ax.plot(x2, y2*100, color=sns.color_palette("rocket", 3)[2], linewidth=0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Prob.',
    data=Data,
    hue = "Maze Type",
    palette = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[3]],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 103)
ax.set_yticks(np.linspace(0, 100, 6))

plt.savefig(join(loc, 'conditional prob.png'), dpi = 600)
plt.savefig(join(loc, 'conditional prob.svg'), dpi = 600)
plt.close()


fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
print(np.cumprod(y1), y2)
x1 = np.linspace(1, 23, 23)
y1 = Sigmoid(x1, a1, b1)
x2 = np.linspace(1, 12, 12)
y2 = Sigmoid(x2, a2, b2)
y3 = np.concatenate([[100], np.cumprod(y1)*100])
y4 = np.concatenate([[100], np.cumprod(y2)*100])
ax.plot(np.linspace(0, 23, 24), y3, color=sns.color_palette("rocket", 3)[1], linewidth=0.5)
ax.plot(np.linspace(0, 12, 13), y4, color=sns.color_palette("rocket", 3)[2], linewidth=0.5)
x = np.linspace(1, 1000, 1000)
y1 = np.cumprod(Sigmoid(x, a1, b1))
y2 = np.cumprod(Sigmoid(x, a2, b2))
print(y1[-1], y2[-1])
sns.stripplot(
    x = 'Duration',
    y = 'Cumulative Prob.',
    data=Data,
    hue = "Maze Type",
    palette = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[3]],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 103)
ax.set_yticks(np.linspace(0, 100, 6))

plt.savefig(join(loc, 'cumulative conditional prob.png'), dpi = 600)
plt.savefig(join(loc, 'cumulative conditional prob.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(4,2))
Data['Recovered Prob.'][Data['Recovered Prob.'] == 0] = np.nan
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Duration',
    y = 'Recovered Prob.',
    data=Data,
    hue = "Maze Type",
    palette=sns.color_palette("rocket", 3)[1:],
    ax=ax,
    markeredgecolor=None,
    errorbar='se',
    legend=False,
    err_kws={'edgecolor':None},
    linewidth=0.5,
)
sns.stripplot(
    x = 'Duration',
    y = 'Recovered Prob.',
    data=Data,
    hue = "Maze Type",
    palette = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[3]],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(-3, 30)
ax.set_yticks(np.linspace(0, 30, 7))

plt.savefig(join(loc, 'recover prob.png'), dpi = 600)
plt.savefig(join(loc, 'recover prob.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(4,2))
Data['Recovered Prob.'][Data['Recovered Prob.'] == 0] = np.nan
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Duration',
    y = 'Recovered Prob.',
    data=Data,
    hue = "Maze Type",
    palette=sns.color_palette("rocket", 3)[1:],
    ax=ax,

    markeredgecolor=None,
    errorbar='se',
    legend=False,
    err_kws={'edgecolor':None},
    linewidth=0.5,
)
sns.stripplot(
    x = 'Duration',
    y = 'Recovered Prob.',
    data=Data,
    hue = "Maze Type",
    palette = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[3]],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0.001, 30)
ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 20, 30])
ax.semilogy()

plt.savefig(join(loc, 'recover prob semilog.png'), dpi = 600)
plt.savefig(join(loc, 'recover prob semilog.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'No Detect Prob.',
    data=Data,
    hue = "Maze Type",
    palette = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[3]],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 100)
ax.set_yticks(np.linspace(0, 100, 6))
plt.close()

fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Re-detect Active Prob.',
    data=Data,
    hue = "Maze Type",
    palette = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[3]],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(-3, 70)
ax.set_yticks(np.linspace(0, 70, 8))
ax.set_xlabel("Not detected duration / session")
plt.show()
"""