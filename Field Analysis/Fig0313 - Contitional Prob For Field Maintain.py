from mylib.statistic_test import *
from mylib.field.field_tracker import conditional_prob, conditional_prob_jumpnan

code_id = "0313 - Conditional Probability for Field Maintain"
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = [
                        'Duration', 'Conditional Prob.', 'No Detect Prob.', 
                        'Recover Prob.', 'Re-detect Active Prob.', 'Re-detect Prob.', 
                        'Cumulative Prob.', 'Paradigm', 'On-Next Num'], 
                        f = f_CellReg_modi, 
                        function = ConditionalProb_Interface, 
                        file_name = code_id, 
                        behavior_paradigm = 'CrossMaze')

from scipy.optimize import curve_fit
def exp_func(x, k, b):
    return 1 - np.exp(-k * (x-b))

def kww_converge(x, a, b, c):
    return 1 - a*np.exp(-np.power(x/b, c))

def powerlaw_decay(x, k, b):
    return k*np.power(x, b)

def kww_decay(x, a, b, c):
    return a*np.exp(-np.power(x/b, c))

def report_para(Data, name: str = 'KWW', key: str = 'Conditional Prob.'):
    mazes = ['Maze 1', 'Maze 2', 'Open Field']
    mice = [10209, 10212, 10224, 10227]
    
    if name == 'KWW':
        params = [[], [], []]
    elif name == 'exp':
        params = [[], []]
        
    for mouse in mice:
        for maze in mazes:
            idx = np.where((Data['Paradigm'] == 'CrossMaze')&(Data['Maze Type'] == maze)&(Data['MiceID'] == mouse)&(np.isnan(Data[key]) == False))[0]
            if len(idx) == 0:
                continue
            if name == 'KWW':
                params0, _ = curve_fit(kww_decay, Data['Duration'][idx], Data[key][idx]/100)
                params[0].append(params0[0])
                params[1].append(params0[1])
                params[2].append(params0[2])
            elif name == 'exp':
                params0, _ = curve_fit(exp_func, Data['Duration'][idx], Data[key][idx]/100)
                params[0].append(params0[0])
                params[1].append(params0[1])
                
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
    

report_para(Data, name='exp', key='Conditional Prob.')
report_para(Data, name='KWW', key='Recover Prob.')

colors = sns.color_palette("rocket", 3)[1:]
markercolors = [sns.color_palette("Blues", 3)[1], sns.color_palette("Blues", 3)[2]]

#Data['hue'] = np.array([Data['Papadigm'][i] + ' ' + Data['Maze Type'][i] for i in range(Data['Duration'].shape[0])])
idx = np.where((Data['Paradigm'] == 'CrossMaze')&(np.isnan(Data['Conditional Prob.']) == False)&(Data['Maze Type'] != 'Open Field'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
idx1 = np.where(SubData['Maze Type'] == 'Maze 1')[0]
idx2 = np.where(SubData['Maze Type'] == 'Maze 2')[0]

initial_guesses = [0.9, 10, 0.5]  # Adjust these based on your data
# Bounds for the parameters: a between 0 and 1, b positive, c between 0 and 1
bounds = ([0, 0, 0], [np.inf, np.inf, 1])
params1, _ = curve_fit(exp_func, SubData['Duration'][idx1], SubData['Conditional Prob.'][idx1]/100)
x1 = np.linspace(min(SubData['Duration'][idx1]), max(SubData['Duration'][idx1]), 10000)
print(params1)
params2, _ = curve_fit(exp_func, SubData['Duration'][idx2], SubData['Conditional Prob.'][idx2]/100)
x2 = np.linspace(min(SubData['Duration'][idx2]), max(SubData['Duration'][idx2]), 10000)
print(params2)

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
k1, b1 = params1
k2, b2 = params2

y1 = exp_func(x1, k1, b1)
y2 = exp_func(x2, k2, b2)
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
ax.set_ylim(1, 40000)
plt.savefig(join(loc, 'on-next num.png'), dpi = 600)
plt.savefig(join(loc, 'on-next num.svg'), dpi = 600)
plt.close()


fig = plt.figure(figsize=(4,2))
idx = np.where((Data['Paradigm'] == 'CrossMaze')&(np.isnan(Data['Recover Prob.']) == False)&(Data['Maze Type'] != 'Open Field'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
idx1 = np.where(SubData['Maze Type'] == 'Maze 1')[0]
idx2 = np.where(SubData['Maze Type'] == 'Maze 2')[0]
params1, _ = curve_fit(kww_decay, SubData['Duration'][idx1], SubData['Recover Prob.'][idx1]/100)
print("Recovery prob. -----------------------------------------")
x1 = np.linspace(min(SubData['Duration'][idx1]), max(SubData['Duration'][idx1]), 10000)
print("Maze 1", params1)
params2, _ = curve_fit(kww_decay, SubData['Duration'][idx2], SubData['Recover Prob.'][idx2]/100)
x2 = np.linspace(min(SubData['Duration'][idx2]), max(SubData['Duration'][idx2]), 10000)
print("Maze 2", params2)

y1 = kww_decay(x1, params1[0], params1[1], params1[2])
y2 = kww_decay(x2, params2[0], params2[1], params2[2])


fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x1-1, y1*100, color=colors[0], label='Maze 1', linewidth = 0.5)
ax.plot(x2-1, y2*100, color=colors[1], label='Maze 2', linewidth = 0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Recover Prob.',
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
ax.set_ylim(-3, 30)
ax.set_yticks(np.linspace(0, 30, 7))

plt.savefig(join(loc, 'recover prob.png'), dpi = 600)
plt.savefig(join(loc, 'recover prob.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x1-1, y1*100, color=colors[0], label='Maze 1', linewidth = 0.5)
ax.plot(x2-1, y2*100, color=colors[1], label='Maze 2', linewidth = 0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Recover Prob.',
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
ax.semilogy()
ax.set_ylim(0.0001, 30)
ax.set_yticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

plt.savefig(join(loc, 'recover prob semilog.png'), dpi = 600)
plt.savefig(join(loc, 'recover prob semilog.svg'), dpi = 600)
plt.close()
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