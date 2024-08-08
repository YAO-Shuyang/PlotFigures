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

print("Global Recover Prob.: ---------------------------------------------------------")
print("Maze A")
idx1m1 = np.where((Data['Paradigm'] == 'CrossMaze')&
                  (Data['Maze Type'] == 'Maze 1')&
                  (Data['Duration'] <= 5)&(Data['Duration'] >= 1)&
                  (Data['Type'] == 'Real'))[0]
idx1m1_shuf = np.where((Data['Paradigm'] == 'CrossMaze')&
                  (Data['Maze Type'] == 'Maze 1')&
                  (Data['Duration'] <= 5)&(Data['Duration'] >= 1)&
                  (Data['Type'] == 'Shuffle'))[0]
glob = Data['Global Recover Prob.'][idx1m1] - Data['Global Recover Prob.'][idx1m1_shuf]
glob = np.reshape(glob, [6, 5])
micewise = np.sum(glob, axis=1)
print_estimator(micewise)
print()
print("Maze B")
idx1m2 = np.where((Data['Paradigm'] == 'CrossMaze')&
                  (Data['Maze Type'] == 'Maze 2')&
                  (Data['Duration'] <= 5)&(Data['Duration'] >= 1)&
                  (Data['Type'] == 'Real'))[0]
idx1m2_shuf = np.where((Data['Paradigm'] == 'CrossMaze')&
                  (Data['Maze Type'] == 'Maze 2')&
                  (Data['Duration'] <= 5)&(Data['Duration'] >= 1)&
                  (Data['Type'] == 'Shuffle'))[0]
glob = Data['Global Recover Prob.'][idx1m2] - Data['Global Recover Prob.'][idx1m2_shuf]
glob = np.reshape(glob, [4, 5])
micewise = np.sum(glob, axis=1)
print_estimator(micewise)
print()

print("MA forward:")
idx_rmp_cis = np.where((Data['Paradigm'] == 'ReverseMaze cis')&
                  (Data['Duration'] <= 5)&(Data['Duration'] >= 1)&
                  (Data['Type'] == 'Real'))[0]
idx_rmp_cis_shuf = np.where((Data['Paradigm'] == 'ReverseMaze cis')&
                  (Data['Duration'] <= 5)&(Data['Duration'] >= 1)&
                  (Data['Type'] == 'Shuffle'))[0]
glob = Data['Global Recover Prob.'][idx_rmp_cis] - Data['Global Recover Prob.'][idx_rmp_cis_shuf]
glob = np.reshape(glob, [4, 5])
micewise = np.sum(glob, axis=1)
print_estimator(micewise)
print()

print("MA backward:")
idx_rmp_trs = np.where((Data['Paradigm'] == 'ReverseMaze trs')&
                  (Data['Duration'] <= 5)&(Data['Duration'] >= 1)&
                  (Data['Type'] == 'Real'))[0]
idx_rmp_trs_shuf = np.where((Data['Paradigm'] == 'ReverseMaze trs')&
                  (Data['Duration'] <= 5)&(Data['Duration'] >= 1)&
                  (Data['Type'] == 'Shuffle'))[0]
glob = Data['Global Recover Prob.'][idx_rmp_trs] - Data['Global Recover Prob.'][idx_rmp_trs_shuf]
glob = np.reshape(glob, [4, 5])
micewise = np.sum(glob, axis=1)
print_estimator(micewise)
print()

print("HP forward:")
idx_hmp_cis = np.where((Data['Paradigm'] == 'HairpinMaze cis')&
                  (Data['Duration'] <= 5)&(Data['Duration'] >= 1)&
                  (Data['Type'] == 'Real'))[0]
idx_hmp_cis_shuf = np.where((Data['Paradigm'] == 'HairpinMaze cis')&
                  (Data['Duration'] <= 5)&(Data['Duration'] >= 1)&
                  (Data['Type'] == 'Shuffle'))[0]
glob = Data['Global Recover Prob.'][idx_hmp_cis] - Data['Global Recover Prob.'][idx_hmp_cis_shuf]
glob = np.reshape(glob, [4, 5])
micewise = np.sum(glob, axis=1)
print_estimator(micewise)
print()

print("HP backward:")
idx_hmp_trs = np.where((Data['Paradigm'] == 'HairpinMaze trs')&
                  (Data['Duration'] <= 5)&(Data['Duration'] >= 1)&
                  (Data['Type'] == 'Real'))[0]
idx_hmp_trs_shuf = np.where((Data['Paradigm'] == 'HairpinMaze trs')&    
                  (Data['Duration'] <= 5)&(Data['Duration'] >= 1)&
                  (Data['Type'] == 'Shuffle'))[0]
glob = Data['Global Recover Prob.'][idx_hmp_trs] - Data['Global Recover Prob.'][idx_hmp_trs_shuf]
glob = np.reshape(glob, [4, 5])
micewise = np.sum(glob, axis=1)
print_estimator(micewise)
print()
                       


print("Conditional probability ---------------------------------------------------")
idx_m1_d1 = np.where((Data['Paradigm'] == 'CrossMaze')&
                      (Data['Maze Type'] == 'Maze 1')&
                      (Data['Duration'] == 1)&
                      ((Data['Stage'] != 'Stage 1') | (Data['MiceID'] != 10212))&
                      (Data['Type'] == 'Real'))[0]
idx_m1_d11 = np.where((Data['Paradigm'] == 'CrossMaze')&
                      (Data['Maze Type'] == 'Maze 1')&
                      ((Data['Stage'] != 'Stage 1') | (Data['MiceID'] != 10212))&
                      (Data['Duration'] == 12)&
                      (Data['Type'] == 'Real'))[0]

idx_m2_d1 = np.where((Data['Paradigm'] == 'CrossMaze')&
                      (Data['Maze Type'] == 'Maze 2')&
                      (Data['Duration'] == 1)&
                      (Data['Type'] == 'Real'))[0]
idx_m2_d12 = np.where((Data['Paradigm'] == 'CrossMaze')&
                      (Data['Maze Type'] == 'Maze 2')&
                      (Data['Duration'] == 12)&
                      (Data['Type'] == 'Real'))[0]
print("Maze 1")
print_estimator(Data['Conditional Prob.'][idx_m1_d1])
print_estimator(Data['Conditional Prob.'][idx_m1_d11])
print(ttest_rel(Data['Conditional Prob.'][idx_m1_d1], Data['Conditional Prob.'][idx_m1_d11]), end='\n\n')

print("Maze 2")
print_estimator(Data['Conditional Prob.'][idx_m2_d1])
print_estimator(Data['Conditional Prob.'][idx_m2_d12])
print(ttest_rel(Data['Conditional Prob.'][idx_m2_d1], Data['Conditional Prob.'][idx_m2_d12]), end='\n\n')

idx_rmp_cis1 = np.where((Data['Paradigm'] == 'ReverseMaze cis')&
                      (Data['Duration'] == 1)&
                      (Data['Type'] == 'Real'))[0]
idx_rmp_cis6 = np.where((Data['Paradigm'] == 'ReverseMaze cis')&
                      (Data['Maze Type'] == 'Maze 1')&
                      (Data['Duration'] == 6)&
                      (Data['Type'] == 'Real'))[0]
print("Reverse Maze cis")
print_estimator(Data['Conditional Prob.'][idx_rmp_cis1])
print_estimator(Data['Conditional Prob.'][idx_rmp_cis6])
print(ttest_rel(Data['Conditional Prob.'][idx_rmp_cis1], Data['Conditional Prob.'][idx_rmp_cis6]), end='\n\n')

idx_rmp_trs1 = np.where((Data['Paradigm'] == 'ReverseMaze trs')&
                      (Data['Duration'] == 1)&
                      (Data['Type'] == 'Real'))[0]
idx_rmp_trs6 = np.where((Data['Paradigm'] == 'ReverseMaze trs')&
                      (Data['Duration'] == 6)&
                      (Data['Type'] == 'Real'))[0]
print("Reverse Maze trans")
print_estimator(Data['Conditional Prob.'][idx_rmp_trs1])
print_estimator(Data['Conditional Prob.'][idx_rmp_trs6])
print(ttest_rel(Data['Conditional Prob.'][idx_rmp_trs1], Data['Conditional Prob.'][idx_rmp_trs6]), end='\n\n')


# Hairpin maze
idx_hmp_cis1 = np.where((Data['Paradigm'] == 'HairpinMaze cis')&
                      (Data['Duration'] == 1)&
                      (Data['Type'] == 'Real'))[0]
idx_hmp_cis6 = np.where((Data['Paradigm'] == 'HairpinMaze cis')&
                      (Data['Duration'] == 6)&
                      (Data['Type'] == 'Real'))[0]
print("Hairpin Maze cis")
print_estimator(Data['Conditional Prob.'][idx_hmp_cis1])
print_estimator(Data['Conditional Prob.'][idx_hmp_cis6])
print(ttest_rel(Data['Conditional Prob.'][idx_hmp_cis1], Data['Conditional Prob.'][idx_hmp_cis6]), end='\n\n')

idx_hmp_trs1 = np.where((Data['Paradigm'] == 'HairpinMaze trs')&
                      (Data['Duration'] == 1)&
                      (Data['Type'] == 'Real'))[0]
idx_hmp_trs6 = np.where((Data['Paradigm'] == 'HairpinMaze trs')&
                      (Data['Duration'] == 6)&
                      (Data['Type'] == 'Real'))[0]
print("Hairpin Maze trans")
print_estimator(Data['Conditional Prob.'][idx_hmp_trs1])
print_estimator(Data['Conditional Prob.'][idx_hmp_trs6])
print(ttest_rel(Data['Conditional Prob.'][idx_hmp_trs1], Data['Conditional Prob.'][idx_hmp_trs6]), end='\n\n')

print("Conditional recovery probability ---------------------------------------------------")
idx_m1_d1 = np.where((Data['Paradigm'] == 'CrossMaze')&
                      (Data['Maze Type'] == 'Maze 1')&
                      (Data['Duration'] == 1)&
                      (Data['Type'] == 'Real'))[0]
idx_m1_d11 = np.where((Data['Paradigm'] == 'CrossMaze')&
                      (Data['Maze Type'] == 'Maze 1')&
                      (Data['Duration'] == 8)&
                      (Data['Type'] == 'Real'))[0]

idx_m2_d1 = np.where((Data['Paradigm'] == 'CrossMaze')&
                      (Data['Maze Type'] == 'Maze 2')&
                      (Data['Duration'] == 1)&
                      (Data['Type'] == 'Real'))[0]
idx_m2_d12 = np.where((Data['Paradigm'] == 'CrossMaze')&
                      (Data['Maze Type'] == 'Maze 2')&
                      (Data['Duration'] == 8)&
                      (Data['Type'] == 'Real'))[0]
print("Maze 1")
print_estimator(Data['Conditional Recover Prob.'][idx_m1_d1])
print_estimator(Data['Conditional Recover Prob.'][idx_m1_d11])
print(ttest_rel(Data['Conditional Recover Prob.'][idx_m1_d1], Data['Conditional Recover Prob.'][idx_m1_d11]), end='\n\n')

print("Maze 2")
print_estimator(Data['Conditional Recover Prob.'][idx_m2_d1])
print_estimator(Data['Conditional Recover Prob.'][idx_m2_d12])
print(ttest_rel(Data['Conditional Recover Prob.'][idx_m2_d1], Data['Conditional Recover Prob.'][idx_m2_d12]), end='\n\n')

idx_rmp_cis1 = np.where((Data['Paradigm'] == 'ReverseMaze cis')&
                      (Data['Duration'] == 1)&
                      (Data['Type'] == 'Real'))[0]
idx_rmp_cis6 = np.where((Data['Paradigm'] == 'ReverseMaze cis')&
                      (Data['Maze Type'] == 'Maze 1')&
                      (Data['Duration'] == 5)&
                      (Data['Type'] == 'Real'))[0]
print("Reverse Maze cis")
print_estimator(Data['Conditional Recover Prob.'][idx_rmp_cis1])
print_estimator(Data['Conditional Recover Prob.'][idx_rmp_cis6])
print(ttest_rel(Data['Conditional Recover Prob.'][idx_rmp_cis1], Data['Conditional Recover Prob.'][idx_rmp_cis6]), end='\n\n')

idx_rmp_trs1 = np.where((Data['Paradigm'] == 'ReverseMaze trs')&
                      (Data['Duration'] == 1)&
                      (Data['Type'] == 'Real'))[0]
idx_rmp_trs6 = np.where((Data['Paradigm'] == 'ReverseMaze trs')&
                      (Data['Duration'] == 5)&
                      (Data['Type'] == 'Real'))[0]
print("Reverse Maze trans")
print_estimator(Data['Conditional Recover Prob.'][idx_rmp_trs1])
print_estimator(Data['Conditional Recover Prob.'][idx_rmp_trs6])
print(ttest_rel(Data['Conditional Recover Prob.'][idx_rmp_trs1], Data['Conditional Recover Prob.'][idx_rmp_trs6]), end='\n\n')


# Hairpin maze
idx_hmp_cis1 = np.where((Data['Paradigm'] == 'HairpinMaze cis')&
                      (Data['Duration'] == 1)&
                      (Data['Type'] == 'Real'))[0]
idx_hmp_cis6 = np.where((Data['Paradigm'] == 'HairpinMaze cis')&
                      (Data['Duration'] == 5)&
                      (Data['Type'] == 'Real'))[0]
print("Hairpin Maze cis")
print_estimator(Data['Conditional Recover Prob.'][idx_hmp_cis1])
print_estimator(Data['Conditional Recover Prob.'][idx_hmp_cis6])
print(ttest_rel(Data['Conditional Recover Prob.'][idx_hmp_cis1], Data['Conditional Recover Prob.'][idx_hmp_cis6]), end='\n\n')

idx_hmp_trs1 = np.where((Data['Paradigm'] == 'HairpinMaze trs')&
                      (Data['Duration'] == 1)&
                      (Data['Type'] == 'Real'))[0]
idx_hmp_trs6 = np.where((Data['Paradigm'] == 'HairpinMaze trs')&
                      (Data['Duration'] == 5)&
                      (Data['Type'] == 'Real'))[0]
print("Hairpin Maze trans")
print_estimator(Data['Conditional Recover Prob.'][idx_hmp_trs1])
print_estimator(Data['Conditional Recover Prob.'][idx_hmp_trs6])
print(ttest_rel(Data['Conditional Recover Prob.'][idx_hmp_trs1], Data['Conditional Recover Prob.'][idx_hmp_trs6]), end='\n\n')

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
ax.set_yticks(np.linspace(0, 40, 9))

plt.savefig(join(loc, 'Conditional recover prob.png'), dpi = 600)
plt.savefig(join(loc, 'Conditional recover prob.svg'), dpi = 600)
plt.close()

print("Maze A&B Conditional Recovery Prob. -----------------------------------------")
for i in range(1, 25):
    print("Day ",i)
    print(ttest_rel(SubData['Conditional Recover Prob.'][SubData['Duration'] == i], 
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

"""
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
    print(ttest_rel(SubData['Conditional Recover Prob.'][SubData['Duration'] == i], 
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
"""
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
bounds = [[0, 0, 0], [np.inf, np.inf, np.inf]]
initial_guesses = [2, 0.1, 0.35]
idx = np.where(SubData['Paradigm'] == 'HairpinMaze cis')[0]
params1, _ = curve_fit(lambda x, b, c: kww_decay(x, 1, b, c), SubData['Duration'][idx], 
                       SubData['Conditional Recover Prob.'][idx]/100, p0=[0.1, 0.35])
x1 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params1)
y1 = kww_decay(x1, 1, *params1)

idx = np.where(SubData['Paradigm'] == 'HairpinMaze trs')[0]
params2, _ = curve_fit(lambda x, b, c: kww_decay(x, 1, b, c), SubData['Duration'][idx], 
                       SubData['Conditional Recover Prob.'][idx]/100, p0=[0.1, 0.35])
x2 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params2)
y2 = kww_decay(x2, 1, *params2)

idx = np.where(SubData['Paradigm'] == 'ReverseMaze cis')[0]
params3, _ = curve_fit(lambda x, b, c: kww_decay(x, 1, b, c), SubData['Duration'][idx], 
                       SubData['Conditional Recover Prob.'][idx]/100, p0=[0.1, 0.35])
x3 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params3)
y3 = kww_decay(x3, 1, *params3)

idx = np.where(SubData['Paradigm'] == 'ReverseMaze trs')[0]
params4, _ = curve_fit(lambda x, b, c: kww_decay(x, 1, b, c), SubData['Duration'][idx], 
                       SubData['Conditional Recover Prob.'][idx]/100, p0=[0.1, 0.35])
x4 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params4)
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
ax.set_ylim(-1, 30)
ax.set_yticks(np.linspace(0, 30, 7))

plt.savefig(join(loc, '[Hairpin&Reverse] conditional recover prob.png'), dpi = 600)
plt.savefig(join(loc, '[Hairpin&Reverse] conditional recover prob.svg'), dpi = 600)
plt.close()

print("HP & MA reversed paradigms - Recovery Statistic Test")
for i in range(1, 11):
    print("Day ",i)
    print(ttest_rel(SubData['Conditional Recover Prob.'][np.where((SubData['Duration'] == i))[0]],
                    ShufData['Conditional Recover Prob.'][np.where((SubData['Duration'] == i))[0]]))
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

print()
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
ax.set_ylim(-1, 40)
ax.set_yticks(np.linspace(0, 40, 9))

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
for i in range(1, 14):
    print("Day ", i)
    print(levene(SubData['Global Recover Prob.'][np.where(SubData['Duration'] == i)[0]], 
                ShufData['Global Recover Prob.'][np.where(ShufData['Duration'] == i)[0]]))
    try:
        print(ttest_rel(SubData['Global Recover Prob.'][np.where(SubData['Duration'] == i)[0]], 
                    ShufData['Global Recover Prob.'][np.where(ShufData['Duration'] == i)[0]]))
    except:
        pass
    
"""
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
    print(ttest_rel(SubData['Global Recover Prob.'][np.where(SubData['Duration'] == i)[0]], 
                ShufData['Global Recover Prob.'][np.where(ShufData['Duration'] == i)[0]], alternative='greater'))
print()
"""

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
params1, _ = curve_fit(lambda x, b, c: kww_decay(x, 5, b, c), SubData['Duration'][idx], 
                       SubData['Global Recover Prob.'][idx]/100, p0=[0.05, 0.4])
x1 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params1)
y1 = kww_decay(x1, 5, *params1)

idx = np.where(SubData['Paradigm'] == 'HairpinMaze trs')[0]
params2, _ = curve_fit(lambda x, b, c: kww_decay(x, 5, b, c), SubData['Duration'][idx], 
                       SubData['Global Recover Prob.'][idx]/100, p0=[0.05, 0.4])
x2 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params2)
y2 = kww_decay(x2, 5, *params2)

idx = np.where(SubData['Paradigm'] == 'ReverseMaze cis')[0]
params3, _ = curve_fit(lambda x, b, c: kww_decay(x, 5, b, c), SubData['Duration'][idx], 
                       SubData['Global Recover Prob.'][idx]/100, p0=[0.05, 0.4])
x3 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params3)
y3 = kww_decay(x3, 5, *params3)

idx = np.where(SubData['Paradigm'] == 'ReverseMaze trs')[0]
params4, _ = curve_fit(lambda x, b, c: kww_decay(x, 5, b, c), SubData['Duration'][idx], 
                       SubData['Global Recover Prob.'][idx]/100, p0=[0.05, 0.4])
x4 = np.linspace(min(SubData['Duration'][idx]), max(SubData['Duration'][idx]), 10000)
print(params4)
y4 = kww_decay(x4, 5, *params4)

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
ax.set_ylim(-1, 30)
ax.set_yticks(np.linspace(0, 30, 7))

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
ax.set_ylim(0.01, 30)

plt.savefig(join(loc, '[Hairpin&Reverse] global recover prob [semilogy].png'), dpi = 600)
plt.savefig(join(loc, '[Hairpin&Reverse] global recover prob [semilogy].svg'), dpi = 600)
plt.close()

print("All test")
for i in range(1, 11):
    print("Day ",i)
    print(ttest_rel(SubData['Global Recover Prob.'][np.where((SubData['Duration'] == i))[0]],
                    ShufData['Global Recover Prob.'][np.where((SubData['Duration'] == i))[0]]))
print()