from mylib.statistic_test import *

code_id = '0322 - Fraction of Survival Place Field'
loc = join(figpath, code_id)
mkdir(loc)

f = pd.read_excel(join(figdata, "0300 - Kinetic Model Simulation.xlsx"))

if os.path.exists(join(figdata, code_id+' [real data].pkl')):
    with open(join(figdata, code_id+' [real data].pkl'), 'rb') as handle:
        RData = pickle.load(handle)
else:
    RData = DataFrameEstablish(variable_names = [
                        'Session Interval', 'Start Session', 'Survival Frac.', 'Paradigm'], 
                        f = f_CellReg_modi, f_member=['Type'],
                        function = SurvivalField_Fraction_Data_Interface, 
                        file_name = code_id + ' [real data]', 
                        behavior_paradigm = 'CrossMaze')


def r2(y_true, y_pred):
    # Calculate the total sum of squares (TSS)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    # Calculate the residual sum of squares (RSS)
    rss = np.sum((y_true - y_pred) ** 2)
    # Calculate R-squared
    r_squared = 1 - (rss / tss)
    return r_squared

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def const(t0, c):
    return np.repeat(c, len(t0))

def const_b(t0, c, dt):
    y = np.zeros(t0.shape[0])
    for i in range(t0.shape[0]):
        y[i] = np.prod(const(np.arange(int(t0[i]), int(t0[i]+dt)), c))
    return y

def exp_func(t0, k, b):
    return 1 - np.exp(-k*(t0-b))

def exp_func_b(t0, k, b, dt):
    y = np.zeros(t0.shape[0])
    for i in range(t0.shape[0]):
        y[i] = np.prod(exp_func(np.arange(int(t0[i]), int(t0[i]+dt)), k, b))
    return y

def reci_func(t0, k, b):
    return 1 - 1/(k*t0+b)

def reci_func_b(t0, k, b, dt):
    y = np.zeros(t0.shape[0])
    for i in range(t0.shape[0]):
        y[i] = np.prod(reci_func(np.arange(int(t0[i]), int(t0[i]+dt)), k, b))
    return y

def log_func(t0, k, b):
    return 1 - 1/np.log(k*t0+b)

def log_func_b(t0, k, b, dt):
    y = np.zeros(t0.shape[0])
    for i in range(t0.shape[0]):
        y[i] = np.prod(log_func(np.arange(int(t0[i]), int(t0[i]+dt)), k, b))
    return y

def poly_func(t0, a, b, c):
    return 1 - 1 / (a*t0**2 + b*t0 + c)

def poly_func_b(t0, a, b, c, dt):
    y = np.zeros(t0.shape[0])
    for i in range(t0.shape[0]):
        y[i] = np.prod(poly_func(np.arange(int(t0[i]), int(t0[i]+dt)), a, b, c))
    return y

"""
# Plot an example
idx = np.where((RData['Maze Type'] == 'Maze 1')&
               (RData['Type'] == 'Real')&
               (RData['Paradigm'] == 'CrossMaze')&
               (np.isnan(RData['Survival Frac.']) == False)&
               (RData['Session Interval'] + RData['Start Session'] <= 14)&
               (RData['Session Interval'] == 2))[0]
SubData1 = SubDict(RData, RData.keys(), idx)

idx = np.where((RData['Maze Type'] == 'Maze 1')&
               (RData['Type'] == 'Real')&
               (RData['Paradigm'] == 'CrossMaze')&
               (np.isnan(RData['Survival Frac.']) == False)&
               (RData['Session Interval'] + RData['Start Session'] <= 14)&
               (RData['Session Interval'] == 5))[0]
SubData4 = SubDict(RData, RData.keys(), idx)

idx = np.where((RData['Maze Type'] == 'Maze 1')&
               (RData['Type'] == 'Real')&
               (RData['Paradigm'] == 'CrossMaze')&
               (np.isnan(RData['Survival Frac.']) == False)&
               (RData['Session Interval'] + RData['Start Session'] <= 14)&
               (RData['Session Interval'] == 9))[0]
SubData8 = SubDict(RData, RData.keys(), idx)

x1 = np.arange(int(min(SubData1['Start Session'])), int(max(SubData1['Start Session']))+1)
x4 = np.arange(int(min(SubData4['Start Session'])), int(max(SubData4['Start Session']))+1)
x8 = np.arange(int(min(SubData8['Start Session'])), int(max(SubData8['Start Session']))+1)

params_cst1 = curve_fit(lambda t0, c: const_b(t0, c, dt=1), SubData1['Start Session'], SubData1['Survival Frac.'],
                           bounds=[[0], [1]], p0=[0.8])[0]
params_exp1 = curve_fit(lambda t0, k, b: exp_func_b(t0, k, b, dt=1), SubData1['Start Session'], SubData1['Survival Frac.'],
                           bounds=[[0, -np.inf], [np.inf, 0]], p0=[0.15, -2])[0]
params_rec1 = curve_fit(lambda t0, k, b: reci_func_b(t0, k, b, dt=1), SubData1['Start Session'], SubData1['Survival Frac.'],
                           bounds=[[0, -np.inf], [np.inf, np.inf]], p0=[0.4, 1.2])[0]
params_log1 = curve_fit(lambda t0, k, b: log_func_b(t0, k, b, dt=1), SubData1['Start Session'], SubData1['Survival Frac.'],
                           bounds=[[-np.inf, -np.inf], [np.inf, np.inf]])[0]
params_poly1 = curve_fit(lambda t0, a, b, c: poly_func_b(t0, a, b, c, dt=1), SubData1['Start Session'], SubData1['Survival Frac.'],
                            bounds=[[0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]])[0]

params_cst4 = curve_fit(lambda t0, c: const_b(t0, c, dt=4), SubData4['Start Session'], SubData4['Survival Frac.'],
                           bounds=[[0], [1]], p0=[0.8])[0]
params_exp4 = curve_fit(lambda t0, k, b: exp_func_b(t0, k, b, dt=4), SubData4['Start Session'], SubData4['Survival Frac.'],
                           bounds=[[0, -np.inf], [np.inf, 0]], p0=[0.15, -2])[0]
params_rec4 = curve_fit(lambda t0, k, b: reci_func_b(t0, k, b, dt=4), SubData4['Start Session'], SubData4['Survival Frac.'],
                           bounds=[[0, -np.inf], [np.inf, np.inf]], p0=[0.4, 1.2])[0]
params_log4 = curve_fit(lambda t0, k, b: log_func_b(t0, k, b, dt=4), SubData4['Start Session'], SubData4['Survival Frac.'],
                           bounds=[[-np.inf, -np.inf], [np.inf, np.inf]])[0]
params_poly4 = curve_fit(lambda t0, a, b, c: poly_func_b(t0, a, b, c, dt=4), SubData4['Start Session'], SubData4['Survival Frac.'],
                            bounds=[[0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]])[0]

params_cst8 = curve_fit(lambda t0, c: const_b(t0, c, dt=8), SubData8['Start Session'], SubData8['Survival Frac.'],
                           bounds=[[0], [1]], p0=[0.8])[0]
params_exp8 = curve_fit(lambda t0, k, b: exp_func_b(t0, k, b, dt=8), SubData8['Start Session'], SubData8['Survival Frac.'],
                           bounds=[[0, -np.inf], [np.inf, 0]], p0=[0.15, -2])[0]
params_rec8 = curve_fit(lambda t0, k, b: reci_func_b(t0, k, b, dt=8), SubData8['Start Session'], SubData8['Survival Frac.'],
                           bounds=[[0, -np.inf], [np.inf, np.inf]], p0=[0.4, 1.2])[0]
params_log8 = curve_fit(lambda t0, k, b: log_func_b(t0, k, b, dt=8), SubData8['Start Session'], SubData8['Survival Frac.'],
                           bounds=[[-np.inf, -np.inf], [np.inf, np.inf]])[0]
params_poly8 = curve_fit(lambda t0, a, b, c: poly_func_b(t0, a, b, c, dt=8), SubData8['Start Session'], SubData8['Survival Frac.'],
                            bounds=[[0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]])[0]

fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(2,3))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[2], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

ax1.plot(x1-1, const_b(x1, *params_cst1, dt=1), color='#003366', linewidth=0.15)
ax1.plot(x1-1, exp_func_b(x1, *params_exp1, dt=1), color='#0099CC', linewidth=0.15)
ax1.plot(x1-1, reci_func_b(x1, *params_rec1, dt=1), color='#66CCCC', linewidth=0.15)
ax1.plot(x1-1, log_func_b(x1, *params_log1, dt=1), color='#99CCFF', linewidth=0.15)
ax1.plot(x1-1, poly_func_b(x1, *params_poly1, dt=1), color='#FFC300', linewidth=0.15)
sns.stripplot(
    x = 'Start Session',
    y = 'Survival Frac.',
    data=SubData1,
    palette=sns.color_palette("rainbow", 12),
    edgecolor='black',
    size=3,
    linewidth=0.15,
    alpha=0.9,
    jitter=0.1,
    ax=ax1
)
ax1.set_xlim(-0.5, 11)
ax1.set_xticks(np.arange(0, 12))
ax1.set_ylim(0, 1)
ax1.set_yticks(np.linspace(0, 1, 6))

ax2.plot(x4-1, const_b(x4, *params_cst4, dt=4), color='#003366', linewidth=0.15)
ax2.plot(x4-1, exp_func_b(x4, *params_exp4, dt=4), color='#0099CC', linewidth=0.15)
ax2.plot(x4-1, reci_func_b(x4, *params_rec4, dt=4), color='#66CCCC', linewidth=0.15)
ax2.plot(x4-1, log_func_b(x4, *params_log4, dt=4), color='#99CCFF', linewidth=0.15)
ax2.plot(x4-1, poly_func_b(x4, *params_poly4, dt=4), color='#FFC300', linewidth=0.15)
sns.stripplot(
    x = 'Start Session',
    y = 'Survival Frac.',
    data=SubData4,
    palette=sns.color_palette("rainbow", 12),
    edgecolor='black',
    size=3,
    linewidth=0.15,
    alpha=0.9,
    jitter=0.1,
    ax=ax2
)
ax2.set_xlim(-0.5, 11)
ax2.set_xticks(np.arange(0, 12))
ax2.set_ylim(0, 1)
ax2.set_yticks(np.linspace(0, 1, 6))

ax3.plot(x8-1, const_b(x8, *params_cst8, dt=8), color='#003366', linewidth=0.15)
ax3.plot(x8-1, exp_func_b(x8, *params_exp8, dt=8), color='#0099CC', linewidth=0.15)
ax3.plot(x8-1, reci_func_b(x8, *params_rec8, dt=8), color='#66CCCC', linewidth=0.15)
ax3.plot(x8-1, log_func_b(x8, *params_log8, dt=8), color='#99CCFF', linewidth=0.15)
ax3.plot(x8-1, poly_func_b(x8, *params_poly8, dt=8), color='#FFC300', linewidth=0.15)
sns.stripplot(
    x = 'Start Session',
    y = 'Survival Frac.',
    data=SubData8,
    palette=sns.color_palette("rainbow", 12),
    edgecolor='black',
    size=3,
    linewidth=0.15,
    alpha=0.9,
    jitter=0.1,
    ax=ax3
)
ax3.set_xlim(-0.5, 11)
ax3.set_xticks(np.arange(0, 12))
ax3.set_ylim(0, 1)
ax3.set_yticks(np.linspace(0, 1, 6))

plt.savefig(join(loc, "[Maze 1] Cumulative Effect Example.png"), dpi=600)
plt.savefig(join(loc, "[Maze 1] Cumulative Effect Example.svg"), dpi=600)
plt.close()
"""


if os.path.exists(join(figdata, code_id+' [cumulative effect].pkl')):
    with open(join(figdata, code_id+' [cumulative effect].pkl'), 'rb') as handle:
        BData = pickle.load(handle)
else:
    # Firstly, fit the breakpoints of super stable fields.
    BData = {'R2': [], 'MSE': [], 'Method': [], 'hue': [], 'dt': []}

    RData['hue'] = np.array([RData['Maze Type'][i] + ' | '+RData['Paradigm'][i] for i in range(RData['Maze Type'].shape[0])])

    uniq_hue = np.unique(RData['hue'])
    maximums = np.array([14, 8, 8, 14, 14, 8, 8])
    
    for i in range(len(uniq_hue)):
        for dt in range(1, maximums[i] - 4):
            if uniq_hue[i] in ['Maze 1 | CrossMaze', 'Maze 2 | CrossMaze']:
                idx = np.where((RData['hue'] == uniq_hue[i])&
                               (np.isnan(RData['Survival Frac.']) == False)&
                               (RData['Type'] == 'Real')&
                               (RData['Session Interval'] + RData['Start Session'] <= maximums[i])&
                               (RData['Session Interval'] == dt+1))[0]
            else:
                idx = np.where((RData['hue'] == uniq_hue[i])&
                               (np.isnan(RData['Survival Frac.']) == False)&
                               (RData['Type'] == 'Real')&
                               (RData['Session Interval'] + RData['Start Session'] <= maximums[i])&
                               (RData['Session Interval'] == dt+1))[0]
                
            SubData = SubDict(RData, RData.keys(), idx)
        
            x, y_real = SubData['Start Session'], SubData['Survival Frac.']
        
            params_cst = curve_fit(lambda t0, c: const_b(t0, c, dt=dt), SubData['Start Session'], SubData['Survival Frac.'],
                                    bounds=[[0], [1]])[0]
            params_exp = curve_fit(lambda t0, k, b: exp_func_b(t0, k, b, dt=dt), SubData['Start Session'], SubData['Survival Frac.'],
                                    bounds=[[0, -np.inf], [np.inf, 0]])[0]
            params_rec = curve_fit(lambda t0, k, b: reci_func_b(t0, k, b, dt=dt), SubData['Start Session'], SubData['Survival Frac.'],
                                    bounds=[[0, -np.inf], [np.inf, np.inf]])[0]
            params_log = curve_fit(lambda t0, k, b: log_func_b(t0, k, b, dt=dt), SubData['Start Session'], SubData['Survival Frac.'],
                                    bounds=[[-np.inf, -np.inf], [np.inf, np.inf]])[0]
            params_poly = curve_fit(lambda t0, a, b, c: poly_func_b(t0, a, b, c, dt=dt), SubData['Start Session'], SubData['Survival Frac.'],
                                        bounds=[[0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]])[0]
        
            print(params_cst)
            print(params_exp)
            print(params_rec)
            print(params_log)
            print(params_poly, end='\n\n\n')
                    
            r2_cst = r2(y_real, const_b(x, *params_cst, dt=dt))
            r2_exp = r2(y_real, exp_func_b(x, *params_exp, dt=dt))
            r2_rec = r2(y_real, reci_func_b(x, *params_rec, dt=dt))
            r2_log = r2(y_real, log_func_b(x, *params_log, dt=dt))
            r2_poly = r2(y_real, poly_func_b(x, *params_poly, dt=dt))
        
                    
            mse_cst = MSE(y_real, const_b(x, *params_cst, dt=dt))
            mse_exp = MSE(y_real, exp_func_b(x, *params_exp, dt=dt))
            mse_rec = MSE(y_real, reci_func_b(x, *params_rec, dt=dt))
            mse_log = MSE(y_real, log_func_b(x, *params_log, dt=dt))
            mse_poly = MSE(y_real, poly_func_b(x, *params_poly, dt=dt))
        
            BData['hue'] = BData['hue'] + [uniq_hue[i]]*5
            BData['R2'] = BData['R2'] + [r2_cst, r2_exp, r2_rec, r2_log, r2_poly]
            BData['Method'] = BData['Method'] + ['const', 'exp', 'reci', 'log', 'poly']
            BData['MSE'] = BData['MSE'] + [mse_cst, mse_exp, mse_rec, mse_log, mse_poly]
            BData['dt'] = BData['dt'] + [dt]*5
        
    for k in BData.keys():
        BData[k] = np.array(BData[k])
        
    with open(join(figdata, code_id+' [cumulative effect].pkl'), 'wb') as handle:
        pickle.dump(BData, handle)
        
    D = pd.DataFrame(BData)
    D.to_excel(join(figdata, code_id+' [cumulative effect].xlsx'), index=False)
    
fig = plt.figure(figsize=(2,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Method',
    y = 'R2',
    data=BData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', '#FFC300'],
    width=0.8,
    capsize=0.2,
    errcolor='black',
    errwidth=0.5,
    ax = ax
)
sns.stripplot(
    x = 'Method',
    y = 'R2',
    data=BData,
    hue='hue',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A', '#C3AED6', '#66C7B4', '#A7D8DE'],
    hue_order=['Open Field | CrossMaze',
               'Maze 1 | CrossMaze',
               'Maze 2 | CrossMaze',
               'Open Field | HairpinMaze cis',
               'Open Field | HairpinMaze trs',
               'Maze 1 | ReverseMaze cis',
               'Maze 1 | ReverseMaze trs'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    jitter=0.2
)
ax.set_ylim([0, 1])
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, '[Cumulative Effect] fitted R2.png'), dpi = 600)
plt.savefig(join(loc, '[Cumulative Effect] fitted R2.svg'), dpi = 600)
plt.close()

print("R2 Statistic Test ------------------------")
print("1. Reci. vs Const")
idx1 = np.where(BData['Method'] == 'const')[0]
idx2 = np.where(BData['Method'] == 'reci')[0]
print_estimator(BData['R2'][idx1])
print_estimator(BData['R2'][idx2])
print(levene(BData['R2'][idx1], BData['R2'][idx2]))
print(ttest_rel(BData['R2'][idx1], BData['R2'][idx2]))
print("2. Reci. vs Exp.")
idx1 = np.where(BData['Method'] == 'exp')[0]
print_estimator(BData['R2'][idx1])
print(levene(BData['R2'][idx1], BData['R2'][idx2]))
print(ttest_rel(BData['R2'][idx1], BData['R2'][idx2]))
print("3. Reci. vs log")
idx1 = np.where(BData['Method'] == 'log')[0]
print_estimator(BData['R2'][idx1])
print(levene(BData['R2'][idx1], BData['R2'][idx2]))
print(ttest_rel(BData['R2'][idx1], BData['R2'][idx2]))
print("4. Reci. vs Poly")
idx1 = np.where(BData['Method'] == 'poly')[0]
print_estimator(BData['R2'][idx1])
print(levene(BData['R2'][idx1], BData['R2'][idx2]))
print(ttest_rel(BData['R2'][idx1], BData['R2'][idx2]), end='\n\n\n')


fig = plt.figure(figsize=(2,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Method',
    y = 'MSE',
    data=BData,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF', '#FFC300'],
    width=0.8,
    capsize=0.2,
    errcolor='black',
    errwidth=0.5,
    ax = ax
)
sns.stripplot(
    x = 'Method',
    y = 'MSE',
    data=BData,
    hue='hue',
    palette=['#F2E8D4', '#D4C9A8', '#8E9F85', '#527C5A', '#C3AED6', '#66C7B4', '#A7D8DE'],
    hue_order=['Open Field | CrossMaze',
               'Maze 1 | CrossMaze',
               'Maze 2 | CrossMaze',
               'Open Field | HairpinMaze cis',
               'Open Field | HairpinMaze trs',
               'Maze 1 | ReverseMaze cis',
               'Maze 1 | ReverseMaze trs'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.2
)
ax.set_ylim([0, 0.02])
ax.set_yticks(np.linspace(0, 0.02, 6))
plt.savefig(join(loc, '[Cumulative Effect] fitted MSE.png'), dpi = 600)
plt.savefig(join(loc, '[Cumulative Effect] fitted MSE.svg'), dpi = 600)
plt.close()

print("MSE Statistic Test ------------------------")
print("1. Reci. vs Const")
idx1 = np.where(BData['Method'] == 'const')[0]
idx2 = np.where(BData['Method'] == 'reci')[0]
print_estimator(BData['MSE'][idx1])
print_estimator(BData['MSE'][idx2])
print(levene(BData['MSE'][idx1], BData['MSE'][idx2]))
print(ttest_rel(BData['MSE'][idx1], BData['MSE'][idx2]))
print("2. Reci. vs Exp.")
idx1 = np.where(BData['Method'] == 'exp')[0]
print_estimator(BData['MSE'][idx1])
print(levene(BData['MSE'][idx1], BData['MSE'][idx2]))
print(ttest_rel(BData['MSE'][idx1], BData['MSE'][idx2]))
print("3. Reci. vs log")
idx1 = np.where(BData['Method'] == 'log')[0]
print_estimator(BData['MSE'][idx1])
print(levene(BData['MSE'][idx1], BData['MSE'][idx2]))
print(ttest_rel(BData['MSE'][idx1], BData['MSE'][idx2]))
print("4. Reci. vs Poly")
idx1 = np.where(BData['Method'] == 'poly')[0]
print_estimator(BData['MSE'][idx1])
print(levene(BData['MSE'][idx1], BData['MSE'][idx2]))
print(ttest_rel(BData['MSE'][idx1], BData['MSE'][idx2]), end='\n\n\n')
