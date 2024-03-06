from mylib.statistic_test import *

code_id = '0321 - Fraction of Superstable Fields changes over time'
loc = join(figpath, code_id)
mkdir(loc)

f = pd.read_excel(join(figdata, "0300 - Kinetic Model Simulation.xlsx"))

if os.path.exists(join(figdata, code_id+' [real data].pkl')):
    with open(join(figdata, code_id+' [real data].pkl'), 'rb') as handle:
        RData = pickle.load(handle)
else:
    RData = DataFrameEstablish(variable_names = [
                        'Duration', 'Superstable Frac.', 'Threshold', 'Paradigm'], 
                        f = f_CellReg_modi, f_member=['Type'],
                        function = Superstable_Fraction_Data_Interface, 
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

def const(x, c):
    return np.repeat(c, len(x))

def const_b(x, c):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = np.prod(const(np.arange(1, int(x[i])), c))
    return y

def exp_func(x, k, b):
    return 1 - np.exp(-k*(x-b))

def exp_func_b(x, k, b):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = np.prod(exp_func(np.arange(1, int(x[i])), k, b))
    return y

def reci_func(x, k, b):
    return 1 - 1/(k*x+b)

def reci_func_b(x, k, b):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = np.prod(reci_func(np.arange(1, int(x[i])), k, b))
    return y

def log_func(x, k, b):
    return 1 - 1/np.log(k*x+b)

def log_func_b(x, k, b):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = np.prod(log_func(np.arange(1, int(x[i])), k, b))
    return y

def poly_func(x, a, b, c):
    return 1 - 1 / (a*x**2 + b*x + c)

def poly_func_b(x, a, b, c):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = np.prod(poly_func(np.arange(1, int(x[i])), a, b, c))
    return y


if os.path.exists(join(figdata, code_id+' [breakpoint].pkl')):
    with open(join(figdata, code_id+' [breakpoint].pkl'), 'rb') as handle:
        BData = pickle.load(handle)
else:
    # Firstly, fit the breakpoints of super stable fields.
    BData = {'R2': [], 'MSE': [], 'Method': [], 'hue': [], 'C': []}

    RData['hue'] = np.array([RData['Maze Type'][i] + ' | '+RData['Paradigm'][i] for i in range(RData['Maze Type'].shape[0])])

    uniq_hue = np.unique(RData['hue'])
    print(uniq_hue)
    
    for i in range(len(uniq_hue)):
        if uniq_hue[i] in ['Maze 1 | CrossMaze', 'Maze 2 | CrossMaze']:
            idx = np.where((RData['hue'] == uniq_hue[i])&
                       (np.isnan(RData['Superstable Frac.']) == False)&
                       (RData['Type'] == 'Real')&
                       (RData['Duration'] - RData['Threshold'] == 0)&
                       (RData['Threshold'] <= 13)&
                       (RData['Duration'] <= 13))[0]
        else:
            idx = np.where((RData['hue'] == uniq_hue[i])&
                       (np.isnan(RData['Superstable Frac.']) == False)&
                       (RData['Type'] == 'Real')&
                       (RData['Duration'] - RData['Threshold'] == 0))[0]
        SubData = SubDict(RData, RData.keys(), idx)
        
        x, y_real = SubData['Duration'], SubData['Superstable Frac.']
        
        params_cst, _ = curve_fit(const_b, x, y_real, bounds=[[0], [1]],
                                              p0=[0.5])
        params_exp, _ = curve_fit(exp_func_b, x, y_real, bounds=[[0, -np.inf], [np.inf, 0]],
                                              p0=[0.15, -2])
        params_rec, _ = curve_fit(reci_func_b, x, y_real, bounds=[[0, -np.inf], [np.inf, np.inf]],
                                              p0=[0.4, 1.2])
        try:
            params_log, _ = curve_fit(log_func_b, x, y_real, bounds=[[-np.inf, -np.inf], [np.inf, np.inf]],
                                              p0 = [4, 0.5])
        except:
            params_log = [np.nan, np.nan]
        
        try:
            params_poly, _ = curve_fit(poly_func_b, x, y_real, bounds=[[0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]], p0 = [0.08, -0.05, 1])
        except:
            params_poly = [np.nan, np.nan, np.nan]
        
        print(params_cst)
        print(params_exp)
        print(params_rec)
        print(params_log)
        print(params_poly, end='\n\n\n')
                    
        r2_cst = r2(y_real, const_b(x, *params_cst))
        r2_exp = r2(y_real, exp_func_b(x, *params_exp))
        r2_rec = r2(y_real, reci_func_b(x, *params_rec))
        r2_log = r2(y_real, log_func_b(x, *params_log))
        r2_poly = r2(y_real, poly_func_b(x, *params_poly))
        
                    
        mse_cst = MSE(y_real, const_b(x, *params_cst))
        mse_exp = MSE(y_real, exp_func_b(x, *params_exp))
        mse_rec = MSE(y_real, reci_func_b(x, *params_rec))
        mse_log = MSE(y_real, log_func_b(x, *params_log))
        mse_poly = MSE(y_real, poly_func_b(x, *params_poly))
        
        BData['hue'] = BData['hue'] + [uniq_hue[i]]*5
        BData['R2'] = BData['R2'] + [r2_cst, r2_exp, r2_rec, r2_log, r2_poly]
        BData['Method'] = BData['Method'] + ['const', 'exp', 'reci', 'log', 'poly']
        BData['C'] = BData['C'] + [params_cst[-1], params_exp[-1], params_rec[-1], params_log[-1], params_poly[-1]]
        BData['MSE'] = BData['MSE'] + [mse_cst, mse_exp, mse_rec, mse_log, mse_poly]
        
    for k in BData.keys():
        BData[k] = np.array(BData[k])
        
    with open(join(figdata, code_id+' [breakpoint].pkl'), 'wb') as handle:
        pickle.dump(BData, handle)
        
    D = pd.DataFrame(BData)
    D.to_excel(join(figdata, code_id+' [breakpoint].xlsx'), index=False)
    
    
fig = plt.figure(figsize=(2,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)  
sns.barplot(
    x = 'Method',
    y = 'C',
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
    y = 'C',
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
plt.savefig(join(loc, '[Breakpoints] fitted C.png'), dpi = 600)
plt.savefig(join(loc, '[Breakpoints] fitted C.svg'), dpi = 600)
plt.close() 
        
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
    dodge=True,
    jitter=0.2
)
ax.set_ylim([0, 1])
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, '[Breakpoints] fitted R2.png'), dpi = 600)
plt.savefig(join(loc, '[Breakpoints] fitted R2.svg'), dpi = 600)
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
ax.set_ylim([0, 0.01])
ax.set_yticks(np.linspace(0, 0.01, 6))
plt.savefig(join(loc, '[Breakpoints] fitted MSE.png'), dpi = 600)
plt.savefig(join(loc, '[Breakpoints] fitted MSE.svg'), dpi = 600)
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