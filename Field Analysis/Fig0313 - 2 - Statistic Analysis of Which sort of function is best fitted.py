from mylib.statistic_test import *
from mylib.statistic_test import GetMultidayIndexmap, ReadCellReg
from mylib.multiday.core import MultiDayCore

code_id = "0313 - Conditional Probability for Field Maintain"
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        RData = pickle.load(handle)
else:
    RData = DataFrameEstablish(variable_names = [
                             'Duration', 'Conditional Prob.', 'Conditional Recover Prob.',
                             'Global Recover Prob.', 'Cumulative Prob.', 
                             'Paradigm', 'On-Next Num', 'Off-Next Num'], 
                             f_member=['Type'],
                             f = f_CellReg_modi, function = ConditionalProb_Interface, 
                             file_name = code_id, behavior_paradigm = 'CrossMaze'
           )

total_global_recovered_frac_m1 = np.array([34.37605233, 30.07888649, 39.20555751, 37.87930946, 36.06190628])

idx1_1 = np.where((RData['Paradigm'] == 'CrossMaze') & 
               (RData['Maze Type'] == 'Maze 1') &
               (RData['Type'] == 'Real') & 
               (RData['Duration'] == 1))[0]
print("Maze 1, Recovery Prob.:")
print("  Silent duration = 1:  ", end='')
print_estimator(RData['Conditional Recover Prob.'][idx1_1])
idx1_8 = np.where((RData['Paradigm'] == 'CrossMaze') & 
               (RData['Maze Type'] == 'Maze 1') &
               (RData['Type'] == 'Real') & 
               (RData['Duration'] == 8))[0]
print(idx1_1.shape, idx1_8.shape)
print("  Silent duration = 8:  ", end='')
print_estimator(RData['Conditional Recover Prob.'][idx1_8])
print("  paired t test: ", end='')
print(ttest_rel(RData['Conditional Recover Prob.'][idx1_1], RData['Conditional Recover Prob.'][idx1_8]), end='\n\n')

idx2_1 = np.where((RData['Paradigm'] == 'CrossMaze') & 
               (RData['Maze Type'] == 'Maze 2') &
               (RData['Type'] == 'Real') & 
               (RData['Duration'] == 1))[0]
print("Maze 2, Recovery Prob.:")
print("  Silent duration = 1:  ", end='')
print_estimator(RData['Conditional Recover Prob.'][idx2_1])
idx2_8 = np.where((RData['Paradigm'] == 'CrossMaze') & 
               (RData['Maze Type'] == 'Maze 2') &
               (RData['Type'] == 'Real') & 
               (RData['Duration'] == 8))[0]
print("  Silent duration = 8:  ", end='')
print_estimator(RData['Conditional Recover Prob.'][idx2_8])
print("  paired t test: ", end='')
print(ttest_rel(RData['Conditional Recover Prob.'][idx2_1], RData['Conditional Recover Prob.'][idx2_8]), end='\n\n')



idx1_1 = np.where((RData['Paradigm'] == 'CrossMaze') & 
               (RData['Maze Type'] == 'Maze 1') &
               (RData['Type'] == 'Real') & 
               (RData['Duration'] == 1))[0]
print("Maze 1, Retention Prob.:")
print("  Retained duration = 1:  ", end='')
print_estimator(RData['Conditional Prob.'][idx1_1])
idx1_1 = np.where((RData['Paradigm'] == 'CrossMaze') & 
               (RData['Maze Type'] == 'Maze 1') &
               (RData['Type'] == 'Real') & 
               ((RData['MiceID'] != 10212) | (RData['Stage'] != 'Stage 1'))&
               (RData['Duration'] == 1))[0]
idx1_12 = np.where((RData['Paradigm'] == 'CrossMaze') & 
               (RData['Maze Type'] == 'Maze 1') &
               (RData['Type'] == 'Real') & 
               (RData['Duration'] == 12))[0]
print("  Retained duration = 12:  ", end='')
print_estimator(RData['Conditional Prob.'][idx1_12])
print("  paired t test: ", end='')
print(ttest_rel(RData['Conditional Prob.'][idx1_1], RData['Conditional Prob.'][idx1_12]), end='\n\n')

idx2_1 = np.where((RData['Paradigm'] == 'CrossMaze') & 
               (RData['Maze Type'] == 'Maze 2') &
               (RData['Type'] == 'Real') & 
               (RData['Duration'] == 1))[0]
print("Maze 2, Retention Prob.:")
print("  Retained duration = 1:  ", end='')
print_estimator(RData['Conditional Prob.'][idx2_1])
idx2_12 = np.where((RData['Paradigm'] == 'CrossMaze') & 
               (RData['Maze Type'] == 'Maze 2') &
               (RData['Type'] == 'Real') & 
               (RData['Duration'] == 12))[0]
print("  Retained duration = 12:  ", end='')
print_estimator(RData['Conditional Prob.'][idx2_12])
print("  paired t test: ", end='')
print(ttest_rel(RData['Conditional Prob.'][idx2_1], RData['Conditional Prob.'][idx2_12]), end='\n\n')


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

def exp_func(x, k, b, c):
    return c - np.exp(-k*(x-b))

def reci_func(x, k, b, c):
    return c - 1/(k*x+b)

def log_func(x, k, b, c):
    return c - 1/np.log(k*x+b)

def poly_func(x, a, b, c, d):
    return d - 1 / (a*x**2 + b*x + c)

from scipy.optimize import curve_fit

if os.path.exists(join(figdata, code_id+' [fit].pkl')):
    with open(join(figdata, code_id+' [fit].pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    BData = {'R2': [], 'MSE': [], 'Method': [], 'hue': [], 'C': [], 'MiceID': []}

    RData['hue'] = np.array([RData['Maze Type'][i] + ' | '+RData['Paradigm'][i] for i in range(RData['Maze Type'].shape[0])])

    uniq_hue = np.unique(RData['hue'])
    print(uniq_hue)

    for mouse in [10209, 10212, 10224, 10227]:    
        for i in range(len(uniq_hue)):
            if uniq_hue[i] in ['Maze 1 | CrossMaze', 'Maze 2 | CrossMaze']:
                idx = np.where((RData['hue'] == uniq_hue[i])&
                       (np.isnan(RData['Conditional Prob.']) == False)&
                       (RData['Type'] == 'Real')&
                       (RData['MiceID'] == mouse))[0]
            else:
                idx = np.where((RData['hue'] == uniq_hue[i])&(RData['Paradigm'] != 'CrossMaze')&
                       (np.isnan(RData['Conditional Prob.']) == False)&
                       (RData['Type'] == 'Real')&
                       (RData['MiceID'] == mouse))[0]
            
            if len(idx) == 0:
                print(mouse, uniq_hue[i], 'no data')
                continue
        
            SubData = SubDict(RData, RData.keys(), idx)                    
                    
            x, y_real = SubData['Duration'], SubData['Conditional Prob.']/100
                    
            params_cst, _ = curve_fit(const, x, y_real, bounds=[[0], [1]],
                                              p0=[0.5])
            params_exp, _ = curve_fit(exp_func, x, y_real, bounds=[[0, -np.inf, 0], [np.inf, 0, 1]],
                                              p0=[0.15, -2, 0.8])
            params_rec, _ = curve_fit(reci_func, x, y_real, bounds=[[0, -np.inf, 0], [np.inf, np.inf, 1]],
                                              p0=[0.4, 1.2, 0.8])
            params_log, _ = curve_fit(log_func, x, y_real, bounds=[[0, -np.inf, 0], [np.inf, np.inf, 1]],
                                              p0 = [4, 0.5, 0.8])
            
            print(params_cst)
            print(params_exp)
            print(params_rec)
            print(params_log, end='\n\n\n')
                    
            r2_cst = r2(y_real, const(x, *params_cst))
            r2_exp = r2(y_real, exp_func(x, *params_exp))
            r2_rec = r2(y_real, reci_func(x, *params_rec))
            r2_log = r2(y_real, log_func(x, *params_log))
                    
            mse_cst = MSE(y_real, const(x, *params_cst))
            mse_exp = MSE(y_real, exp_func(x, *params_exp))
            mse_rec = MSE(y_real, reci_func(x, *params_rec))
            mse_log = MSE(y_real, log_func(x, *params_log))
            
            BData['hue'] = BData['hue'] + [uniq_hue[i]+' | CrossMaze']*4
            BData['R2'] = BData['R2'] + [r2_cst, r2_exp, r2_rec, r2_log]
            BData['Method'] = BData['Method'] + ['const', 'exp', 'reci', 'log']
            BData['C'] = BData['C'] + [params_cst[-1], params_exp[-1], params_rec[-1], params_log[-1]]
            BData['MSE'] = BData['MSE'] + [mse_cst, mse_exp, mse_rec, mse_log]
            BData['MiceID'] = BData['MiceID'] + [mouse]*4
                
    for k in BData.keys():
        BData[k] = np.array(BData[k])
        
    with open(join(figdata, code_id+' [fit].pkl'), 'wb') as handle:
        pickle.dump(BData, handle)
        
    D = pd.DataFrame(BData)
    D.to_excel(join(figdata, code_id+' [fit].xlsx'), index=False)
    Data = BData

print("R2 statistics")
idx = np.where(Data['Method'] == 'reci')[0]
print_estimator(Data['R2'][idx])
print_estimator(Data['C'][idx])

fig = plt.figure(figsize=(1.6,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)  
sns.barplot(
    x = 'Method',
    y = 'C',
    data=Data,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    width=0.8,
    capsize=0.2,
    errcolor='black',
    errwidth=0.5,
    ax = ax
)
sns.stripplot(
    x = 'Method',
    y = 'C',
    data=Data,
    hue='hue',
    palette=['#D4C9A8', '#8E9F85', '#527C5A', '#C3AED6', '#66C7B4', '#A7D8DE'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    jitter=0.2
)
ax.set_ylim(0, 1.03)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, 'C.png'), dpi = 600)
plt.savefig(join(loc, 'C.svg'), dpi = 600)
plt.close() 
        
fig = plt.figure(figsize=(1.6,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Method',
    y = 'R2',
    data=Data,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    width=0.8,
    capsize=0.2,
    errcolor='black',
    errwidth=0.5,
    ax = ax
)
sns.stripplot(
    x = 'Method',
    y = 'R2',
    data=Data,
    hue='hue',
    palette=['#D4C9A8', '#8E9F85', '#527C5A', '#C3AED6', '#66C7B4', '#A7D8DE'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    jitter=0.2
)
ax.set_ylim([0, 1.03])
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, 'R2.png'), dpi = 600)
plt.savefig(join(loc, 'R2.svg'), dpi = 600)
plt.close()

print("R2 Statistic Test ------------------------")
print("1. Reci. vs Const")
idx1 = np.where(Data['Method'] == 'const')[0]
idx2 = np.where(Data['Method'] == 'reci')[0]
print_estimator(Data['R2'][idx1])
print_estimator(Data['R2'][idx2])
print(levene(Data['R2'][idx1], Data['R2'][idx2]))
print(ttest_rel(Data['R2'][idx1], Data['R2'][idx2]))
print("2. Reci. vs Exp.")
idx1 = np.where(Data['Method'] == 'exp')[0]
print_estimator(Data['R2'][idx1])
print(levene(Data['R2'][idx1], Data['R2'][idx2]))
print(ttest_rel(Data['R2'][idx1], Data['R2'][idx2]))
print("3. Reci. vs log")
idx1 = np.where(Data['Method'] == 'log')[0]
print_estimator(Data['R2'][idx1])
print(levene(Data['R2'][idx1], Data['R2'][idx2]))
print(ttest_rel(Data['R2'][idx1], Data['R2'][idx2]))


fig = plt.figure(figsize=(1.6,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Method',
    y = 'MSE',
    data=Data,
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    width=0.8,
    capsize=0.2,
    errcolor='black',
    errwidth=0.5,
    ax = ax
)
sns.stripplot(
    x = 'Method',
    y = 'MSE',
    data=Data,
    hue='hue',
    palette=['#D4C9A8', '#8E9F85', '#527C5A', '#C3AED6', '#66C7B4', '#A7D8DE'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    jitter=0.2
)
ax.set_ylim([0, 0.025])
ax.set_yticks(np.linspace(0, 0.025, 6))
plt.savefig(join(loc, 'MSE.png'), dpi = 600)
plt.savefig(join(loc, 'MSE.svg'), dpi = 600)
plt.close()

print("MSE Statistic Test ------------------------")
print("1. Reci. vs Const")
idx1 = np.where(Data['Method'] == 'const')[0]
idx2 = np.where(Data['Method'] == 'reci')[0]
print_estimator(Data['MSE'][idx1])
print_estimator(Data['MSE'][idx2])
print(levene(Data['MSE'][idx1], Data['MSE'][idx2]))
print(ttest_rel(Data['MSE'][idx1], Data['MSE'][idx2]))
print("2. Reci. vs Exp.")
idx1 = np.where(Data['Method'] == 'exp')[0]
print_estimator(Data['MSE'][idx1])
print(levene(Data['MSE'][idx1], Data['MSE'][idx2]))
print(ttest_rel(Data['MSE'][idx1], Data['MSE'][idx2]))
print("3. Reci. vs log")
idx1 = np.where(Data['Method'] == 'log')[0]
print_estimator(Data['MSE'][idx1])
print(levene(Data['MSE'][idx1], Data['MSE'][idx2]))
print(ttest_rel(Data['MSE'][idx1], Data['MSE'][idx2]), end='\n\n\n')