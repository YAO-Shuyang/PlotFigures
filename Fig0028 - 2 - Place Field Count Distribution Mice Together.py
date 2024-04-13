from mylib.statistic_test import *
import scipy.stats
from mylib.stats.gamma_poisson import gamma_poisson_pdf
from mylib.stats.ks import poisson_kstest, normal_discrete_kstest, nbinom_kstest, poisson_pmf, nbinom_pmf, norm_pdf
from mylib.stats.ks import poisson_cdf, norm_cdf, nbinom_cdf

code_id = '0028 - Place Field Number Distribution [All]'
loc = os.path.join(figpath, code_id)
mkdir(loc)
        
# Test data against negative binomial distribution
if os.path.exists(os.path.join(figdata,code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Field Number'], f = f1,
                              function = FieldNumber_0028_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata,code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata,code_id+' [Hairpin].pkl')) == False:
    HData = DataFrameEstablish(variable_names = ['Field Number', 'Direction'], f = f4,
                              function = FieldNumber_0028_Reverse_Interface, 
                              file_name = code_id+' [Hairpin]', behavior_paradigm = 'ReverseMaze')
else:
    with open(os.path.join(figdata,code_id+' [Hairpin].pkl'), 'rb') as handle:
        HData = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata,code_id+' [Reverse].pkl')) == False:
    RData = DataFrameEstablish(variable_names = ['Field Number', 'Direction'], f = f3,
                              function = FieldNumber_0028_Reverse_Interface, 
                              file_name = code_id+' [Reverse]', behavior_paradigm = 'ReverseMaze')
else:
    with open(os.path.join(figdata,code_id+' [Reverse].pkl'), 'rb') as handle:
        RData = pickle.load(handle)

print("KS statistics for all mice, Extended Fig. 9:")
poisson_ksd = np.array([
    0.062, 0.009, 0.15, 0.047, 0.083, 0.069,
    0.051, 0.007, 0.098, 0.045, 0.064, 0.086,
    0.131, 0.043, 0.036, 0.043,
    0.091, 0.053, 0.097, 0.062, 
    0.112, 0.082, 0.068, 0.082,
    0.201, 0.066, 0.084, 0.072
])

norm_ksd = np.array([
    0.125, 0.036, 0.054, 0.011, 0.025, 0.060, 
    0.094, 0.031, 0.026, 0.014, 0.050, 0.086,
    0.098, 0.012, 0.087, 0.077,
    0.034, 0.015, 0.100, 0.085,
    0.065, 0.020, 0.083, 0.083,
    0.101, 0.025, 0.094, 0.077
])

nbinom_ksd = np.array([
    0.034, 0.005, 0.051, 0.022, 0.033, 0.006, 
    0.020, 0.007, 0.064, 0.042, 0.010, 0.017,
    0.014, 0.036, 0.021, 0.010,
    0.033, 0.025, 0.015, 0.015,
    0.030, 0.035, 0.026, 0.007, 
    0.020, 0.014, 0.015, 0.007
])

print_estimator(poisson_ksd)
print_estimator(norm_ksd)
print_estimator(nbinom_ksd)
print(ttest_rel(nbinom_ksd, norm_ksd))
print(ttest_rel(nbinom_ksd, poisson_ksd))
 
for env in ['Maze 1', 'Maze 2']:
    print(env, " ------------------------------")
    for mouse in [11095, 11092, 10209, 10212, 10224, 10227]:
        print("  ", mouse)
        
        if env == 'Maze 1':
            idx = np.where((Data['MiceID'] == mouse) & 
                   (Data['Maze Type'] == env) & 
                   ((Data['Stage'] == 'Stage 2') | 
                    ((Data['Training Day'] != 'Day 1') & 
                    (Data['Training Day'] != 'Day 2') &
                    (Data['Training Day'] != 'Day 3'))))[0]
        elif env == 'Maze 2':
            idx = np.where((Data['MiceID'] == mouse) & 
                   (Data['Maze Type'] == env) & 
                    (Data['Training Day'] != 'Day 1') & 
                    (Data['Training Day'] != 'Day 2') &
                    (Data['Training Day'] != 'Day 3'))[0]
        print("    Neurons included: ", len(idx))
        
        if len(idx) == 0:
            continue
        fig = plt.figure(figsize=(4,3))
        ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        xmin, xmax = int(np.min(Data['Field Number'][idx])), int(np.max(Data['Field Number'][idx]))
        count = ax.hist(Data['Field Number'][idx], bins=xmax, range=(0.5, xmax+0.5), density=True, rwidth=0.8, color='lightgray')[0]
        
        lam = EqualPoissonFit(np.arange(1, xmax+1), count)
        y_pred = poisson_pmf(np.arange(1, xmax+1), lam, max_num=xmax)
        ax.plot(np.arange(1, xmax+1), y_pred, linewidth = 1, label = 'Poisson', alpha = 0.8)
        res = poisson_kstest(Data['Field Number'][idx], resample_size=500)
        print("        Poisson Params: ", lam)
        print("                KS: ", res)
        
        mean, sigma = norm.fit(Data['Field Number'][idx])
        y_pred = norm_pdf(np.arange(1, xmax+1), mean, sigma, max_num=xmax)
        ax.plot(np.arange(1, xmax+1), y_pred, linewidth = 1, label = 'Normal', alpha = 0.8)
        res = normal_discrete_kstest(Data['Field Number'][idx], resample_size=500)
        print("        Normal Params: ", mean, sigma)
        print("                KS: ", res)
        
        params = NegativeBinomialFit(np.arange(1, xmax+1), count)
        y_pred = nbinom_pmf(np.arange(1, xmax+1), params[0], params[1], max_num=xmax)
        ax.plot(np.arange(1, xmax+1), y_pred, linewidth = 1, label = 'Negative Binomial', alpha = 0.8)
        res = nbinom_kstest(Data['Field Number'][idx], monte_carlo_times=1000, resample_size=500)
        print("        Negative Binomial: ", params)
        print("                KS: ", res, end='\n\n')
        ax.legend()
    
        plt.savefig(os.path.join(loc, env+' - '+str(mouse)+'.png'), dpi = 600)
        plt.savefig(os.path.join(loc, env+' - '+str(mouse)+'.svg'), dpi = 600)
        plt.close() 
  
    
