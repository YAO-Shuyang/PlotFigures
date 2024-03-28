from mylib.statistic_test import *
import scipy.stats
from mylib.stats.gamma_poisson import gamma_poisson_pdf
from mylib.stats.kstest import poisson_kstest, normal_discrete_kstest, nbinom_kstest, poisson_pmf, nbinom_pmf, norm_pdf

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

for env in ['Open Field', 'Maze 1', 'Maze 2']:
    print(env, " ------------------------------")
    for mouse in [11095, 11092, 10209, 10212, 10224, 10227]:
        print("    ", mouse)
        idx = np.where((Data['MiceID'] == mouse) & 
                   (Data['Maze Type'] == env) & 
                   (Data['Stage'] != 'PRE') &
                   ((Data['Stage'] == 'Stage 2') | 
                    ((Data['Training Day'] != 'Day 1') & 
                     (Data['Training Day'] != 'Day 2') &
                     (Data['Training Day'] != 'Day 3'))))[0]
        print("    Neurons included: ", len(idx))
        fig = plt.figure(figsize=(4,3))
        ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        xmin, xmax = int(np.min(Data['Field Number'][idx])), int(np.max(Data['Field Number'][idx]))
        count = ax.hist(Data['Field Number'][idx], bins=xmax, range=(0.5, xmax+0.5), density=True, rwidth=0.8, color='gray')[0]
    
        lam = EqualPoissonFit(np.arange(1, xmax+1), count)
        y_pred = poisson_pmf(np.arange(1, xmax+1), lam)
        ax.plot(np.arange(1, xmax+1), y_pred, linewidth = 0.5, label = 'Poisson')
        res = poisson_kstest(Data['Field Number'][idx], resample_size=20000)
        print("        Poisson Params: ", lam)
        print("                KS: ", res)

        mean, sigma = norm.fit(Data['Field Number'][idx])
        y_pred = norm_pdf(np.arange(1, xmax+1), mean, sigma)
        ax.plot(np.arange(1, xmax+1), y_pred, linewidth = 0.5, label = 'Normal')
        res = normal_discrete_kstest(Data['Field Number'][idx], resample_size=20000)
        print("        Normal Params: ", mean, sigma)
        print("                KS: ", res)

        params = NegativeBinomialFit(np.arange(1, xmax+1), count)
        y_pred = nbinom_pmf(np.arange(1, xmax+1), params[0], params[1])
        ax.plot(np.arange(1, xmax+1), y_pred, linewidth = 0.5, label = 'Negative Binomial')
        res = nbinom_kstest(Data['Field Number'][idx], monte_carlo_times=1000, resample_size=20000)
        print("        Negative Binomial: ", params)
        print("                KS: ", res, end='\n\n')
        ax.legend()
    
        plt.savefig(os.path.join(loc, env+' - '+str(mouse)+'.png'), dpi = 600)
        plt.savefig(os.path.join(loc, env+' - '+str(mouse)+'.svg'), dpi = 600)
        plt.close() 
  
    
    
    