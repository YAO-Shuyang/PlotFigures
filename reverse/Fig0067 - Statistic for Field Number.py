from mylib.statistic_test import *
from scipy.stats import lognorm
from mylib.stats.ks import poisson_kstest, normal_discrete_kstest, nbinom_kstest, poisson_pmf, nbinom_pmf, norm_pdf
from mylib.stats.ks import poisson_cdf, norm_cdf, nbinom_cdf

code_id = "0067 - Statistic for Field Number"
loc = os.path.join(figpath, "reverse", code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+' [All Fields - MA].pkl')):
    with open(join(figdata, code_id+' [All Fields - MA].pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Field Number', 'Direction'], 
                              f = f3, function = FieldNumberAll_Reverse_Interface, 
                              file_name = code_id+' [All Fields - MA]', behavior_paradigm = 'ReverseMaze')
    
if os.path.exists(join(figdata, code_id+' [All Fields - HP].pkl')):
    with open(join(figdata, code_id+' [All Fields - HP].pkl'), 'rb') as handle:
        HPData = pickle.load(handle)
else:
    HPData = DataFrameEstablish(variable_names = ['Field Number', 'Direction'], 
                              f = f4, function = FieldNumberAll_Reverse_Interface, 
                              file_name = code_id+' [All Fields - HP]', behavior_paradigm = 'HairpinMaze')
    
def plotfigures(
    Data: dict,
    mouse: int, 
    Prefix: str = '',
):
    print(mouse, " --------------------------------")
    
    idx = np.where((Data['MiceID'] == mouse)&(Data['Direction'] == 'cis')&(Data['Field Number'] > 0))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    print("    Forward: ", len(SubData['Field Number']))
    fig = plt.figure(figsize = (4,3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    bin_num = xmax = int(np.max(SubData['Field Number']))
    freq = ax.hist(SubData['Field Number'], bins=bin_num, range=(0.5, bin_num+0.5), color = 'lightgray', density=True, rwidth=0.8)[0]
    x = np.arange(1, bin_num+1)
    
    lam = EqualPoissonFit(np.arange(1, xmax+1), freq)
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
        
    params = NegativeBinomialFit(np.arange(1, xmax+1), freq)
    y_pred = nbinom_pmf(np.arange(1, xmax+1), params[0], params[1], max_num=xmax)
    ax.plot(np.arange(1, xmax+1), y_pred, linewidth = 1, label = 'Negative Binomial', alpha = 0.8)
    res = nbinom_kstest(Data['Field Number'][idx], monte_carlo_times=1000, resample_size=500)
    print("        Negative Binomial: ", params)
    print("                KS: ", res, end='\n\n')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(loc, Prefix, f'Mouse {mouse} - Cis - Field Number Distribution.png'), dpi = 600)
    plt.savefig(os.path.join(loc, Prefix, f'Mouse {mouse} - Cis - Field Number Distribution.svg'), dpi = 600)
    plt.close()
    
    
    idx = np.where((Data['MiceID'] == mouse)&(Data['Direction'] == 'trs')&(Data['Field Number'] > 0))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    print("    Backward: ", len(SubData['Field Number']))
    fig = plt.figure(figsize = (4,3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    bin_num = xmax = int(np.max(SubData['Field Number']))
    freq = ax.hist(SubData['Field Number'], bins=bin_num, range=(0.5, bin_num+0.5), color = 'lightgray', density=True, rwidth=0.8)[0]
    x = np.arange(1, bin_num+1)
    
    lam = EqualPoissonFit(np.arange(1, xmax+1), freq)
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
        
    params = NegativeBinomialFit(np.arange(1, xmax+1), freq)
    y_pred = nbinom_pmf(np.arange(1, xmax+1), params[0], params[1], max_num=xmax)
    ax.plot(np.arange(1, xmax+1), y_pred, linewidth = 1, label = 'Negative Binomial', alpha = 0.8)
    res = nbinom_kstest(Data['Field Number'][idx], monte_carlo_times=1000, resample_size=500)
    print("        Negative Binomial: ", params)
    print("                KS: ", res, end='\n\n')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(loc, Prefix, f'Mouse {mouse} - Tis - Field Number Distribution.png'), dpi = 600)
    plt.savefig(os.path.join(loc, Prefix, f'Mouse {mouse} - Tis - Field Number Distribution.svg'), dpi = 600)
    plt.close()
    print(end='\n\n\n')

mkdir(os.path.join(loc, 'Hairpin'))
mkdir(os.path.join(loc, 'Reversed'))
plotfigures(Data=Data, mouse=10209, Prefix = 'Reversed')
plotfigures(Data=Data, mouse=10212, Prefix = 'Reversed')
plotfigures(Data=Data, mouse=10224, Prefix = 'Reversed')
plotfigures(Data=Data, mouse=10227, Prefix = 'Reversed')

plotfigures(Data=HPData, mouse=10209, Prefix = 'Hairpin')
plotfigures(Data=HPData, mouse=10212, Prefix = 'Hairpin')
plotfigures(Data=HPData, mouse=10224, Prefix = 'Hairpin')
plotfigures(Data=HPData, mouse=10227, Prefix = 'Hairpin')         
