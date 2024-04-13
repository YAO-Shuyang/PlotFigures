from mylib.statistic_test import *
from mylib.stats.ks import gamma_kstest

code_id = '0071 - Estimate True Distribution of Propensity'
loc = join(figpath, code_id)
mkdir(loc)
mkdir(join(loc, 'mean'))
mkdir(join(loc, 'sigma'))

def EstimatePropensity(f: pd.DataFrame, i:int):
    if os.path.exists(f['cellreg_folder'][i]):
        index_map = ReadCellReg(join(f['cellreg_folder'][i], "cellRegistered.mat"))
        index_map = index_map.astype(np.int32)
        
        idx = np.where((index_map[0, :] > 0) & (index_map[1, :] > 0))[0]
        index_map = index_map[:, idx]
        
        hairpin_idx = np.where((f4['MiceID'] == f['MiceID'][i]) & (f4['date'] == f['date'][i]))[0][0]
        reverse_idx = np.where((f3['MiceID'] == f['MiceID'][i]) & (f3['date'] == f['date'][i]))[0][0]
        
        with open(f4['Trace File'][hairpin_idx], 'rb') as handle:
            trace_h = pickle.load(handle)
            
        with open(f3['Trace File'][reverse_idx], 'rb') as handle:
            trace_r = pickle.load(handle)
    
        field_num = np.zeros((index_map.shape[1], 4), np.float64)
        for j in range(index_map.shape[1]):
            field_num[j, 0] = trace_h['cis']['place_field_num_multiday'][index_map[0, j]-1]/10.3
            field_num[j, 1] = trace_h['trs']['place_field_num_multiday'][index_map[0, j]-1]/10.3
            field_num[j, 2] = trace_r['cis']['place_field_num_multiday'][index_map[1, j]-1]/6.3
            field_num[j, 3] = trace_r['trs']['place_field_num_multiday'][index_map[1, j]-1]/6.3
        
        field_num[np.where(field_num == 0)] = np.nan
        is_num = np.where(np.isnan(field_num), 0, 1)
        idx = np.where(np.sum(is_num, axis=1) >= 4)[0]
        field_num = field_num[idx, :]
        
        mean = np.nanmean(field_num, axis=1)
        sigma = np.nanstd(field_num, axis=1)
        
        """
        fig = plt.figure(figsize=(4,3))
        ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        ax.hist(mean, range=(0.5, 15.5), bins=15, rwidth=0.8, color='gray')
        a, locc, scale = gamma.fit(mean)
        y = gamma.pdf(np.linspace(1, 16, 1501), a, loc=locc, scale=scale)*field_num.shape[0]
        ax.plot(np.linspace(1, 16, 1501), y, color='red')
        
        ax.set_title(str(field_num.shape[0])+" cell pairs")
        plt.savefig(join(loc, 'mean', str(f['MiceID'][i])+"_"+str(f['date'][i])+".png"), dpi=600)
        plt.savefig(join(loc, 'mean', str(f['MiceID'][i])+"_"+str(f['date'][i])+".svg"), dpi=600)
        plt.close()
        
        fig = plt.figure(figsize=(4,3))
        ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        ax.hist(sigma, range=(0, 10), bins=20, rwidth=0.8, color='gray')
        ax.set_title(str(field_num.shape[0])+" cell pairs")
        plt.savefig(join(loc, 'sigma', str(f['MiceID'][i])+"_"+str(f['date'][i])+".png"), dpi=600)
        plt.savefig(join(loc, 'sigma', str(f['MiceID'][i])+"_"+str(f['date'][i])+".svg"), dpi=600)
        plt.close()
        """
        return field_num
        
    else:
        print(f"{f['cellreg_folder'][i]} doesn't exist")

# 10227 loc = 0
# 10209, 10224 loc = 1
# 10212
for m in [10209, 10212, 10224, 10227]: #np.unique(f_CellReg_reverse['MiceID'])
    
    idx = np.where(f_CellReg_reverse['MiceID'] == m)[0]
    field_nums = np.zeros((1,4), np.float64)
    for i in idx:
        num = EstimatePropensity(f_CellReg_reverse, i)
        field_nums = np.concatenate([field_nums, num], axis=0)
    
    field_nums = field_nums[1:, :]
    
    mean = np.nanmean(field_nums, axis=1)
    sigma = np.nanstd(field_nums, axis=1)
    
    print(m, field_nums.shape)
    fig = plt.figure(figsize=(4,3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    d = ax.hist(mean, range=(0, 2.5), bins=25, color='lightgray', density=True)[0]

    a, locc, scale = gamma.fit(mean, 40)

    y = gamma.pdf(np.linspace(0, 2.5, 2501), a, loc=locc, scale=scale)
    print(kstest(mean, 'gamma', args=(a, locc, scale)))
    #print(gamma_kstest(mean, monte_carlo_times=1000))
    print("    Params", a, locc, 1/scale)
    print("    CV: ", 1 / np.sqrt(a), end='\n\n')
    ax.plot(np.linspace(0, 2.5, 2501), y, color='red', linewidth=1)
        
    ax.set_title(str(field_nums.shape[0])+" cell pairs")
    ax.set_xlim(0, 2)
    ax.set_xticks(np.linspace(0, 2.5, 6))
    plt.savefig(join(loc, str(f_CellReg_reverse['MiceID'][i])+" [mean].png"), dpi=600)
    plt.savefig(join(loc, str(f_CellReg_reverse['MiceID'][i])+" [mean].svg"), dpi=600)
    plt.close()
    
    """
    fig = plt.figure(figsize=(4,3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.hist(mean, range=(0.5, 15.5), bins=15, rwidth=0.8, color='gray')
    a, locc, scale = gamma.fit(mean, loc=0, scale=0.4)
    y = gamma.pdf(np.linspace(1, 16, 1501), a, loc=locc, scale=scale)*field_nums.shape[0]
    print(gamma_kstest(mean, monte_carlo_times=10000))
    print(a, locc, scale)
    ax.plot(np.linspace(1, 16, 1501), y, color='red', linewidth=0.5)
        
    ax.set_title(str(field_nums.shape[0])+" cell pairs")
    ax.set_xlim(0.5, 15.5)
    ax.set_xticks(np.linspace(1, 15, 15))
    plt.savefig(join(loc, str(f_CellReg_reverse['MiceID'][i])+" [mean].png"), dpi=600)
    plt.savefig(join(loc, str(f_CellReg_reverse['MiceID'][i])+" [mean].svg"), dpi=600)
    plt.close()
        
    fig = plt.figure(figsize=(4,3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.hist(sigma, range=(0, 10), bins=20, rwidth=0.8, color='gray')
    ax.set_title(str(field_nums.shape[0])+" cell pairs")
    plt.savefig(join(loc, str(f_CellReg_reverse['MiceID'][i])+" [sigma].png"), dpi=600)
    plt.savefig(join(loc, str(f_CellReg_reverse['MiceID'][i])+" [sigma].svg"), dpi=600)
    plt.close()
    
    fig = plt.figure(figsize=(3,3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.plot(mean, sigma**2, 'o', markeredgewidth=0, markersize=2, color='k')
    ax.set_title(str(field_nums.shape[0])+" cell pairs")
    plt.savefig(join(loc, str(f_CellReg_reverse['MiceID'][i])+" [mean-sigma].png"), dpi=600)
    plt.savefig(join(loc, str(f_CellReg_reverse['MiceID'][i])+" [mean-sigma].svg"), dpi=600)
    plt.close()
    """
