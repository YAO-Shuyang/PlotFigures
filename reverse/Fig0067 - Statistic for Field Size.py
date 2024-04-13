from mylib.statistic_test import *
from mylib.stats.ks import nbinom_kstest, lognorm_kstest, gamma_size_kstest
from scipy.stats import lognorm

code_id = "0067 - Statistic for Field Size"
loc = os.path.join(figpath, "reverse", code_id)
mkdir(loc)
"""
if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Log-normal P-value', 'Log-normal Statistics', 'Direction'], 
                              f = f3, function = FieldSizeTestLogNormal_Reverse_Interface, 
                              file_name = code_id, behavior_paradigm = 'ReverseMaze')
    
if os.path.exists(join(figdata, code_id+' [Hairpin].pkl')):
    with open(join(figdata, code_id+' [Hairpin].pkl'), 'rb') as handle:
        HPData = pickle.load(handle)
else:
    HPData = DataFrameEstablish(variable_names = ['Log-normal P-value', 'Log-normal Statistics', 'Direction'], 
                              f = f4, function = FieldSizeTestLogNormal_Reverse_Interface, 
                              file_name = code_id + ' [Hairpin]', behavior_paradigm = 'HairpinMaze')
    
print(np.where(Data['Log-normal P-value'][np.where(Data['Direction'] == 'cis')[0]] >= 0.05)[0].shape[0])
print(np.where(Data['Log-normal P-value'][np.where(Data['Direction'] == 'trs')[0]] >= 0.05)[0].shape[0], end='\n\n')   
print(np.where(HPData['Log-normal P-value'][np.where(HPData['Direction'] == 'cis')[0]] >= 0.05)[0].shape[0])
print(np.where(HPData['Log-normal P-value'][np.where(HPData['Direction'] == 'trs')[0]] >= 0.05)[0].shape[0], end='\n\n')   
"""
if os.path.exists(join(figdata, code_id+' [All Fields - MA].pkl')):
    with open(join(figdata, code_id+' [All Fields - MA].pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Field Size', 'Direction'], 
                              f = f3, function = FieldSizeAll_Reverse_Interface, 
                              file_name = code_id+' [All Fields - MA]', behavior_paradigm = 'ReverseMaze')
    
if os.path.exists(join(figdata, code_id+' [All Fields - HP].pkl')):
    with open(join(figdata, code_id+' [All Fields - HP].pkl'), 'rb') as handle:
        HPData = pickle.load(handle)
else:
    HPData = DataFrameEstablish(variable_names = ['Field Size', 'Direction'], 
                              f = f4, function = FieldSizeAll_Reverse_Interface, 
                              file_name = code_id+' [All Fields - HP]', behavior_paradigm = 'HairpinMaze')
    
def plotfigures(
    Data: dict,
    mouse: int, 
    bin_num: int = 400,
    Prefix: str = '',
):
    idx = np.where((Data['MiceID'] == mouse)&(Data['Direction'] == 'cis'))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)

    fig = plt.figure(figsize = (4,3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    freq = ax.hist(SubData['Field Size'], bins=bin_num, range=(0.5, bin_num+0.5), color = 'lightgray', density=True)[0]
    x = np.arange(1, bin_num+1)
    shape, locc, scale = lognorm.fit(SubData['Field Size'], floc = 0)
    y = lognorm.pdf(x, shape, loc=locc, scale=scale)
    #print(shape, locc, scale, np.sum(y))
    ax.plot(x, y, linewidth = 1, label='Lognormal', alpha = 0.8)
    print(f"Cis, Mouse {mouse}, Shape Num {len(SubData['Field Size'])}:")
    statistic_p, lognorm_p = lognorm_kstest(SubData['Field Size'], resample_size=1621, monte_carlo_times=10000)
    print()
    print(f"    Lognormal: shape {shape}, loc {locc}, scale {scale}")
    print(f"    Lognormal Statistic, {statistic_p}, Lognormal P-value: {lognorm_p}", end="\n\n")
    ax.plot(x, y, linewidth = 1, label='Lognormal', alpha = 0.8)

    alpha, c, beta = gamma.fit(SubData['Field Size'], floc = 0)
    y = gamma.pdf(x, alpha, scale=beta, loc=c)
    ax.plot(x, y, linewidth = 1, label='Gamma', alpha = 0.8)
    statistic_p, gamma_p = gamma_size_kstest(SubData['Field Size'], resample_size=1621, is_floc=True, monte_carlo_times=10000)
    print()
    print(f"    Gamma: alpha {alpha}, beta {beta}")
    print(f"    Gamma Statistic, {statistic_p}, Gamma P-value: {gamma_p}", end="\n\n")
    
    ax.legend()
    ax.set_xlim([-0.5, bin_num])
    ax.set_xlim([0, bin_num])
    ax.set_xticks(np.linspace(0, bin_num, 5))
    ax.set_xlabel("Field Size / bin")
    ax.set_ylabel("Field Count")
    plt.tight_layout()
    plt.savefig(os.path.join(loc, Prefix, f'Mouse {mouse} - Cis - Field Size Distribution.png'), dpi = 600)
    plt.savefig(os.path.join(loc, Prefix, f'Mouse {mouse} - Cis - Field Size Distribution.svg'), dpi = 600)
    plt.close()
    
    idx = np.where((Data['MiceID'] == mouse)&(Data['Direction'] == 'trs'))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)

    fig = plt.figure(figsize = (4,3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    freq = ax.hist(SubData['Field Size'], bins=bin_num, range=(0.5, bin_num+0.5), color = 'lightgray', density=True)[0]
    x = np.arange(1, bin_num+1)
    shape, locc, scale = lognorm.fit(SubData['Field Size'], floc = 0)
    y = lognorm.pdf(x, shape, loc=locc, scale=scale)
    #print(shape, locc, scale, np.sum(y))
    ax.plot(x, y, linewidth = 1, label='Lognormal', alpha = 0.8)
    print(f"Trs, Mouse {mouse}, Shape Num {len(SubData['Field Size'])}:")
    statistic_p, lognorm_p = lognorm_kstest(SubData['Field Size'], resample_size=1621, monte_carlo_times=10000)
    print()
    print(f"    Lognormal: shape {shape}, loc {locc}, scale {scale}")
    print(f"    Lognormal Statistic, {statistic_p}, Lognormal P-value: {lognorm_p}", end="\n\n")
    ax.plot(x, y, linewidth = 1, label='Lognormal', alpha = 0.8)

    alpha, c, beta = gamma.fit(SubData['Field Size'], floc = 0)
    y = gamma.pdf(x, alpha, scale=beta, loc=c)
    ax.plot(x, y, linewidth = 1, label='Gamma', alpha = 0.8)
    statistic_p, gamma_p = gamma_size_kstest(SubData['Field Size'], resample_size=1621, is_floc=True, monte_carlo_times=10000)
    print()
    print(f"    Gamma: alpha {alpha}, beta {beta}")
    print(f"    Gamma Statistic, {statistic_p}, Gamma P-value: {gamma_p}", end="\n\n")
    
    ax.legend()
    ax.set_xlim([-0.5, bin_num])
    ax.set_xlim([0, bin_num])
    ax.set_xticks(np.linspace(0, bin_num, 5))
    ax.set_xlabel("Field Size / bin")
    ax.set_ylabel("Field Count")
    plt.tight_layout()
    plt.savefig(os.path.join(loc, Prefix, f'Mouse {mouse} - Tis - Field Size Distribution.png'), dpi = 600)
    plt.savefig(os.path.join(loc, Prefix, f'Mouse {mouse} - Tis - Field Size Distribution.svg'), dpi = 600)
    plt.close()

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
