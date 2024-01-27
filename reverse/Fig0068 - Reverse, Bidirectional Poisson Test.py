from mylib.statistic_test import *

code_id = '0068 - Reverse, Bidirectional Poisson Test'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        RData = pickle.load(handle)
else:
    RData = DataFrameEstablish(variable_names = ['lam', 'KS Statistics', 'KS P-Value',
                                                                            'r', 'p', 'KS Gamma', 'KS Gamma P-value',
                                                                            'Direction'], 
                              f = f3, function = PoissonTest_Reverse_Interface, 
                              file_name = code_id, behavior_paradigm = 'ReverseMaze')
    
if os.path.exists(join(figdata, '0068 - Hairpin, Bidirectional Poisson Test'+'.pkl')):
    with open(join(figdata, '0068 - Hairpin, Bidirectional Poisson Test'+'.pkl'), 'rb') as handle:
        HData = pickle.load(handle)
else:
    HData = DataFrameEstablish(variable_names = ['lam', 'KS Statistics', 'KS P-Value',
                                                                            'r', 'p', 'KS Gamma', 'KS Gamma P-value',
                                                                            'Direction'], 
                              f = f4, function = PoissonTest_Reverse_Interface, 
                              file_name = '0068 - Hairpin, Bidirectional Poisson Test', behavior_paradigm = 'HairpinMaze')

print("Reverse")
print(len(np.where((RData['Direction'] == 'cis')&(RData['KS P-Value'] >= 0.05))[0]))
print(len(np.where((RData['Direction'] == 'cis')&(RData['KS P-Value'] >= 0.01))[0]))
print(len(np.where((RData['Direction'] == 'trs')&(RData['KS P-Value'] >= 0.05))[0]))
print(len(np.where((RData['Direction'] == 'trs')&(RData['KS P-Value'] >= 0.01))[0]), end='\n\n')

print("Hairpin")
print(len(np.where((HData['Direction'] == 'cis')&(HData['KS P-Value'] >= 0.05))[0]))
print(len(np.where((HData['Direction'] == 'cis')&(HData['KS P-Value'] >= 0.01))[0]))
print(len(np.where((HData['Direction'] == 'trs')&(HData['KS P-Value'] >= 0.05))[0]))
print(len(np.where((HData['Direction'] == 'trs')&(HData['KS P-Value'] >= 0.01))[0]))

fig = plt.figure(figsize=(4,4))
colors = [
    sns.color_palette('Blues', 9)[3],
    sns.color_palette('YlOrRd', 9)[3],
    sns.color_palette('crest', 9)[3],
    sns.color_palette('YlOrRd', 9)[7],
]
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axvline(0.05, color='k', linestyle='--', linewidth=0.5)
ax.axhline(0.05, color='k', linestyle='--', linewidth=0.5)
ax.axvline(0.01, color='k', linestyle='--', linewidth=0.5)
ax.axhline(0.01, color='k', linestyle='--', linewidth=0.5)
ax.plot(RData['KS P-Value'][np.where((RData['Direction'] == 'cis')&(RData['MiceID'] == 10209))[0]], 
        RData['KS P-Value'][np.where((RData['Direction'] == 'trs')&(RData['MiceID'] == 10209))[0]], 
        'o', color = colors[0], markeredgewidth=0, markersize=4, alpha=0.8, label = '10209'
)
ax.plot(RData['KS P-Value'][np.where((RData['Direction'] == 'cis')&(RData['MiceID'] == 10212))[0]], 
        RData['KS P-Value'][np.where((RData['Direction'] == 'trs')&(RData['MiceID'] == 10212))[0]], 
        'o', color = colors[1], markeredgewidth=0, markersize=4, alpha=0.8, label = '10212'
)
ax.plot(RData['KS P-Value'][np.where((RData['Direction'] == 'cis')&(RData['MiceID'] == 10224))[0]], 
        RData['KS P-Value'][np.where((RData['Direction'] == 'trs')&(RData['MiceID'] == 10224))[0]], 
        'o', color = colors[2], markeredgewidth=0, markersize=4, alpha=0.8, label = '10224'
)
ax.plot(RData['KS P-Value'][np.where((RData['Direction'] == 'cis')&(RData['MiceID'] == 10227))[0]], 
        RData['KS P-Value'][np.where((RData['Direction'] == 'trs')&(RData['MiceID'] == 10227))[0]], 
        'o', color = colors[3], markeredgewidth=0, markersize=4, alpha=0.8, label = '10227'
)
ax.set_aspect("equal")
ax.axis([0.0001, 1, 0.0001, 1])
ax.semilogx()
ax.semilogy()
plt.legend()
plt.tight_layout()
plt.savefig(join(loc, 'Reverse KS Poisson Pvalues.png'), dpi=600)
plt.savefig(join(loc, 'Reverse KS Poisson Pvalues.svg'), dpi=600)
plt.close()

fig = plt.figure(figsize=(4,4))
colors = [
    sns.color_palette('Blues', 9)[3],
    sns.color_palette('YlOrRd', 9)[3],
    sns.color_palette('crest', 9)[3],
    sns.color_palette('YlOrRd', 9)[7],
]
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axvline(0.05, color='k', linestyle='--', linewidth=0.5)
ax.axhline(0.05, color='k', linestyle='--', linewidth=0.5)
ax.axvline(0.01, color='k', linestyle='--', linewidth=0.5)
ax.axhline(0.01, color='k', linestyle='--', linewidth=0.5)
ax.plot(RData['KS Gamma P-value'][np.where((RData['Direction'] == 'cis')&(RData['MiceID'] == 10209))[0]], 
        RData['KS Gamma P-value'][np.where((RData['Direction'] == 'trs')&(RData['MiceID'] == 10209))[0]], 
        'o', color = colors[0], markeredgewidth=0, markersize=4, alpha=0.8, label = '10209'
)
ax.plot(RData['KS Gamma P-value'][np.where((RData['Direction'] == 'cis')&(RData['MiceID'] == 10212))[0]], 
        RData['KS Gamma P-value'][np.where((RData['Direction'] == 'trs')&(RData['MiceID'] == 10212))[0]], 
        'o', color = colors[1], markeredgewidth=0, markersize=4, alpha=0.8, label = '10212'
)
ax.plot(RData['KS Gamma P-value'][np.where((RData['Direction'] == 'cis')&(RData['MiceID'] == 10224))[0]], 
        RData['KS Gamma P-value'][np.where((RData['Direction'] == 'trs')&(RData['MiceID'] == 10224))[0]], 
        'o', color = colors[2], markeredgewidth=0, markersize=4, alpha=0.8, label = '10224'
)
ax.plot(RData['KS Gamma P-value'][np.where((RData['Direction'] == 'cis')&(RData['MiceID'] == 10227))[0]], 
        RData['KS Gamma P-value'][np.where((RData['Direction'] == 'trs')&(RData['MiceID'] == 10227))[0]], 
        'o', color = colors[3], markeredgewidth=0, markersize=4, alpha=0.8, label = '10227'
)
ax.set_aspect("equal")
ax.axis([0.0001, 1, 0.0001, 1])
ax.semilogx()
ax.semilogy()
plt.legend()
plt.tight_layout()
plt.savefig(join(loc, 'Reverse KS Gamma-Poisson Pvalues.png'), dpi=600)
plt.savefig(join(loc, 'Reverse KS Gamma-Poisson Pvalues.svg'), dpi=600)
plt.close()


fig = plt.figure(figsize=(4,4))
colors = [
    sns.color_palette('Blues', 9)[3],
    sns.color_palette('YlOrRd', 9)[3],
    sns.color_palette('crest', 9)[3],
    sns.color_palette('YlOrRd', 9)[7],
]
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axvline(0.05, color='k', linestyle='--', linewidth=0.5)
ax.axhline(0.05, color='k', linestyle='--', linewidth=0.5)
ax.axvline(0.01, color='k', linestyle='--', linewidth=0.5)
ax.axhline(0.01, color='k', linestyle='--', linewidth=0.5)
ax.plot(HData['KS P-Value'][np.where((HData['Direction'] == 'cis')&(HData['MiceID'] == 10209))[0]], 
        HData['KS P-Value'][np.where((HData['Direction'] == 'trs')&(HData['MiceID'] == 10209))[0]], 
        'o', color = colors[0], markeredgewidth=0, markersize=4, alpha=0.8, label = '10209'
)
ax.plot(HData['KS P-Value'][np.where((HData['Direction'] == 'cis')&(HData['MiceID'] == 10212))[0]], 
        HData['KS P-Value'][np.where((HData['Direction'] == 'trs')&(HData['MiceID'] == 10212))[0]], 
        'o', color = colors[1], markeredgewidth=0, markersize=4, alpha=0.8, label = '10212'
)
ax.plot(HData['KS P-Value'][np.where((HData['Direction'] == 'cis')&(HData['MiceID'] == 10224))[0]], 
        HData['KS P-Value'][np.where((HData['Direction'] == 'trs')&(HData['MiceID'] == 10224))[0]], 
        'o', color = colors[2], markeredgewidth=0, markersize=4, alpha=0.8, label = '10224'
)
ax.plot(HData['KS P-Value'][np.where((HData['Direction'] == 'cis')&(HData['MiceID'] == 10227))[0]], 
        HData['KS P-Value'][np.where((HData['Direction'] == 'trs')&(HData['MiceID'] == 10227))[0]], 
        'o', color = colors[3], markeredgewidth=0, markersize=4, alpha=0.8, label = '10227'
)
ax.set_aspect("equal")
ax.axis([0.0001, 1, 0.0001, 1])
ax.semilogx()
ax.semilogy()
plt.legend()
plt.tight_layout()
plt.savefig(join(loc, 'Hairpin KS Poisson Pvalues.png'), dpi=600)
plt.savefig(join(loc, 'Hairpin KS Poisson Pvalues.svg'), dpi=600)
plt.close()

fig = plt.figure(figsize=(4,4))
colors = [
    sns.color_palette('Blues', 9)[3],
    sns.color_palette('YlOrRd', 9)[3],
    sns.color_palette('crest', 9)[3],
    sns.color_palette('YlOrRd', 9)[7],
]
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axvline(0.05, color='k', linestyle='--', linewidth=0.5)
ax.axhline(0.05, color='k', linestyle='--', linewidth=0.5)
ax.axvline(0.01, color='k', linestyle='--', linewidth=0.5)
ax.axhline(0.01, color='k', linestyle='--', linewidth=0.5)
ax.plot(HData['KS Gamma P-value'][np.where((HData['Direction'] == 'cis')&(HData['MiceID'] == 10209))[0]], 
        HData['KS Gamma P-value'][np.where((HData['Direction'] == 'trs')&(HData['MiceID'] == 10209))[0]], 
        'o', color = colors[0], markeredgewidth=0, markersize=4, alpha=0.8, label = '10209'
)
ax.plot(HData['KS Gamma P-value'][np.where((HData['Direction'] == 'cis')&(HData['MiceID'] == 10212))[0]], 
        HData['KS Gamma P-value'][np.where((HData['Direction'] == 'trs')&(HData['MiceID'] == 10212))[0]], 
        'o', color = colors[1], markeredgewidth=0, markersize=4, alpha=0.8, label = '10212'
)
ax.plot(HData['KS Gamma P-value'][np.where((HData['Direction'] == 'cis')&(HData['MiceID'] == 10224))[0]], 
        HData['KS Gamma P-value'][np.where((HData['Direction'] == 'trs')&(HData['MiceID'] == 10224))[0]], 
        'o', color = colors[2], markeredgewidth=0, markersize=4, alpha=0.8, label = '10224'
)
ax.plot(HData['KS Gamma P-value'][np.where((HData['Direction'] == 'cis')&(HData['MiceID'] == 10227))[0]], 
        HData['KS Gamma P-value'][np.where((HData['Direction'] == 'trs')&(HData['MiceID'] == 10227))[0]], 
        'o', color = colors[3], markeredgewidth=0, markersize=4, alpha=0.8, label = '10227'
)
ax.set_aspect("equal")
ax.axis([0.0001, 1, 0.0001, 1])
ax.semilogx()
ax.semilogy()
plt.legend()
plt.tight_layout()
plt.savefig(join(loc, 'Hairpin KS Gamma-Poisson Pvalues.png'), dpi=600)
plt.savefig(join(loc, 'Hairpin KS Gamma-Poisson Pvalues.svg'), dpi=600)
plt.close()

def add_distribution(
    ax: Axes,
    data: np.ndarray
):
    ax = Clear_Axes(ax, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    xmin, xmax = np.nanmin(data), np.nanmax(data)
    a = ax.hist(
        data,
        range=(1 - 0.5, xmax + 0.5),
        bins=int(xmax),
        rwidth=0.8,
        color='gray'
    )[0]
    ax.set_xlim(1 - 0.5, xmax + 0.5)
    ax.set_xticks([1, xmax])

    prob = a / np.nansum(a)
    lam = EqualPoissonFit(np.arange(1, xmax+1), prob)
    y = EqualPoisson(np.arange(1,xmax+1), l=lam)
    y = y / np.sum(y) * np.sum(a)

    r, p = NegativeBinomialFit(np.arange(1, xmax+1), prob)
    y2 = NegativeBinomial(np.arange(1,xmax+1), r=r, p=p)
    y2 = y2 / np.sum(y2) * np.sum(a)

    num = len(data)
    # Kolmogorov-Smirnov Test for Normal Distribution
    ks_sta, pvalue = poisson_kstest(data)
    print("  KS Test:", ks_sta, pvalue)
    ax.plot(np.arange(1, xmax+1), y, color = 'red', label = 'λ '+str(round(lam,3))+'\n'+f"p {round(pvalue,5)}", linewidth=0.5, alpha = 0.8)
    ax.plot(np.arange(1, xmax+1), y2, color = 'cornflowerblue', label = f"r {round(r,2)} p {round(p, 2)}", linewidth=0.5, alpha = 0.8, ls='--')
    ax.legend(facecolor = 'white', edgecolor = 'white', loc = 'upper right', bbox_to_anchor=(1, 1, 0.3, 0.2))
    ax.set_yticks(ColorBarsTicks(peak_rate=np.max(a), is_auto=True, tick_number=4))
    ax.set_ylim(0, max(np.max(y), np.max(a))+5)
    return a
"""
saveloc = join(loc, "Hairpin Distribution Examples")
mkdir(saveloc)
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
for i in tqdm(range(len(f4))):
    with open(f4['Trace File'][i], 'rb') as handle:
        trace = pickle.load(handle)
        
    field_num_cis = cp.deepcopy(trace['cis']['place_field_num'])
    field_num_trs = cp.deepcopy(trace['trs']['place_field_num'])
    
    field_num_cis = field_num_cis[np.where(field_num_cis >0)[0]]
    field_num_trs = field_num_trs[np.where(field_num_trs >0)[0]]
    
    print(i, len(field_num_cis), len(field_num_trs))
    
    add_distribution(ax1, field_num_cis)
    add_distribution(ax2, field_num_trs)
    
    plt.savefig(join(saveloc, f"{int(f4['MiceID'][i])} - {int(f4['date'][i])}.png"), dpi=600)
    plt.savefig(join(saveloc, f"{int(f4['MiceID'][i])} - {int(f4['date'][i])}.svg"), dpi=600)
    ax1.clear()
    ax2.clear()
    
plt.close()


saveloc = join(loc, "Distribution Examples")
mkdir(saveloc)
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
for i in tqdm(range(13, len(f3))):
    with open(f3['Trace File'][i], 'rb') as handle:
        trace = pickle.load(handle)
        
    field_num_cis = cp.deepcopy(trace['cis']['place_field_num'])
    field_num_trs = cp.deepcopy(trace['trs']['place_field_num'])
    
    field_num_cis = field_num_cis[np.where(field_num_cis >0)[0]]
    field_num_trs = field_num_trs[np.where(field_num_trs >0)[0]]
    
    print(len(field_num_cis), len(field_num_trs))
    
    add_distribution(ax1, field_num_cis)
    add_distribution(ax2, field_num_trs)
    
    plt.savefig(join(saveloc, f"{int(f3['MiceID'][i])} - {int(f3['date'][i])}.png"), dpi=600)
    plt.savefig(join(saveloc, f"{int(f3['MiceID'][i])} - {int(f3['date'][i])}.svg"), dpi=600)
    ax1.clear()
    ax2.clear()
    
plt.close()
"""