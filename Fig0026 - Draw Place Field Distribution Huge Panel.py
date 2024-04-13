from mylib.statistic_test import *

code_id = "0026 - Draw Place Field Distribution Huge Panel"
loc = os.path.join(figpath, code_id)
mkdir(loc)

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
        color='gray',
        density=True
    )[0]
    ax.set_xlim(1 - 0.5, xmax + 0.5)
    ax.set_xticks([1, xmax])

    prob = a / np.nansum(a)
    lam = EqualPoissonFit(np.arange(1, xmax+1), prob)
    y = EqualPoisson(np.arange(1,xmax+1), l=lam)

    r, p = NegativeBinomialFit(np.arange(1, xmax+1), prob)
    y2 = NegativeBinomial(np.arange(1,xmax+1), r=r, p=p)

    num = len(data)
    # Kolmogorov-Smirnov Test for Normal Distribution
    ks_sta, pvalue = poisson_kstest(data)
    print("  KS Test:", ks_sta, pvalue)
    ax.plot(np.arange(1, xmax+1), y, color = 'red', label = 'λ '+str(round(lam,3))+'\n'+f"p {round(pvalue,5)}", linewidth=0.5, alpha = 0.8)
    ax.plot(np.arange(1, xmax+1), y2, color = 'cornflowerblue', label = f"r {round(r,2)} p {round(p, 2)}", linewidth=0.5, alpha = 0.8, ls='--')
    #ax.legend(facecolor = 'white', edgecolor = 'white', loc = 'upper right', bbox_to_anchor=(1, 1, 0.3, 0.2))
    return a
"""
mice = [10209, 10212, 10224, 10227]
fig, axes = plt.subplots(ncols=8, nrows=13, figsize=(8*3,13*2))


for c, m in enumerate(mice):
    idx = np.where((f1['MiceID'] == m)&(f1['Stage'] == 'Stage 1')&(f1['session'] == 1))[0]
    for i, index in enumerate(idx):
        print(m, 'Stage 1', f1['date'][index], 'session', f1['session'][index])
        if os.path.exists(f1['Trace File'][index]) and f1['include'][index] == 1:
            with open(f"{f1['Trace File'][index]}", 'rb') as handle:
                trace = pickle.load(handle)
            
            field_num = []
            for j in range(trace['n_neuron']):
                if trace['is_placecell'][j] == 1:
                    num = len(trace['place_field_all_multiday'][j].keys())
                    if num != 0:
                        field_num.append(len(trace['place_field_all_multiday'][j].keys()))
                    
            field_num = np.array(field_num)
            a = add_distribution(axes[i, c*2], field_num)
            del trace
        else:
            Clear_Axes(axes[i, c*2])
    
    idx = np.where((f1['MiceID'] == m)&(f1['Stage'] == 'Stage 1')&(f1['session'] == 3))[0]
    for i, index in enumerate(idx):
        print(m, 'Stage 1', f1['date'][index], 'session', f1['session'][index])
        if os.path.exists(f1['Trace File'][index]) and f1['include'][index] == 1:
            with open(f"{f1['Trace File'][index]}", 'rb') as handle:
                trace = pickle.load(handle)
            
            field_num = []
            for j in range(trace['n_neuron']):
                if trace['is_placecell'][j] == 1:
                    num = len(trace['place_field_all_multiday'][j].keys())
                    if num != 0:
                        field_num.append(len(trace['place_field_all_multiday'][j].keys()))
                    
            field_num = np.array(field_num)
            a = add_distribution(axes[i, c*2+1], field_num)
            del trace
        else:
            Clear_Axes(axes[i, c*2+1])
            
plt.tight_layout()
plt.savefig(join(loc, "Stage 1 Open Field.png"), dpi=600)
plt.savefig(join(loc, "Stage 1 Open Field.svg"), dpi=600)

plt.close()



mice = [10209, 10212, 10224, 10227, 11095, 11092]
fig, axes = plt.subplots(ncols=12, nrows=13, figsize=(12*3,13*2))
for c, m in enumerate(mice):
    idx = np.where((f1['MiceID'] == m)&(f1['Stage'] == 'Stage 2')&(f1['session'] == 1))[0]
    for i, index in enumerate(idx):
        print(m, 'Stage 1', f1['date'][index], 'session', f1['session'][index])
        if os.path.exists(f1['Trace File'][index]) and f1['include'][index] == 1:
            with open(f"{f1['Trace File'][index]}", 'rb') as handle:
                trace = pickle.load(handle)
            
            field_num = []
            for j in range(trace['n_neuron']):
                if trace['is_placecell'][j] == 1:
                    num = len(trace['place_field_all_multiday'][j].keys())
                    if num != 0:
                        field_num.append(len(trace['place_field_all_multiday'][j].keys()))
                    
            field_num = np.array(field_num)
            a = add_distribution(axes[i, c*2], field_num)
            del trace
        else:
            Clear_Axes(axes[i, c*2])
    
    idx = np.where((f1['MiceID'] == m)&(f1['Stage'] == 'Stage 2')&(f1['session'] == 4))[0]
    for i, index in enumerate(idx):
        print(m, 'Stage 1', f1['date'][index], 'session', f1['session'][index])
        if os.path.exists(f1['Trace File'][index]) and f1['include'][index] == 1:
            with open(f"{f1['Trace File'][index]}", 'rb') as handle:
                trace = pickle.load(handle)
            
            field_num = []
            for j in range(trace['n_neuron']):
                if trace['is_placecell'][j] == 1:
                    num = len(trace['place_field_all_multiday'][j].keys())
                    if num != 0:
                        field_num.append(len(trace['place_field_all_multiday'][j].keys()))
                    
            field_num = np.array(field_num)
            a = add_distribution(axes[i, c*2+1], field_num)
            del trace
        else:
            Clear_Axes(axes[i, c*2+1])
            
plt.tight_layout()
plt.savefig(join(loc, "Stage 2 Open Field.png"), dpi=600)
plt.savefig(join(loc, "Stage 2 Open Field.svg"), dpi=600)
plt.close()
"""

mice = [10209, 10212, 10224, 10227, 11095, 11092]
fig, axes = plt.subplots(ncols=12, nrows=14, figsize=(12*3,14*2))
for c, m in enumerate(mice):
    idx = np.where((f1['MiceID'] == m)&(f1['Stage'] == 'Stage 1')&(f1['session'] == 2)&(f1['date'] >= 20220813))[0]
    tot = np.zeros(25)
    tot_field = []
    for i, index in enumerate(idx):
        print(m, 'Stage 1', f1['date'][index], 'session', f1['session'][index])
        p = join(r'F:\Data\Cross_maze', str(int(f1['MiceID'][index])), str(int(f1['date'][index])), 'session '+str(int(f1['session'][index])), 'trace.pkl')
        if os.path.exists(p) and f1['include'][index] == 1:
            with open(p, 'rb') as handle:
                trace = pickle.load(handle)
            
            field_num = trace['place_field_num'][np.where(trace['is_placecell'] == 1)[0]]
            a = add_distribution(axes[i, c*2], field_num)
            del trace
        else:
            Clear_Axes(axes[i, c*2])
    
    tot = np.zeros(25)
    tot_field = []
    idx = np.where((f1['MiceID'] == m)&(f1['Stage'] == 'Stage 2')&(f1['session'] == 2)&(f1['date'] >= 20220813))[0]
    for i, index in enumerate(idx):
        print(m, 'Stage 1', f1['date'][index], 'session', f1['session'][index])
        p = join(r'F:\Data\Cross_maze', str(int(f1['MiceID'][index])), str(int(f1['date'][index])), 'session '+str(int(f1['session'][index])), 'trace.pkl')
        if os.path.exists(p) and f1['include'][index] == 1:
            with open(p, 'rb') as handle:
            #if os.path.exists(f1['Trace File'][index]) and f1['include'][index] == 1:
            #    with open(f"{f1['Trace File'][index]}", 'rb') as handle:
                trace = pickle.load(handle)
            
            field_num = trace['place_field_num'][np.where(trace['is_placecell'] == 1)[0]]
            a = add_distribution(axes[i, c*2+1], field_num)
            del trace
        else:
            Clear_Axes(axes[i, c*2+1])
            
plt.tight_layout()
plt.savefig(join(loc, "Stage 1+2 Maze 1.png"), dpi=600)
plt.savefig(join(loc, "Stage 1+2 Maze 1.svg"), dpi=600)
plt.close()


mice = [10209, 10212, 10224, 10227,11095, 11092]
fig, axes = plt.subplots(ncols=6, nrows=14, figsize=(6*3,14*2))
for c, m in enumerate(mice):
    idx = np.where((f1['MiceID'] == m)&(f1['Stage'] == 'Stage 2')&(f1['session'] == 3)&(f1['include'] == 1))[0]
    tot = np.zeros(25)
    tot_field = []
    for i, index in enumerate(idx):
        print(m, 'Stage 1', f1['date'][index], 'session', f1['session'][index])
        p = join(r'F:\Data\Cross_maze', str(int(f1['MiceID'][index])), str(int(f1['date'][index])), 'session '+str(int(f1['session'][index])), 'trace.pkl')
        if os.path.exists(p) and f1['include'][index] == 1:
            with open(p, 'rb') as handle:
            #if os.path.exists(f1['Trace File'][index]) and f1['include'][index] == 1:
            #    with open(f"{f1['Trace File'][index]}", 'rb') as handle:
                trace = pickle.load(handle)
            
            field_num = trace['place_field_num'][np.where(trace['is_placecell'] == 1)[0]]
            a = add_distribution(axes[i, c], field_num)
            del trace
        else:
            Clear_Axes(axes[i, c])
            
plt.tight_layout()
plt.savefig(join(loc, "Stage 2 Maze 2.png"), dpi=600)
plt.savefig(join(loc, "Stage 2 Maze 2.svg"), dpi=600)
plt.close()
"""

# "Reverse Maze 1"
mice = [10209, 10212]
fig, axes = plt.subplots(ncols=4, nrows=12, figsize=(4*3,12*2))
for c, m in enumerate(mice):
    idx = np.where((f3['MiceID'] == m))[0]
    
    for i, index in enumerate(idx):
        print(m, f3['date'][index], 'session', f3['session'][index])
        if os.path.exists(f3['Trace File'][index]) and f3['include'][index] == 1:
            with open(f"{f3['Trace File'][index]}", 'rb') as handle:
                trace = pickle.load(handle)
            
            field_num = []
            for j in range(trace['n_neuron']):
                if trace['cis']['is_placecell'][j] == 1:
                    num = len(trace['cis']['place_field_all_multiday'][j].keys())
                    if num != 0:
                        field_num.append(len(trace['cis']['place_field_all_multiday'][j].keys()))
            
            a = add_distribution(axes[i, c*2], field_num)

            field_num = []
            for j in range(trace['n_neuron']):
                if trace['trs']['is_placecell'][j] == 1:
                    num = len(trace['trs']['place_field_all_multiday'][j].keys())
                    if num != 0:
                        field_num.append(len(trace['trs']['place_field_all_multiday'][j].keys()))
            a = add_distribution(axes[i, c*2+1], field_num)
        else:
            Clear_Axes(axes[i, c*2+1])
    
plt.tight_layout()
plt.savefig(join(loc, "Reverse Maze.png"), dpi=600)
plt.savefig(join(loc, "Reverse Maze.svg"), dpi=600)
plt.close()


# "Hairpin Maze 1"
mice = [10209, 10212]
fig, axes = plt.subplots(ncols=4, nrows=7, figsize=(4*3,7*2))
for c, m in enumerate(mice):
    idx = np.where((f4['MiceID'] == m))[0]
    
    for i, index in enumerate(idx):
        print(m, f4['date'][index], 'session', f4['session'][index])
        if os.path.exists(f4['Trace File'][index]) and f4['include'][index] == 1:
            with open(f"{f4['Trace File'][index]}", 'rb') as handle:
                trace = pickle.load(handle)
            
            field_num = []
            for j in range(trace['n_neuron']):
                if trace['cis']['is_placecell'][j] == 1:
                    num = len(trace['cis']['place_field_all_multiday'][j].keys())
                    if num != 0:
                        field_num.append(len(trace['cis']['place_field_all_multiday'][j].keys()))
            
            a = add_distribution(axes[i, c*2], field_num)

            field_num = []
            for j in range(trace['n_neuron']):
                if trace['trs']['is_placecell'][j] == 1:
                    num = len(trace['trs']['place_field_all_multiday'][j].keys())
                    if num != 0:
                        field_num.append(len(trace['trs']['place_field_all_multiday'][j].keys()))
            a = add_distribution(axes[i, c*2+1], field_num)
        else:
            Clear_Axes(axes[i, c*2+1])
    
plt.tight_layout()
plt.savefig(join(loc, "Hairpin Maze.png"), dpi=600)
plt.savefig(join(loc, "Hairpin Maze.svg"), dpi=600)
plt.close()
"""