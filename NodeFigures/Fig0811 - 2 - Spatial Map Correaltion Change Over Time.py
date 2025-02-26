from mylib.statistic_test import *

code_id = '0811 - Cross-route correlation'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)


if os.path.exists(os.path.join(figdata, code_id+' [with length].pkl')):
    with open(os.path.join(figdata, code_id+' [with length].pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    raise FileNotFoundError(f"Please run Fig0811 - Cross-route Correlation.py to generate this file.")


if os.path.exists(os.path.join(figdata, code_id+' [Plus].pkl')) == False:
    SData = {
        "MiceID": [],
        "Routes": [],
        "Correlation": [],
        "Training Day": [],
        "Group": []
    }
    
    route_mat = np.zeros((10, 10))
    route_mat[:, np.array([0, 4, 5, 9])] = 1
    route_mat[:, 1] = 2
    route_mat[:, 2] = 3
    route_mat[:, 3] = 4
    route_mat[:, 6] = 5
    route_mat[:, 7] = 6
    route_mat[:, 8] = 7
    
    triu_idx = np.concatenate([
        np.arange(1, 10),
        np.arange(4, 10),
        np.arange(5, 10)
    ])
    I = np.intersect1d
    
    for mice in [10209, 10212, 10224, 10227, 10232]:
        idx = np.where(f2['MiceID'] == mice)[0]
    
        print(f"Mouse {mice}")
        for i in tqdm(idx):
            with open(f2['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
            
            corr_value = trace['route_wise_corr'][3, np.concatenate([np.arange(3), np.arange(4, 10)])]
            routes = route_mat[3, np.concatenate([np.arange(3), np.arange(4, 10)])]
            
            SData['MiceID'].append(np.repeat(mice, routes.shape[0]))
            SData['Correlation'].append(corr_value)
            SData['Routes'].append(routes)
            SData['Training Day'].append(np.repeat(f2['training_day'][i], routes.shape[0]))
            SData['Group'].append(np.repeat('Exp.2', routes.shape[0]))

    for k in SData.keys():
        SData[k] = np.concatenate(SData[k])
    
    with open(os.path.join(figdata, code_id+' [Plus].pkl'), 'wb') as handle:
        pickle.dump(SData, handle)
        
    D = pd.DataFrame(SData)
    D.to_excel(os.path.join(figdata, code_id+' [Plus].xlsx'), index = False)
else:
    with open(os.path.join(figdata, code_id+' [Plus].pkl'), 'rb') as handle:
        SData = pickle.load(handle)

for k in Data.keys():
    Data[k] = np.concatenate([Data[k], SData[k]])


fig = plt.figure(figsize=(2, 3))
idx = np.where((np.isin(Data['Routes'], [2, 6, 3])) & (Data['Group'] == 'Exp.'))[0]
SubData = SubDict(Data, Data.keys(), idx)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

sns.lineplot(
    x = 'Training Day',
    y = 'Correlation',
    hue = 'Routes',
    hue_order=[2, 6, 3],
    data = SubData,
    palette=[DSPPalette[2], DSPPalette[6], DSPPalette[3]],
    linewidth=0.5,
    err_kws={'edgecolor':None},
    ax=ax,
    #err_style="bars",
    #err_kws={'elinewidth': 0.5, 'capsize': 3, 'capthick': 0.5},
    marker='o',
    markersize=4,
    markeredgewidth = 0,
)
ax.set_ylim([0, 0.6])
ax.set_yticks(np.linspace(0, 0.6, 7))
plt.savefig(join(loc, "Similarity with Route 1.png"), dpi = 600)
plt.savefig(join(loc, "Similarity with Route 1.svg"), dpi = 600)
plt.show()

for i in [2, 6, 3]:
    idxs = [
        np.where(
            (Data['Routes'] == i) & 
            (Data['Group'] == 'Exp.') & 
            (np.isnan(Data['Correlation']) == False) & 
            (Data['Training Day'] == f"Day {d}")
        )[0] for d in range(1, 8) 
    ]
    x = np.concatenate([np.repeat(i, idx.shape[0]) for i, idx in enumerate(idxs)])
    y = np.concatenate([Data['Correlation'][idx] for idx in idxs])
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    print(f"Route {i}: {p_value}")
    
    

fig = plt.figure(figsize=(2, 3))
idx = np.where((np.isin(Data['Routes'], [3, 7, 4])) & (Data['Group'] == 'Exp.2'))[0]
SubData = SubDict(Data, Data.keys(), idx)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

sns.lineplot(
    x = 'Training Day',
    y = 'Correlation',
    hue = 'Routes',
    hue_order=[3, 7, 4],
    data = SubData,
    palette=[DSPPalette[2], DSPPalette[6], DSPPalette[3]],
    linewidth=0.5,
    err_style="bars",
    err_kws={'elinewidth': 0.5, 'capsize': 3, 'capthick': 0.5},
    marker='o',
    markersize=4,
    markeredgewidth = 0,
)
ax.set_ylim([0, 0.5])
ax.set_yticks(np.linspace(0, 0.5, 6))
plt.savefig(join(loc, "Similarity with Route 4.png"), dpi = 600)
plt.savefig(join(loc, "Similarity with Route 4.svg"), dpi = 600)
plt.show()

for i in [3, 7]:
    idxs = [
        np.where(
            (Data['Routes'] == i) & 
            (Data['Group'] == 'Exp.2') & 
            (np.isnan(Data['Correlation']) == False) & 
            (Data['Training Day'] == f"Day {d}")
        )[0] for d in range(1, 8) 
    ]
    x = np.concatenate([np.repeat(i, idx.shape[0]) for i, idx in enumerate(idxs)])
    y = np.concatenate([Data['Correlation'][idx] for idx in idxs])
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    print(f"Route {i}: {p_value}")