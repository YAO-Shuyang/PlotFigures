from mylib.statistic_test import *
import networkx as nx
# 3d plot
from mpl_toolkits.mplot3d import Axes3D
from mylib.maze_graph import CP_DSP as CP
from mylib.calcium.dsp_ms import get_son_area

code_id = '0811 - Cross-route correlation'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    mat = np.zeros((50, 70))
    for j, mice in enumerate([10209, 10212, 10224, 10227, 10232]):
        idx = np.where(f2['MiceID'] == mice)[0]
    
        for d, i in enumerate(idx):
            with open(f2['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
        
            idx = np.ix_(np.arange(10) + 10 * j, np.arange(10) + 10 * d)
            mat[idx] += trace['route_wise_corr']
            mat[(np.arange(10) + 10 * j, np.arange(10) + 10 * d)] = np.nan
    
    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(mat, handle)
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        mat = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata, code_id+' [with length].pkl')) == False:

    Data = {
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
            
            corr_value = np.concatenate([
                trace['route_wise_corr'][0, np.arange(1, 10)],
                trace['route_wise_corr'][4, np.concatenate([np.arange(4), np.arange(5, 10)])],
                trace['route_wise_corr'][5, np.concatenate([np.arange(5), np.arange(6, 10)])],
                trace['route_wise_corr'][9, np.arange(9)]
            ])
            routes = np.concatenate([
                route_mat[0, np.arange(1, 10)],
                route_mat[4, np.concatenate([np.arange(4), np.arange(5, 10)])],
                route_mat[5, np.concatenate([np.arange(5), np.arange(6, 10)])],
                route_mat[9, np.arange(9)]
            ])
            
            Data['MiceID'].append(np.repeat(mice, routes.shape[0]))
            Data['Correlation'].append(corr_value)
            Data['Routes'].append(routes)
            Data['Training Day'].append(np.repeat(f2['training_day'][i], routes.shape[0]))
            Data['Group'].append(np.repeat('Exp.', routes.shape[0]))
            
            # Calculate control correlation
            group = [(0, 4), (0, 5), (0, 9), (4, 5), (4, 9), (5, 9)]
            ctrl_corr = np.zeros((6, 6))
            ctrl_dist = np.vstack([np.arange(2, 8) for i in range(6)])
            ctrl_type = np.concatenate([
                np.repeat(i, 7) for i in ['0-4', '0-5', '0-9', '4-5', '4-9', '5-9']
            ])
            for n, item in enumerate(group):
                j, k = item
                pc_idx = np.where(
                    (trace[f'node {j}']['is_placecell'] == 1) |
                    (trace[f'node {k}']['is_placecell'] == 1)
                )[0]
                for route in range(1, 7):
                    bins = get_son_area(CP_DSP[route])-1
                    
                    corr = np.zeros(trace['n_neuron'], np.float64) * np.nan
                    for cell in pc_idx:
                        corr[cell], _ = pearsonr(
                            trace['node '+str(j)]['smooth_map_all'][cell, bins], 
                            trace['node '+str(k)]['smooth_map_all'][cell, bins]
                        )
                    ctrl_corr[n, route-1] = np.nanmean(corr)
                    
            ctrl_corr = ctrl_corr.flatten()
            ctrl_dist = ctrl_dist.flatten()
            
            Data['MiceID'].append(np.repeat(mice, ctrl_dist.shape[0]))
            Data['Correlation'].append(ctrl_corr)
            Data['Routes'].append(ctrl_dist)
            Data['Training Day'].append(np.repeat(f2['training_day'][i], ctrl_dist.shape[0]))
            Data['Group'].append(np.repeat('Ctrl.', ctrl_dist.shape[0]))
            
            rs = np.array([0, 1, 2, 3, 0, 0, 4, 5, 6, 0])
            for j in tqdm(range(10)):
                shuf_corr = np.zeros((10, 10), dtype=np.float64) * np.nan
            
                for x in range(len(shuf_corr)-1):
                    for y in range(x+1, len(shuf_corr)):
                        idx = np.where(
                            (trace['node '+str(x)]['is_placecell'] == 1) |
                            (trace['node '+str(y)]['is_placecell'] == 1)
                        )[0]
                        bins = get_son_area(I(CP_DSP[rs[x]], CP_DSP[rs[y]]))-1
                        shuf_corr[x, y] = np.nanmean(
                            np.array([
                                pearsonr(
                                    trace['node '+str(x)]['smooth_map_all'][cell_x, bins],
                                    trace['node '+str(y)]['smooth_map_all'][cell_y, bins]
                                )[0] for cell_x, cell_y in zip(idx, np.random.permutation(idx))
                            ])
                        )
                        
                shuf_corr = np.concatenate([
                    shuf_corr[0, np.arange(1, 10)],
                    shuf_corr[4, np.concatenate([np.arange(4), np.arange(5, 10)])],
                    shuf_corr[5, np.concatenate([np.arange(5), np.arange(6, 10)])],
                    shuf_corr[9, np.arange(9)]
                ])
                routes = np.concatenate([
                    route_mat[0, np.arange(1, 10)],
                    route_mat[4, np.concatenate([np.arange(4), np.arange(5, 10)])],
                    route_mat[5, np.concatenate([np.arange(5), np.arange(6, 10)])],
                    route_mat[9, np.arange(9)]
                ])
                
                Data['MiceID'].append(np.repeat(mice, routes.shape[0]))
                Data['Correlation'].append(shuf_corr)
                Data['Routes'].append(routes)
                Data['Training Day'].append(np.repeat(f2['training_day'][i], routes.shape[0]))
                Data['Group'].append(np.repeat('Shuffle', routes.shape[0]))
                
    for k in Data.keys():
        Data[k] = np.concatenate(Data[k])
    
    with open(os.path.join(figdata, code_id+' [with length].pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
        
    D = pd.DataFrame(Data)
    D.to_excel(os.path.join(figdata, code_id+' [with length].xlsx'), index = False)
else:
    with open(os.path.join(figdata, code_id+' [with length].pkl'), 'rb') as handle:
        Data = pickle.load(handle)
 
idx = np.where(Data['Routes'] != 1)[0]
Data = SubDict(Data, Data.keys(), idx)

print(np.unique(Data['Routes']))

fig = plt.figure(figsize=(2.5,3))
idx = np.concatenate([
    np.where((Data['Routes'] == route)&(np.isnan(Data['Correlation']) == False))[0] for route in [5, 2, 6, 3, 7, 4]
])
Data = SubDict(Data, Data.keys(), idx)
Data['X'] = np.array([f"{i}" for i in Data['Routes']])
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
box = sns.boxplot(
    x = 'X', 
    y = 'Correlation', 
    data = Data, 
    hue_order=['Ctrl.', 'Exp.', 'Shuffle'],
    hue = 'Group',
    ax=ax,
    #err_style="bars",
    linewidth=0.5,
    width=0.8,
    linecolor='black',
    gap=0.1,
    fliersize=0,
    palette=[DSPPalette[0], DSPPalette[1], sns.color_palette("Grays", 3)[1]],
    #err_kws={'elinewidth': 0.5, 'capsize': 4, 'capthick': 0.5},
    #marker='o',
    #markersize=4,
    #markeredgewidth = 0,
    #palette=DSPPalette[:3]
)
for b in box.patches:
    b.set_linewidth(0)
    
ax.set_ylim(-0.1, 0.8)
ax.set_yticks(np.linspace(-0.1, 0.8, 10))
plt.savefig(os.path.join(loc, f'withlength.svg'), dpi=600)
plt.savefig(os.path.join(loc, f'withlength.png'), dpi=600)
plt.show()

for route in range(2, 8):
    idx_exp = np.where((Data['Routes'] == route) & (Data['Group'] == 'Exp.'))[0]
    idx_ctrl = np.where((Data['Routes'] == route) & (Data['Group'] == 'Ctrl.'))[0]
    idx_shuf = np.where((Data['Routes'] == route) & (Data['Group'] == 'Shuffle'))[0]
    print(f"Route {route}:")
    print(f"    Exp. vs. Control", ttest_ind(Data['Correlation'][idx_exp], Data['Correlation'][idx_ctrl]))
    print(f"    Exp. vs. Shuffle", ttest_ind(Data['Correlation'][idx_exp], Data['Correlation'][idx_shuf]), end='\n\n')

# Sort Day
sort_day_idx = np.concatenate([np.arange(7)*10 + i for i in range(10)])
sort_mice_idx = np.concatenate([np.arange(4, -1, -1)*10 + i for i in range(10)])
sorted_mat = mat[sort_mice_idx, :][:, sort_day_idx]
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes())
sns.heatmap(
    sorted_mat, vmin = 0, ax=ax, cmap='Blues'
)
ax.set_aspect("equal")
plt.savefig(os.path.join(loc, f'allmice.svg'), dpi=600)
plt.savefig(os.path.join(loc, f'allmice.png'), dpi=600)
plt.close()

