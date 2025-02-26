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
            
    bins = [
        np.concatenate([Father2SonGraph[i] for i in CP_DSP[j]])-1
        for j in [0, 1, 2, 3, 0, 0, 4, 5, 6, 0]
    ]    
    routes_stad = [0, 1, 2, 3, 0, 0, 4, 5, 6, 0]
    for mice in [10209, 10212, 10224, 10227, 10232]:
        idx = np.where(f2['MiceID'] == mice)[0]
    
        print(f"Mouse {mice}")
        for i in tqdm(idx):
            with open(f2['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
            
            for r in [1, 2, 3, 6, 7, 8]:
                if r in [1, 2, 3]:
                    ctrl0, ctrl1 = 0, 4
                else:
                    ctrl0, ctrl1 = 5, 9
                
                corr = np.zeros((trace['n_neuron'], 3), np.float64)
                
                for k in range(trace['n_neuron']):
                    corr[k, 0] = np.corrcoef(
                        trace[f'node {r}']['smooth_map_all'][k, bins[r]], 
                        trace[f'node {ctrl0}']['smooth_map_all'][k, bins[r]],
                    )[0, 1]
                    
                    corr[k, 1] = np.corrcoef(
                        trace[f'node {r}']['smooth_map_all'][k, bins[r]], 
                        trace[f'node {ctrl1}']['smooth_map_all'][k, bins[r]],
                    )[0, 1]
                
                    corr[k, 2] = np.corrcoef(
                        trace[f'node {ctrl0}']['smooth_map_all'][k, bins[r]], 
                        trace[f'node {ctrl1}']['smooth_map_all'][k, bins[r]],
                    )[0, 1]
                
                corr_value = np.nanmean(corr[:, :2])
                ctrl_value = np.nanmean(corr[:, 2])
                
                shuf_corr = np.zeros((trace['n_neuron'], 10), np.float64)
                cell_idx_perm = [
                    np.random.permutation(trace['n_neuron']) for i in range(10)
                ]
                for k in range(trace['n_neuron']):
                    for j in range(10):
                        shuf_corr[k, j] = np.corrcoef(
                            trace[f'node {r}']['smooth_map_all'][k, bins[r]], 
                            trace[f'node {r}']['smooth_map_all'][cell_idx_perm[j][k], bins[r]],
                        )[0, 1]
                
                shuf_value = np.nanmean(shuf_corr)
    
                Data['MiceID'].append(np.repeat(mice, 3))
                Data['Correlation'].append(np.array([corr_value, ctrl_value, shuf_value]))
                Data['Routes'].append(np.repeat(routes_stad[r], 3))
                Data['Training Day'].append(np.repeat(f2['training_day'][i], 3))
                Data['Group'].append(np.array(["Exp.", "Ctrl.", "Shuffle"]))
                
    for k in Data.keys():
        Data[k] = np.concatenate(Data[k])
        
    print(Data['Correlation'].shape)
    
    with open(os.path.join(figdata, code_id+' [with length].pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
        
    D = pd.DataFrame(Data)
    D.to_excel(os.path.join(figdata, code_id+' [with length].xlsx'), index = False)
else:
    with open(os.path.join(figdata, code_id+' [with length].pkl'), 'rb') as handle:
        Data = pickle.load(handle)

print(np.unique(Data['Routes']))
SubData = SubDict(Data, Data.keys(), np.where(Data['MiceID'] != 10209)[0])
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
box = sns.boxplot(
    x = 'Routes', 
    order=[4, 1, 5, 2, 6, 3],
    y = 'Correlation', 
    data = SubData, 
    hue_order=['Ctrl.', 'Exp.', 'Shuffle'],
    hue = 'Group',
    ax=ax,
    #err_style="bars",
    linewidth=0.5,
    width=0.8,
    linecolor='black',
    gap=0.1,
    fliersize=1,
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

for route in [4, 1, 5, 2, 6, 3]:
    idx_exp = np.where((Data['Routes'] == route) & (Data['Group'] == 'Exp.'))[0]
    idx_ctrl = np.where((Data['Routes'] == route) & (Data['Group'] == 'Ctrl.'))[0]
    idx_shuf = np.where((Data['Routes'] == route) & (Data['Group'] == 'Shuffle'))[0]
    print(f"Route {route}:")
    print(f"    Exp. vs. Control", ttest_rel(Data['Correlation'][idx_exp], Data['Correlation'][idx_ctrl], alternative='less'))
    print(f"    Exp. vs. Shuffle", ttest_rel(Data['Correlation'][idx_exp], Data['Correlation'][idx_shuf], alternative='greater'), end='\n\n')

# Sort Day
sort_day_idx = np.concatenate([np.arange(7)*10 + i for i in range(10)])
sort_mice_idx = np.concatenate([np.arange(4, 0, -1)*10 + i for i in range(10)])
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

