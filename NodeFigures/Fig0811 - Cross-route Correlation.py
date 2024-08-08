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
    mat = np.zeros((10, 10))
    for mice in [10209, 10212, 10224, 10227]:
        idx = np.where(f2['MiceID'] == mice)[0]
    
        for i in idx:
            with open(f2['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
        
            mat += trace['route_wise_corr']
        
    mat /= 28
    mat[(np.arange(10), np.arange(10))] = np.nan
    
    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(mat, handle)
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        mat = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata, code_id+' [with length].pkl')) == False:
    I = np.intersect1d
    routes = [0, 1, 2, 3, 0, 0, 4, 5, 6, 0]
    shared_dist = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            shared_dist[i, j] = I(CP[routes[i]], CP[routes[j]]).shape[0]
            
    dist = np.triu(shared_dist, 1)
    triu_idx = np.where(dist != 0)
    dist = dist[triu_idx]
    print(dist)
    Data = {
        "MiceID": [],
        "Shared Distance": [],
        "Correlation": [],
        "Date": [],
        "Group": [],
        "Control Type": []
    }
    
    for mice in [10209, 10212, 10224, 10227]:
        idx = np.where(f2['MiceID'] == mice)[0]
    
        print(f"Mouse {mice}")
        for i in tqdm(idx):
            with open(f2['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
            
            Data['MiceID'].append(np.repeat(mice, dist.shape[0]))
            Data['Correlation'].append(trace['route_wise_corr'][triu_idx])
            Data['Shared Distance'].append(dist)
            Data['Date'].append(np.repeat(f2['date'][i], dist.shape[0]))
            Data['Group'].append(np.repeat('Exp.', dist.shape[0]))
            Data['Control Type'].append(np.repeat('Not Control', dist.shape[0]))
            
            # Calculate control correlation
            group = [(0, 4), (0, 5), (0, 9), (4, 5), (4, 9), (5, 9)]
            ctrl_corr = np.zeros((6, 7))
            ctrl_dist = np.zeros((6, 7))
            ctrl_type = np.concatenate([
                np.repeat(i, 7) for i in ['0-4', '0-5', '0-9', '4-5', '4-9', '5-9']
            ])
            for n, item in enumerate(group):
                j, k = item
                pc_idx = np.where(
                    (trace[f'node {j}']['is_placecell'] == 1) |
                    (trace[f'node {k}']['is_placecell'] == 1)
                )[0]
                for route in range(0, 7):
                    ctrl_dist[n, route] = np.intersect1d(CP[0], CP_DSP[route]).shape[0]
                    bins = get_son_area(CP_DSP[route])-1
                    
                    corr = np.zeros(trace['n_neuron'], np.float64) * np.nan
                    for cell in pc_idx:
                        corr[cell], _ = pearsonr(
                            trace['node '+str(j)]['smooth_map_all'][cell, bins], 
                            trace['node '+str(k)]['smooth_map_all'][cell, bins]
                        )
                    ctrl_corr[n, route] = np.nanmean(corr)
                    
            ctrl_corr = ctrl_corr.flatten()
            ctrl_dist = ctrl_dist.flatten()
            
            Data['MiceID'].append(np.repeat(mice, ctrl_dist.shape[0]))
            Data['Correlation'].append(ctrl_corr)
            Data['Shared Distance'].append(ctrl_dist)
            Data['Date'].append(np.repeat(f2['date'][i], ctrl_dist.shape[0]))
            Data['Group'].append(np.repeat('Ctrl.', ctrl_dist.shape[0]))
            Data['Control Type'].append(ctrl_type)
        
    for k in Data.keys():
        Data[k] = np.concatenate(Data[k])
    
    with open(os.path.join(figdata, code_id+' [with length].pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
        
    D = pd.DataFrame(Data)
    D.to_excel(os.path.join(figdata, code_id+' [with length].xlsx'), index = False)
else:
    with open(os.path.join(figdata, code_id+' [with length].pkl'), 'rb') as handle:
        Data = pickle.load(handle)

idx = np.where(Data['Control Type'] != 'Not Control')[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(2.5,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Shared Distance', 
    y = 'Correlation', 
    data = SubData, 
    hue = 'Control Type',
    ax=ax,
    linewidth=0.5,
    marker='o',
    markersize=4,
    markeredgecolor = None,
    palette="rainbow",
    err_style='bars',
    err_kws={'capsize': 3, 'elinewidth': 0.5, 'capthick': 0.5},
)
plt.show()
"""        
fig = plt.figure(figsize=(2.5,3))
Data['Shared Distance'] = 111-Data['Shared Distance']
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Shared Distance', 
    y = 'Correlation', 
    data = Data, 
    hue = 'Group',
    ax=ax,
    linewidth=0.5,
    marker='o',
    markersize=4,
    markeredgecolor = None,
    palette=['#003366', '#0099CC'],
    err_style='bars',
    err_kws={'capsize': 3, 'elinewidth': 0.5, 'capthick': 0.5},
)
ax.set_xlim([-5, 111])
ax.set_xticks([0, 11, 31, 51, 71, 91, 111])
ax.set_ylim([0, 0.6])
ax.set_yticks(np.linspace(0, 0.6, 7))
plot_segments(ax, dy=0.05)
plt.savefig(os.path.join(loc, f'withlength.svg'), dpi=600)
plt.savefig(os.path.join(loc, f'withlength.png'), dpi=600)
plt.close()

uniq_dist = np.unique(Data['Shared Distance'])
for i in range(uniq_dist.shape[0]):
    idx_exp = np.where((Data['Shared Distance'] == uniq_dist[i])&(Data['Group'] == 'Exp.'))[0]
    idx_ctrl = np.where((Data['Shared Distance'] == uniq_dist[i])&(Data['Group'] == 'Ctrl.'))[0]
    print(uniq_dist[i], ttest_ind(Data['Correlation'][idx_exp], Data['Correlation'][idx_ctrl]))

fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes())
sns.heatmap(
    mat, vmin = 0, ax=ax
)
ax.set_aspect("equal")
plt.savefig(os.path.join(loc, f'allmice.svg'), dpi=600)
plt.savefig(os.path.join(loc, f'allmice.png'), dpi=600)
plt.close()


mat[(np.arange(10), np.arange(10))] = 0

G = nx.Graph(mat)
pos = nx.spring_layout(G, k=1, weight='weight', scale=2, dim=3, threshold=0.00001, seed=42)
colors = [
    DSPPalette[0], DSPPalette[1], DSPPalette[2], DSPPalette[3], DSPPalette[0],
    DSPPalette[0], DSPPalette[4], DSPPalette[5], DSPPalette[6], DSPPalette[0]]

x = [pos[i][0] for i in range(len(pos))]
y = [pos[i][1] for i in range(len(pos))]
z = [pos[i][2] for i in range(len(pos))]

fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=colors, s=100, edgecolors = 'black', alpha=1)
cmap = sns.color_palette('rocket', as_cmap=True)

vmax = np.max(mat)
for i, j in G.edges():
    edge_color = cmap(mat[i, j] / vmax)
    ax.plot(
        [x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c=edge_color, linewidth = 0.5
    )

ax.view_init(elev=40, azim=130)
plt.savefig(os.path.join(loc, f'graph.svg'), dpi=600)
plt.savefig(os.path.join(loc, f'graph.png'), dpi=600)
plt.show()
"""
