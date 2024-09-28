from mylib.statistic_test import *
from mylib.field.tracker_v2 import Tracker2d
from mylib.field.sfer import get_surface, get_data, compare_fit_recover, compare_fit_retention

code_id = "0334 - Recover and Retention probability"
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, f'{code_id}.pkl')) == False:
    # Loss was determined by RMSE
    Data = {"Spatial Map": [], "Retention Loss": [], "Retention Function": [], "I": [], "A": [],
            "Recovery Loss": [], "Recovery Function": []}
    
    data_info = [
        (1, 'CrossMaze'),
        (2, 'CrossMaze'),
        (3, 'HairpinMaze', 'cis'),
        (3, 'HairpinMaze', 'trs'),
        (1, 'ReverseMaze', 'cis'),
        (1, 'ReverseMaze', 'trs'),
    ]
    
    spatial_maps = ['MA', 'MB', 'HPf', 'HPb', 'MAf', 'MAb']
    
    for i, info in enumerate(data_info):
        I, A, P = get_data(*info)
        
        
        IV, RMSE, func_names = compare_fit_retention(I, A, P)
        Data['Retention Loss'].append(RMSE)
        Data['Retention Function'].append(func_names)
        Data['I'].append(IV)
        
        AV, RMSE, func_names = compare_fit_recover(I, A, P)
        Data['Recovery Loss'].append(RMSE)
        Data['Recovery Function'].append(func_names)
        Data['A'].append(AV)
        
        Data['Spatial Map'] += [spatial_maps[i]] * len(AV)
    
    for k in Data.keys():
        if k != 'Spatial Map':
            Data[k] = np.concatenate(Data[k])
    
    Data['Spatial Map'] = np.array(Data['Spatial Map'])
    
    with open(join(figdata, f'{code_id}.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
        
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, f"{code_id}.xlsx"), index = False)
    print(Data['Spatial Map'].shape)
else:
    with open(join(figdata, f'{code_id}.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
fig = plt.figure(figsize=(2,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Retention Function',
    y = 'Retention Loss',
    data=Data,
    hue = 'Retention Function',
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    width=0.7,
    capsize=0.2,
    err_kws={'linewidth': 0.5, 'color': 'black'},
    ax = ax,
    zorder = 1
)
sns.stripplot(
    x = 'Retention Function',
    y = 'Retention Loss',
    data=Data,
    hue='Spatial Map',
    palette=['#D4C9A8', '#8E9F85', '#527C5A', '#C3AED6', '#66C7B4', '#A7D8DE'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    jitter=0.2,
    zorder = 0
)
ax.set_ylim([0, 0.5])
ax.set_yticks(np.linspace(0, 0.5, 6))
plt.savefig(join(loc, 'RMSE [Retention].png'), dpi = 600)
plt.savefig(join(loc, 'RMSE [Retention].svg'), dpi = 600)
plt.close()

func_names = np.unique(Data['Retention Function'])
print("Fit Retention Prob.:")
for i in range(len(func_names)-1):
    for j in range(i+1, len(func_names)):
        idx1 = np.where((Data['Retention Function'] == func_names[i]))[0]
        idx2 = np.where((Data['Retention Function'] == func_names[j]))[0]
        d = np.where(
            (np.isnan(Data['Retention Loss'][idx1]) == False) &
            (np.isnan(Data['Retention Loss'][idx2]) == False)
        )
        print(f"    {func_names[i]} vs {func_names[j]}: {ttest_rel(Data['Retention Loss'][idx1[d]], Data['Retention Loss'][idx2[d]])}")
print()

# Recovery Fitting
fig = plt.figure(figsize=(2,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Recovery Function',
    y = 'Recovery Loss',
    data=Data,
    hue = 'Recovery Function',
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'],
    width=0.7,
    capsize=0.2,
    err_kws={'linewidth': 0.5, 'color': 'black'},
    ax = ax,
    zorder = 1
)
sns.stripplot(
    x = 'Recovery Function',
    y = 'Recovery Loss',
    data=Data,
    hue='Spatial Map',
    palette=['#D4C9A8', '#8E9F85', '#527C5A', '#C3AED6', '#66C7B4', '#A7D8DE'],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    jitter=0.2,
    zorder = 0
)
ax.set_ylim([0, 0.15])
ax.set_yticks(np.linspace(0, 0.15, 6))
plt.savefig(join(loc, 'RMSE [Recovery].png'), dpi = 600)
plt.savefig(join(loc, 'RMSE [Recovery].svg'), dpi = 600)
plt.close()

print("Fit Retention Prob.:")
func_names = np.unique(Data['Recovery Function'])
for i in range(len(func_names)-1):
    for j in range(i+1, len(func_names)):
        idx1 = np.where(Data['Recovery Function'] == func_names[i])[0]
        idx2 = np.where(Data['Recovery Function'] == func_names[j])[0]
        d = np.where(
            (np.isnan(Data['Recovery Loss'][idx1]) == False) &
            (np.isnan(Data['Recovery Loss'][idx2]) == False)
        )
        print(f"    {func_names[i]} vs {func_names[j]}: {ttest_rel(Data['Recovery Loss'][idx1[d]], Data['Recovery Loss'][idx2[d]])}")