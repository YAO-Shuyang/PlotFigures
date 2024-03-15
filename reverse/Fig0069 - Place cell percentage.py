from mylib.statistic_test import *

code_id = '0069 - Place cell percentage - Reverse&Hairpin'
loc = os.path.join(figpath, code_id)
mkdir(loc)


if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Percentage', 'Direction'], 
                              f = f3, function = PlacecellPercentage_Reverse_Interface, 
                              file_name = code_id, behavior_paradigm = 'ReverseMaze')
    
if os.path.exists(join(figdata, code_id+' [hairpin].pkl')):
    with open(join(figdata, code_id+' [hairpin].pkl'), 'rb') as handle:
        HPData = pickle.load(handle)
else:
    HPData = DataFrameEstablish(variable_names = ['Percentage', 'Direction'], 
                              f = f4, function = PlacecellPercentage_Reverse_Interface, 
                              file_name = code_id + ' [hairpin]', behavior_paradigm = 'HairpinMaze')
    
if os.path.exists(join(figdata, code_id+' [overlap MA].pkl')):
    with open(join(figdata, code_id+' [overlap MA].pkl'), 'rb') as handle:
        Overlap = pickle.load(handle)
else:
    Overlap = DataFrameEstablish(variable_names = ['Percentage'], 
                              f = f3, function = PlacecellPercentageOverlap_Reverse_Interface, 
                              file_name = code_id+' [overlap MA]', behavior_paradigm = 'ReverseMaze')    
    
if os.path.exists(join(figdata, code_id+' [overlap HP].pkl')):
    with open(join(figdata, code_id+' [overlap HP].pkl'), 'rb') as handle:
        HPOverlap = pickle.load(handle)
else:
    HPOverlap = DataFrameEstablish(variable_names = ['Percentage'], 
                              f = f4, function = PlacecellPercentageOverlap_Reverse_Interface, 
                              file_name = code_id + ' [overlap HP]', behavior_paradigm = 'HairpinMaze')
    
Data['Percentage'] = Data['Percentage']*100
HPData['Percentage'] = HPData['Percentage']*100

print_estimator(Data['Percentage'][np.where(Data['Direction'] == 'cis')[0]])
print_estimator(Data['Percentage'][np.where(Data['Direction'] == 'trs')[0]])
print_estimator(HPData['Percentage'][np.where(HPData['Direction'] == 'cis')[0]])
print_estimator(HPData['Percentage'][np.where(HPData['Direction'] == 'trs')[0]])


print_estimator(Overlap['Percentage'])
print_estimator(HPOverlap['Percentage'])

colors = [sns.color_palette('Blues', 9)[5], sns.color_palette('YlOrRd', 9)[5], sns.color_palette('Blues', 9)[7], sns.color_palette('YlOrRd', 9)[7]]
fig = plt.figure(figsize = (4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x="Training Day",
    y="Percentage",
    hue="Direction",
    data=HPData,
    err_style='bars',
    palette=colors[:2],
    ax=ax, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
sns.lineplot(
    x="Training Day",
    y="Percentage",
    hue="Direction",
    data=Data,
    err_style='bars',
    palette=colors[2:],
    ax=ax, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
ax.set_ylim(0,100)
ax.set_yticks(np.linspace(0, 100, 6))
plt.show()