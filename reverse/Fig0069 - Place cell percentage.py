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

Data['Percentage'] = Data['Percentage']*100
HPData['Percentage'] = HPData['Percentage']*100

print_estimator(Data['Percentage'][np.where(Data['Direction'] == 'Cis')[0]])
print_estimator(Data['Percentage'][np.where(Data['Direction'] == 'Trs')[0]])

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