from mylib.statistic_test import *

code_id = '0066 - Reverse, Place cell overlap'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Cis Percentage', 'Trs Percentage', 'Overlap Percentage'], 
                              f = f3, function = PlacecellOverlap_Reverse_Interface, 
                              file_name = code_id, behavior_paradigm = 'ReverseMaze')
    
if os.path.exists(join(figdata, code_id+' [hairpin].pkl')):
    with open(join(figdata, code_id+' [hairpin].pkl'), 'rb') as handle:
        HPData = pickle.load(handle)
else:
    HPData = DataFrameEstablish(variable_names = ['Cis Percentage', 'Trs Percentage', 'Overlap Percentage'], 
                              f = f4, function = PlacecellOverlap_Reverse_Interface, 
                              file_name = code_id+' [hairpin]', behavior_paradigm = 'HairpinMaze')

Data['Overlap Percentage'] = Data['Overlap Percentage']*100
Data['Cis Percentage'] = Data['Cis Percentage']*100
Data['Trs Percentage'] = Data['Trs Percentage']*100

print_estimator(Data['Overlap Percentage'], end='\n\n')

HPData['Overlap Percentage'] = HPData['Overlap Percentage']*100
HPData['Cis Percentage'] = HPData['Cis Percentage']*100
HPData['Trs Percentage'] = HPData['Trs Percentage']*100
print_estimator(HPData['Overlap Percentage'])
print_estimator(HPData['Overlap Percentage'][np.where(HPData['Training Day'] == 'Day 1')[0]])
print_estimator(HPData['Overlap Percentage'][np.where(HPData['Training Day'] == 'Day 7')[0]])
print_estimator(HPData['Cis Percentage'][np.where(HPData['Training Day'] == 'Day 1')[0]])
print_estimator(HPData['Cis Percentage'][np.where(HPData['Training Day'] == 'Day 7')[0]])
print_estimator(HPData['Trs Percentage'][np.where(HPData['Training Day'] == 'Day 1')[0]])
print_estimator(HPData['Trs Percentage'][np.where(HPData['Training Day'] == 'Day 7')[0]])

print("Comparison of Reverse Maze and Hairpin Maze (Day 1)")
print(ttest_ind(Data['Overlap Percentage'][np.where(Data['Training Day'] == 'Day 1')[0]], 
                HPData['Overlap Percentage'][np.where(HPData['Training Day'] == 'Day 1')[0]]))
print(ttest_ind(Data['Cis Percentage'][np.where(Data['Training Day'] == 'Day 1')[0]], 
                HPData['Cis Percentage'][np.where(HPData['Training Day'] == 'Day 1')[0]]))
print(ttest_ind(Data['Trs Percentage'][np.where(Data['Training Day'] == 'Day 1')[0]], 
                HPData['Trs Percentage'][np.where(HPData['Training Day'] == 'Day 1')[0]]), end='\n\n')
print("Comparison of Reverse Maze and Hairpin Maze (Day 7)")
print(ttest_ind(Data['Overlap Percentage'][np.where(Data['Training Day'] == 'Day 7')[0]], 
                HPData['Overlap Percentage'][np.where(HPData['Training Day'] == 'Day 7')[0]]))
print(ttest_ind(Data['Cis Percentage'][np.where(Data['Training Day'] == 'Day 7')[0]], 
                HPData['Cis Percentage'][np.where(HPData['Training Day'] == 'Day 7')[0]]))
print(ttest_ind(Data['Trs Percentage'][np.where(Data['Training Day'] == 'Day 7')[0]], 
                HPData['Trs Percentage'][np.where(HPData['Training Day'] == 'Day 7')[0]]))

colors = [sns.color_palette('Blues', 9)[3], sns.color_palette('YlOrRd', 9)[3]]
fig = plt.figure(figsize = (4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x="Training Day",
    y="Cis Percentage",
    hue="Maze Type",
    data=Data,
    err_style='bars',
    palette=colors,
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
    y="Trs Percentage",
    hue="Maze Type",
    data=Data,
    err_style='bars',
    palette=colors[1:],
    ax=ax, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
ax.set_ylim(0, 100)
ax.set_yticks(np.linspace(0, 100, 6))
plt.savefig(os.path.join(loc, 'Place cell overlap.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Place cell overlap.svg'), dpi=600)
plt.close()


fig = plt.figure(figsize = (3,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x="Training Day",
    y="Cis Percentage",
    hue="Maze Type",
    data=HPData,
    err_style='bars',
    palette=colors,
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
    y="Trs Percentage",
    hue="Maze Type",
    data=HPData,
    err_style='bars',
    palette=colors[1:],
    ax=ax, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
ax.set_ylim(0, 100)
ax.set_yticks(np.linspace(0, 100, 6))
plt.savefig(os.path.join(loc, 'Place cell overlap Hairpin.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Place cell overlap Hairpin.svg'), dpi=600)
plt.close()