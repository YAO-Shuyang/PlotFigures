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
print("Reversed Maze Overlaped percentage of Place cells")
print_estimator(Data['Overlap Percentage'], end='\n\n')

HPData['Overlap Percentage'] = HPData['Overlap Percentage']*100
HPData['Cis Percentage'] = HPData['Cis Percentage']*100
HPData['Trs Percentage'] = HPData['Trs Percentage']*100
print("Reversed Maze Overlaped percentage of Place cells")
print_estimator(HPData['Cis Percentage'])
print_estimator(HPData['Trs Percentage'])
print_estimator(HPData['Overlap Percentage'])

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


fig = plt.figure(figsize=(1.5,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x=np.concatenate([np.repeat("Cis", Data['Cis Percentage'].shape[0]), np.repeat("Trs", Data['Trs Percentage'].shape[0])]),
    y=np.concatenate([Data['Cis Percentage'], Data['Trs Percentage']]),
    palette=['#003366', '#0099CC'],
    ax=ax,
    errcolor='black',
    errwidth=0.5,
    capsize=0.3,
    width=0.8
)

sns.stripplot(
    x=np.concatenate([np.repeat("Cis", Data['Cis Percentage'].shape[0]), np.repeat("Trs", Data['Trs Percentage'].shape[0])]),
    y=np.concatenate([Data['Cis Percentage'], Data['Trs Percentage']]),
    hue=np.concatenate([Data['MiceID'], Data['MiceID']]),
    palette=['#F2E8D4', '#8E9F85', '#C3AED6', '#A7D8DE'],
    edgecolor='black',
    size=4,
    linewidth=0.15,
    ax = ax,
    jitter=0.15
)
ax.set_ylim([0, 100])
ax.set_yticks(np.linspace(0, 100, 6))
plt.savefig(join(loc, 'Comparison of Direction [Reverse].png'), dpi=2400)
plt.savefig(join(loc, 'Comparison of Direction [Reverse].svg'), dpi=2400)
plt.close()

print("Cross Direction:", ttest_rel(Data['Cis Percentage'], Data['Trs Percentage']))
print("Cross Direction [HP]:", ttest_rel(HPData['Cis Percentage'], HPData['Trs Percentage']))

fig = plt.figure(figsize=(1.5,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x=np.concatenate([np.repeat("Cis", HPData['Cis Percentage'].shape[0]), np.repeat("Trs", HPData['Trs Percentage'].shape[0])]),
    y=np.concatenate([HPData['Cis Percentage'], HPData['Trs Percentage']]),
    palette=['#003366', '#0099CC'],
    ax=ax,
    errcolor='black',
    errwidth=0.5,
    capsize=0.3,
    width=0.8
)
sns.stripplot(
    x=np.concatenate([np.repeat("Cis", HPData['Cis Percentage'].shape[0]), np.repeat("Trs", HPData['Trs Percentage'].shape[0])]),
    y=np.concatenate([HPData['Cis Percentage'], HPData['Trs Percentage']]),
    hue=np.concatenate([HPData['MiceID'], HPData['MiceID']]),
    palette=['#F2E8D4', '#8E9F85', '#C3AED6', '#A7D8DE'],
    edgecolor='black',
    size=4,
    linewidth=0.15,
    ax = ax,
    jitter=0.15
)
ax.set_ylim([0, 100])
ax.set_yticks(np.linspace(0, 100, 6))
plt.savefig(join(loc, 'Comparison of Direction [Hairpin].png'), dpi=2400)
plt.savefig(join(loc, 'Comparison of Direction [Hairpin].svg'), dpi=2400)
plt.close()


fig = plt.figure(figsize=(2,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x=np.concatenate([np.repeat("Cis", Data['Cis Percentage'].shape[0]), np.repeat("Trs", Data['Trs Percentage'].shape[0])]),
    y=np.concatenate([Data['Cis Percentage'], Data['Trs Percentage']]),
    palette=['#003366', '#0099CC'],
    ax=ax,
    errcolor='black',
    errwidth=0.5,
    capsize=0.3,
    width=0.8
)

sns.stripplot(
    x=np.concatenate([np.repeat("Cis", Data['Cis Percentage'].shape[0]), np.repeat("Trs", Data['Trs Percentage'].shape[0])]),
    y=np.concatenate([Data['Cis Percentage'], Data['Trs Percentage']]),
    hue=np.concatenate([Data['MiceID'], Data['MiceID']]),
    palette=['#F2E8D4', '#8E9F85', '#C3AED6', '#A7D8DE'],
    edgecolor='black',
    size=4,
    linewidth=0.15,
    ax = ax,
    jitter=0.15
)
ax.set_ylim([0, 100])
ax.set_yticks(np.linspace(0, 100, 6))
plt.savefig(join(loc, 'Comparison of Direction [Reverse].png'), dpi=2400)
plt.savefig(join(loc, 'Comparison of Direction [Reverse].svg'), dpi=2400)
plt.close()

print("Cross Direction:", ttest_rel(Data['Cis Percentage'], Data['Trs Percentage']))
print("Cross Direction [HP]:", ttest_rel(HPData['Cis Percentage'], HPData['Trs Percentage']))

fig = plt.figure(figsize=(1.5,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
d1_idx, d7_idx = np.where(HPData['Training Day'] == 'Day 1')[0], np.where(HPData['Training Day'] == 'Day 7')[0]
SubData = {
    "Percentage": np.concatenate([HPData['Cis Percentage'][d1_idx], HPData['Trs Percentage'][d1_idx], 
                                  HPData['Cis Percentage'][d7_idx], HPData['Trs Percentage'][d7_idx]]),
    "MiceID": np.concatenate([HPData['MiceID'][d1_idx], HPData['MiceID'][d1_idx], 
                              HPData['MiceID'][d7_idx], HPData['MiceID'][d7_idx]]),
    "Direction": np.concatenate([np.repeat("Cis", d1_idx.shape[0]), np.repeat("Trs", d1_idx.shape[0]), 
                                  np.repeat("Cis", d7_idx.shape[0]), np.repeat("Trs", d7_idx.shape[0])]),
    "Training Day": np.concatenate([np.repeat("Day 1", d1_idx.shape[0]), np.repeat("Day 1", d1_idx.shape[0]), 
                                    np.repeat("Day 7", d7_idx.shape[0]), np.repeat("Day 7", d7_idx.shape[0])])
}
print(SubData, end='\n\n')
print("HP Day 1 vs Day 7:")
print_estimator(HPData['Cis Percentage'][d1_idx]),
print_estimator(HPData['Trs Percentage'][d1_idx]),
print_estimator(HPData['Cis Percentage'][d7_idx]),
print_estimator(HPData['Trs Percentage'][d7_idx]),
print("  cis: ", ttest_rel(HPData['Cis Percentage'][d1_idx], HPData['Cis Percentage'][d7_idx]))
print("  trs: ", ttest_rel(HPData['Trs Percentage'][d1_idx], HPData['Trs Percentage'][d7_idx]))
sns.barplot(
    x='Direction',
    y='Percentage',
    hue = 'Training Day',
    data=SubData,
    palette=['#003366', '#0099CC'],
    ax=ax,
    errcolor='black',
    errwidth=0.5,
    capsize=0.3,
    width=0.8
)
sns.stripplot(
    x='Direction',
    y='Percentage',
    hue='MiceID',
    data=SubData,
    palette=['#F2E8D4', '#8E9F85', '#C3AED6', '#A7D8DE'],
    edgecolor='black',
    size=4,
    linewidth=0.15,
    dodge=True,
    ax = ax,
    jitter=0.1
)
ax.set_ylim([0, 100])
ax.set_yticks(np.linspace(0, 100, 6))
plt.savefig(join(loc, 'Comparison of Days [Hairpin].png'), dpi=2400)
plt.savefig(join(loc, 'Comparison of Days [Hairpin].svg'), dpi=2400)
plt.close()