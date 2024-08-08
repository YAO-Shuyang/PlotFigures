from mylib.statistic_test import *

code_id = '0064 - Place Field Num - Reverse&Hairpin'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Field Number', 'Direction'], 
                              f = f3, function = PlaceFieldNum_Reverse_Interface, 
                              file_name = code_id, behavior_paradigm = 'ReverseMaze')
    
if os.path.exists(join(figdata, code_id+' [Hairpin].pkl')):
    with open(join(figdata, code_id+' [Hairpin].pkl'), 'rb') as handle:
        HPData = pickle.load(handle)
else:
    HPData = DataFrameEstablish(variable_names = ['Field Number', 'Direction'], 
                              f = f4, function = PlaceFieldNum_Reverse_Interface, 
                              file_name = code_id + ' [Hairpin]', behavior_paradigm = 'HairpinMaze')
    
print_estimator(Data['Field Number'][np.where(Data['Direction'] == 'Cis')[0]])
print_estimator(Data['Field Number'][np.where(Data['Direction'] == 'Trs')[0]])
print(ttest_rel(Data['Field Number'][np.where(Data['Direction'] == 'Cis')[0]], Data['Field Number'][np.where(Data['Direction'] == 'Trs')[0]]))
print_estimator(HPData['Field Number'][np.where(HPData['Direction'] == 'Cis')[0]])
print_estimator(HPData['Field Number'][np.where(HPData['Direction'] == 'Trs')[0]])
print(ttest_rel(HPData['Field Number'][np.where(HPData['Direction'] == 'Cis')[0]], HPData['Field Number'][np.where(HPData['Direction'] == 'Trs')[0]]))

colors = [sns.color_palette('Blues', 9)[3], sns.color_palette('YlOrRd', 9)[3]]
fig = plt.figure(figsize = (4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x="Training Day",
    y="Field Number",
    hue="Direction",
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
ax.set_ylim(0, 15)
ax.set_yticks(np.linspace(0, 15, 6))
plt.savefig(os.path.join(loc, 'Field Number.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Field Number.svg'), dpi=600)
plt.close()


fig = plt.figure(figsize = (3,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x="Training Day",
    y="Field Number",
    hue="Direction",
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
ax.set_ylim(0, 15)
ax.set_yticks(np.linspace(0, 15, 6))
plt.savefig(os.path.join(loc, 'Field Number Hairpin.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Field Number Hairpin.svg'), dpi=600)
plt.close()


fig = plt.figure(figsize=(2,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x='Direction',
    y='Field Number',
    data=Data,
    palette=['#003366', '#0099CC'],
    ax=ax,
    errcolor='black',
    errwidth=0.5,
    capsize=0.3,
    width=0.8
)

sns.stripplot(
    x='Direction',
    y='Field Number',
    hue='MiceID',
    data=Data,
    palette=['#F2E8D4', '#8E9F85', '#C3AED6', '#A7D8DE'],
    edgecolor='black',
    size=4,
    linewidth=0.15,
    ax = ax,
    jitter=0.15
)
ax.set_ylim(0, 15)
ax.set_yticks(np.linspace(0, 15, 6))
plt.savefig(join(loc, 'Comparison of Direction [Reverse].png'), dpi=2400)
plt.savefig(join(loc, 'Comparison of Direction [Reverse].svg'), dpi=2400)
plt.close()

fig = plt.figure(figsize=(2,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x='Direction',
    y='Field Number',
    data=HPData,
    palette=['#003366', '#0099CC'],
    ax=ax,
    errcolor='black',
    errwidth=0.5,
    capsize=0.3,
    width=0.8
)

sns.stripplot(
    x='Direction',
    y='Field Number',
    hue='MiceID',
    data=HPData,
    palette=['#F2E8D4', '#8E9F85', '#C3AED6', '#A7D8DE'],
    edgecolor='black',
    size=4,
    linewidth=0.15,
    ax = ax,
    jitter=0.15
)
ax.set_ylim(0, 15)
ax.set_yticks(np.linspace(0, 15, 6))
plt.savefig(join(loc, 'Comparison of Direction [Hairpin].png'), dpi=2400)
plt.savefig(join(loc, 'Comparison of Direction [Hairpin].svg'), dpi=2400)
plt.close()