from mylib.statistic_test import *

code_id = '0830 - Starting Cell Encoded Route Number Distribution'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+' .pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Number Of Routes', 'Proportion'],
                              f=f2, file_idx=np.where(f2['MiceID'] != 10209)[0],
                              function = StartingCellEncodedRouteNumberDistribution_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

idx = np.where(Data['Number Of Routes'] == 7)[0]
print_estimator(Data['Proportion'][idx])

idx = np.where(Data['Number Of Routes'] != 7)[0]
d = np.reshape(Data['Proportion'][idx], (28, 5))
print(d)
d = np.sum(d, axis=1)

print_estimator(d)

fig = plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
order = [0, 4, 1, 5, 2, 6, 3]
sns.barplot(
    x="Number Of Routes",
    y="Proportion",
    data=Data,
    ax=ax,
    hue="Number Of Routes",
    palette="rainbow",
    width=0.8,
    capsize=0.3,
    err_kws={"linewidth": 0.5, 'color':'k'},
    edgecolor = None,
    zorder=2
)
sns.stripplot(
    x="Number Of Routes",
    y="Proportion",
    data=Data,
    hue = 'MiceID',
    ax=ax,
    palette=['#D4C9A8', '#8E9F85', '#C3AED6', '#FED7D7'],
    zorder=1,
    jitter=0.2,
    size=5,
    linewidth=0.2
)
ax.set_ylim(0, 0.5)
ax.set_yticks(np.linspace(0, 0.5, 6))

plt.savefig(join(loc, 'Starting Cell Encoded Route Number Distribution.svg'))
plt.savefig(join(loc, 'Starting Cell Encoded Route Number Distribution.png'), dpi=600)
plt.show()