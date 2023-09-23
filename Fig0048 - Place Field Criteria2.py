from mylib.statistic_test import *

code_id = "0048 - Place Cell Criteria"
loc = join(figpath, code_id)
mkdir(loc)


if os.path.exists(os.path.join(figdata, code_id+' - FSC Stability.pkl')) == False:
    StabilityData = DataFrameEstablish(variable_names = ['FSC Stability', 'criteria', 'x'], file_idx=np.arange(56,88),
                              f = f1, function = PlaceFieldFSCStabilityWithCriteria_Interface, 
                              file_name = code_id+' - FSC Stability', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+' - FSC Stability.pkl'), 'rb') as handle:
        StabilityData = pickle.load(handle)


idx = np.where(StabilityData['criteria'] == 'A')
SubStabilityData = DivideData(StabilityData, index=idx)
fig = plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette('rocket', 3)
sns.lineplot(
    x='x',
    y='FSC Stability',
    data=SubStabilityData,
    hue='Maze Type',
    palette=colors
)
ax.legend(facecolor = None, edgecolor = None, fontsize=8)
ax.set_xlim([0, 1])
ax.set_xticks(np.linspace(0,1,6))
ax.set_xlabel('Fraction of Peak Rate')
ax.set_ylabel('Field Stability')
plt.savefig(join(loc, '[Fraction] Field Stability.png'), dpi=600)
plt.savefig(join(loc, '[Fraction] Field Stability.svg'), dpi=600)
plt.close()


idx = np.where(StabilityData['criteria'] == 'B')
SubStabilityData = DivideData(StabilityData, index=idx)
fig = plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette('rocket', 3)
sns.lineplot(
    x='x',
    y='FSC Stability',
    data=SubStabilityData,
    hue='Maze Type',
    palette=colors
)
ax.legend(facecolor = None, edgecolor = None, fontsize=8)
ax.set_xlim([0, 5])
ax.set_xticks(np.linspace(0,5,6))
ax.set_xlabel('Threshold Rate / Hz')
ax.set_ylabel('Field Stability')
plt.savefig(join(loc, '[Absolute threshold] Field Stability.png'), dpi=600)
plt.savefig(join(loc, '[Absolute threshold] Field Stability.svg'), dpi=600)
plt.close()