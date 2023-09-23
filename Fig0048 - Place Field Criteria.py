from mylib.statistic_test import *

code_id = "0048 - Place Cell Criteria"
loc = join(figpath, code_id)
mkdir(loc)


if os.path.exists(os.path.join(figdata, code_id+' - Field Number.pkl')) == False:
    NumberData = DataFrameEstablish(variable_names = ['Field Number', 'criteria', 'x'], file_idx=np.arange(56,88),
                              f = f1, function = PlaceFieldNumberWithCriteria_Interface, 
                              file_name = code_id+' - Field Number', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+' - Field Number.pkl'), 'rb') as handle:
        NumberData = pickle.load(handle)

idx = np.where(NumberData['criteria'] == 'A')
SubNumberData = DivideData(NumberData, index=idx)
fig = plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette('rocket', 3)
sns.lineplot(
    x='x',
    y='Field Number',
    data=SubNumberData,
    hue='Maze Type',
    palette=colors
)
ax.legend(facecolor = None, edgecolor = None, fontsize=8)
ax.set_xlim([0, 1])
ax.set_xticks(np.linspace(0,1,6))
ax.set_xlabel('Fraction of Peak Rate')
ax.set_ylabel('Field Number')
plt.tight_layout()
plt.savefig(join(loc, '[Fraction] Field Number.png'), dpi=600)
plt.savefig(join(loc, '[Fraction] Field Number.svg'), dpi=600)
plt.close()


idx = np.where(NumberData['criteria'] == 'B')
SubNumberData = DivideData(NumberData, index=idx)
fig = plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette('rocket', 3)
sns.lineplot(
    x='x',
    y='Field Number',
    data=SubNumberData,
    hue='Maze Type',
    palette=colors
)
ax.legend(facecolor = None, edgecolor = None, fontsize=8)
ax.set_xlim([0, 5])
ax.set_xticks(np.linspace(0,5,6))
ax.set_xlabel('Threshold Rate / Hz')
ax.set_ylabel('Field Number')
plt.tight_layout()
plt.savefig(join(loc, '[Absolute threshold] Field Number.png'), dpi=600)
plt.savefig(join(loc, '[Absolute threshold] Field Number.svg'), dpi=600)
plt.close()