from mylib.statistic_test import *

code_id = '0824 - Route Specific Place Fields and Allocentric Place Fields'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Segment', 'Proportion'],
                              f=f2, 
                              function = AllocentricFieldProportion_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
fig = plt.figure(figsize = (2, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
palette = sns.color_palette('rainbow', 7)[::-1][1:]
sns.barplot(
    x='Segment',
    y='Proportion',
    data = Data,
    ax = ax,
    palette=palette,
    width=0.8,
    capsize=0.3,
    errcolor='black',
    errwidth=0.5,
    zorder=2
)

sns.stripplot(
    x='Segment',
    y='Proportion',
    data = Data,
    ax = ax,
    palette=palette,
    edgecolor='black',
    linewidth=0.1,
    jitter=0.2,
    size=3,
    
    zorder=1
)

ax.set_yticks(np.linspace(0, 1, 6))
ax.set_ylim(0, 1)
ax.set_xlim(-1, 6)
plt.savefig(os.path.join(loc, 'Allocentric Field Proportion.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Allocentric Field Proportion.svg'), dpi=600)
plt.show()