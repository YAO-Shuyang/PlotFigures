from mylib.statistic_test import *

code_id = '0825 - Field Turn-on and off With Segment'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Segment', 'Proportion', 'Category'],
                              f=f2, 
                              function = FieldStateSwitchWithSegment_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

file_names = [
    'Formation - 1', 'Disappearance - 1', 'Retention - 1', # 0 vs 4
    'Formation - 2', 'Disappearance - 2', 'Retention - 2', # 5 vs 9
    'Formation - 3', 'Disappearance - 3', 'Retention - 3'  # 4 vs 5
]
for i in range(9):
    idx = np.where((Data['Category'] == i))[0]
    SubData = SubDict(Data, Data.keys(), idx)

    fig = plt.figure(figsize = (2, 3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    palette = sns.color_palette('rainbow', 7)[::-1]
    sns.barplot(
        x='Segment',
        y='Proportion',
        data = SubData,
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
        data = SubData,
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
    ax.set_xlim(-1, 7)
    plt.savefig(os.path.join(loc, file_names[i]+'.svg'), dpi=600)
    plt.savefig(os.path.join(loc, file_names[i]+'.png'), dpi=600)
    plt.close()