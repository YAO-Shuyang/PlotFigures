from mylib.statistic_test import *

code_id = '0807 - Running Speed'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:

    Data = DataFrameEstablish(variable_names = ['Route', 'Lap', 'Position', 'Speed'], f = f2, 
                              function = RunningSpeed_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

hue = np.array([f"{Data['Lap'][i]}-{Data['MiceID'][i]}-{Data['Training Day'][i]}" for i in range(len(Data['Lap']))])
Data['hue'] = hue

fig = plt.figure(figsize=(3,2)),
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
"""
for i in range(7):
    SubData = SubDict(Data, Data.keys(), np.where(Data['Route'] == i)[0])
    sns.lineplot(
        x = 'Position',
        y = 'Speed',
        data = SubData,
        hue = 'hue',
        palette=[DSPPalette[i]],
        linewidth = 0.1,
        ax = ax,
        err_kws={'edgecolor':None}
    )
"""

sns.lineplot(
    x = 'Position',
    y = 'Speed',
    data = Data,
    hue = 'Route',
    palette=DSPPalette,
    linewidth=0.5,
    ax = ax,
    err_kws={'edgecolor':None}
)

ax.set_ylim(0, 80),
ax.set_yticks(np.linspace(0, 80, 9))
ax.set_xlim(0, 112.5)
ax.set_xticks(np.linspace(0, 112.5, 10))
plt.savefig(join(loc, 'speed.svg'), dpi = 600)
plt.savefig(join(loc, 'speed.png'), dpi = 600)
plt.show()