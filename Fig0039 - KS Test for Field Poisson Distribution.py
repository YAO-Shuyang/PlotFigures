from mylib.statistic_test import *
from scipy.stats import kstest

code_id = "0039 - KS test for Poisson Distribution"
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Statistic', 'PValue'], f = f1, 
                              function = KSTestPoisson_Interface, func_kwgs = {'is_placecell': True},
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
fig, axes = plt.subplots(ncols=2, nrows=1, figsize = (8,3))
ax1, ax2 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True), Clear_Axes(axes[1], close_spines=['top', 'right', 'left'],ifxticks=True)
colors = sns.color_palette("rocket", 3)
uniq_day = ['Day 1', 'Day 2', 'Day 3', 'Day 4',
            'Day 5', 'Day 6', 'Day 7', 'Day 8',
            'Day 9', '>=Day 10']
idx = np.concatenate([np.where(Data['Training Day'] == day)[0] for day in uniq_day])
Data = SubDict(Data, Data.keys(), idx)

stage_indices = np.where(Data['Stage'] == 'Stage 1')[0]
SubData = SubDict(Data, Data.keys(), idx=stage_indices)
sns.lineplot(
    x='Training Day',
    y='PValue',
    data=SubData,
    hue='Maze Type',
    ax=ax1,
    legend=False,
    palette=colors
)

stage_indices = np.where(Data['Stage'] == 'Stage 2')[0]
SubData = SubDict(Data, Data.keys(), idx=stage_indices)
sns.lineplot(
    x='Training Day',
    y='PValue',
    data=SubData,
    hue='Maze Type',
    ax=ax2,
    palette=colors
)
ax1.semilogy()
ax2.semilogy()
ax1.set_ylim([0.00001, 1])
ax2.set_ylim([0.00001, 1])
ax1.axhline(0.05, color='black', linestyle='--', linewidth=0.8)
ax1.axhline(0.01, color='black', linestyle='--', linewidth=0.8)
ax1.axhline(0.001, color='black', linestyle='--', linewidth=0.8)
ax1.axhline(0.0001, color='black', linestyle='--', linewidth=0.8)
ax2.axhline(0.05, color='black', linestyle='--', linewidth=0.8)
ax2.axhline(0.01, color='black', linestyle='--', linewidth=0.8)
ax2.axhline(0.001, color='black', linestyle='--', linewidth=0.8)
ax2.axhline(0.0001, color='black', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig(os.path.join(loc, "PValue-KSTest"+'.png'), dpi = 2400)
plt.savefig(os.path.join(loc, "PValue-KSTest"+'.svg'), dpi = 2400)
plt.close()