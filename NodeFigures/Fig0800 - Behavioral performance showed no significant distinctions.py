from mylib.statistic_test import *

code_id = '0800 - Behavioral performance showed no significant distinctions'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Route', 'Correct Rate', 'Pass Number', 'Error Number', 'Pure Guess Correct Rate'], f = f2, 
                              function = LearningCurveBehavioralScore_DSP_Interface, is_behav = True,
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

fig = plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
"""
box = sns.boxplot(
    x='Training Day',
    y='Correct Rate',
    hue='Route',
    data=Data,
    ax=ax,
    palette=DSPPalette,
    linecolor='black',
    linewidth=0.5,
    gap=0.2,
    flierprops={'marker': '.', 'color': 'k', 'markersize': 1}
)
for line in box.patches:
    line.set_linewidth(0)
"""
sns.stripplot(
    x='Training Day',
    y='Correct Rate',
    hue='Route',
    data=Data,
    hue_order=[0, 4, 1, 5, 2, 6, 3],
    ax=ax,
    palette=[DSPPalette[i] for i in [0, 4, 1, 5, 2, 6, 3]],
    size=3, 
    linewidth=0.3,
    alpha=0.8,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 1.03)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(os.path.join(loc, 'correct decision rate.png'), dpi=600)
plt.savefig(os.path.join(loc, 'correct decision rate.svg'), dpi=600)
plt.close()

# Perform One-way ANOVA
from scipy.stats import f_oneway
for d in np.unique(Data['Training Day']):
    
    grouplists = []
    for route in np.unique(Data['Route']):
        idx = np.where((Data['Training Day'] == d) & (Data['Route'] == route))[0]
        grouplists.append(Data['Correct Rate'][idx])
    
    F, p = f_oneway(*grouplists)
    print(f"{d} - F = {F}, p = {p}")

print_estimator(Data['Correct Rate'])