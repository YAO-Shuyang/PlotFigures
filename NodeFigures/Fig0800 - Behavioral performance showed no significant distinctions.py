from mylib.statistic_test import *

code_id = '0800 - Behavioral performance showed no significant distinctions'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

with open(f2_behav['Trace Behav File'][0], 'rb') as handle:
    trace = pickle.load(handle)

print(trace.keys())
print(trace['behav_position_original'].shape, trace['processed_pos'].shape, trace['processed_pos_new'].shape, trace['correct_pos'].shape)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Route', 'Correct Rate', 'Pass Number', 'Error Number', 'Pure Guess Correct Rate'], f = f2_behav, 
                              function = LearningCurveBehavioralScore_DSP_Interface, is_behav = True,
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
fig = plt.figure(figsize=(6, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x='Training Day',
    y='Correct Rate',
    hue='Route',
    data=Data,
    ax=ax,
    palette=DSPPalette,
    capsize=0.05,
    errwidth=0.3,
    errcolor='k',
    edgecolor = 'k',
    linewidth = 0.3,
    zorder=2
)
sns.stripplot(
    x='Training Day',
    y='Correct Rate',
    hue='Route',
    data=Data,
    ax=ax,
    palette=DSPPalette,
    dodge=True,
    size=3,
    linewidth=0.2,
    jitter=0.1,
    edgecolor='k',
    zorder=1
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