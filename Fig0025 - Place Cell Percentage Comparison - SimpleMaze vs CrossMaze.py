
from mylib.statistic_test import *

code_id = '0025'

if os.path.exists(os.path.join(figdata, code_id+'data.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['percentage', 'place cell'], f = f2, function = PlaceCellPercentage_Interface, 
                              file_name = code_id, behavior_paradigm = 'SimpleMaze')   
else:
    with open(os.path.join(figdata, code_id+'data.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

if os.path.exists(os.path.join(figdata, '0023data.pkl')) == False:
    Data2 = DataFrameEstablish(variable_names = ['percentage', 'place cell'], f = f1, function = PlaceCellPercentage_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')   
else:
    with open(os.path.join(figdata, '0023data.pkl'), 'rb') as handle:
        Data2 = pickle.load(handle)

# Training Day >= 4 and total cells >= 30
sm_idx = np.concatenate([np.where((Data['Training Day'] == 'Day '+str(k))&(Data['place cell'] >= 10))[0] for k in range(4,10)])
cm_idx = np.concatenate([np.where((Data2['Training Day'] == 'Day '+str(k))&(Data2['place cell'] >= 10)&(Data2['Maze Type'] != 'Open Field'))[0] for k in range(4,10)])

data = {
    'percentage':np.concatenate([Data['percentage'][sm_idx]*100, Data2['percentage'][cm_idx]*100]),
    'Environment':np.concatenate([np.repeat('Dark', len(sm_idx)), np.repeat('Light', len(cm_idx))])
}

# ttest
_, pvalue = scipy.stats.ttest_ind(Data['percentage'][sm_idx]*100, Data2['percentage'][cm_idx]*100)

# Comparison of the percentage of place cells when mice are exploring in dark and light environment.
fig = plt.figure(figsize = (2,4))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
dark_set = Data['percentage'][sm_idx]*100
light_set = Data2['percentage'][cm_idx]*100
sns.barplot(x = 'Environment', y = 'percentage', data = data, width = 0.6, alpha = 0.8)
sns.scatterplot(x = 'Environment', y = 'percentage', data = data)
ax.set_yticks(np.linspace(0,100,6))
ax.set_ylabel('Percentage of Place Cells / %')
ax.axis([-0.5,1.5,0,100])
plt.xticks(ticks = [0,1], labels = ['Dark', 'Light'])
plot_star(ax = ax, left = [0.1], right = [0.9], height = [95], p = [pvalue], fontsize = 14, barwidth=3)
ax.set_xlabel("Environment")
plt.tight_layout()
plt.savefig(os.path.join(figpath, code_id+'Place Cell Percentage Comparison-Simple&CrossMaze.png'), dpi = 600)
plt.savefig(os.path.join(figpath, code_id+'Place Cell Percentage Comparison-Simple&CrossMaze.svg'), dpi = 600)
plt.close()
