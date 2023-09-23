
from mylib.statistic_test import *

code_id = '0026'

if os.path.exists(os.path.join(figdata, '0016data.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Cell', 'SI'], f = f2, function = SpatialInformation_Interface, 
                              file_name = code_id, behavior_paradigm = 'SimpleMaze')   
else:
    with open(os.path.join(figdata, '0016data.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

if os.path.exists(os.path.join(figdata, '0017data.pkl')) == False:
    Data2 = DataFrameEstablish(variable_names = ['Cell', 'SI'], f = f1, function = SpatialInformation_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')   
else:
    with open(os.path.join(figdata, '0017data.pkl'), 'rb') as handle:
        Data2 = pickle.load(handle)

# Training Day >= 4 and total cells >= 30
sm_idx = np.concatenate([np.where((Data['Training Day'] == 'Day '+str(k)))[0] for k in range(4,10)])
cm_idx = np.concatenate([np.where((Data2['Training Day'] == 'Day '+str(k))&(Data2['Maze Type'] != 'Open Field'))[0] for k in range(4,10)])

data = {
    'SI':np.concatenate([Data['SI'][sm_idx], Data2['SI'][cm_idx]]),
    'Environment':np.concatenate([np.repeat('Dark', len(sm_idx)), np.repeat('Light', len(cm_idx))])
}

# ttest
_, pvalue = scipy.stats.ttest_ind(Data['SI'][sm_idx], Data2['SI'][cm_idx])

# Comparison of the SI of place cells when mice are exploring in dark and light environment.
fig = plt.figure(figsize = (2,4))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
dark_set = Data['SI'][sm_idx]*100
light_set = Data2['SI'][cm_idx]*100
sns.barplot(x = 'Environment', y = 'SI', data = data, width = 0.6, alpha = 0.8)
#sns.scatterplot(x = 'Environment', y = 'SI', data = data)
ax.set_yticks(np.linspace(0,1.5,7))
ax.set_ylabel('Spatial Information of Cells')
ax.axis([-0.5,1.5,0,1.5])
plt.xticks(ticks = [0,1], labels = ['Dark', 'Light'])
plot_star(ax = ax, left = [0.1], right = [0.9], height = [1.4], delt_h=[0.03], p = [pvalue], fontsize = 14, barwidth=3)
ax.set_xlabel("Environment")
plt.tight_layout()
plt.savefig(os.path.join(figpath, code_id+'Spatial Information Comparison-Simple&CrossMaze.png'), dpi = 600)
plt.savefig(os.path.join(figpath, code_id+'Spatial Information Comparison-Simple&CrossMaze.svg'), dpi = 600)
plt.show()