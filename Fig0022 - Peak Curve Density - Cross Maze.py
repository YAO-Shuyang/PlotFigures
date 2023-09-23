# Fig0022-1, Barplot of Peak Curve Density

from mylib.statistic_test import *

code_id = '0022'

if os.path.exists(os.path.join(figdata, code_id+'data.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['MAE', 'RMSE', 'data_type'], f = f1, function = PeakDistributionDensity_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'data.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

def SignificanceTest(ax = None, data = None, test_item = 'RMSE', 
                     training_day = np.array(['Pre 2', 'Day 1', 'Day 2', 'Day 4', 'Day 5', 'Day 6','Day 7', 'Day 8', 'Day 9']), 
                     height = 24, **kwargs):

    for d in range(len(training_day)):
        test_idx = np.where((data['Training Day'] == training_day[d])&(data['Data Type'] == 'Experiment value'))[0]
        shuffle_idx = np.where((data['Training Day'] == training_day[d])&(data['Data Type'] == 'Chance Level'))[0]

        test_nan = np.where(np.isnan(data[test_item][test_idx]))[0]
        shuffle_nan = np.where(np.isnan(data[test_item][shuffle_idx]))[0]

        items1 = np.delete(data[test_item][test_idx], test_nan)
        items2 = np.delete(data[test_item][shuffle_idx], shuffle_nan)

        
        plt.figure(figsize = (8,6))
        plt.hist(items2, bins = 100)
        plt.show()
        _, p_value = scipy.stats.ttest_ind(items1, items2)

        ax.text(d, height[d], star(p_value), ha = 'center', **kwargs)

    return ax


maze_type = 'Maze'
idx = np.where((Data['Maze Type'] == 'Maze 1')|(Data['Maze Type'] == 'Maze 2'))[0]
data = {'Training Day':Data['Training Day'][idx],'MiceID':Data['MiceID'][idx],'Maze Type':Data['Maze Type'][idx],'Data Type':Data['data_type'][idx],
        'MAE':Data['MAE'][idx],'RMSE':Data['RMSE'][idx]}
idx = np.where(((Data['Maze Type'] == 'Maze 1')|(Data['Maze Type'] == 'Maze 2'))&(Data['data_type'] == 'Experiment value'))[0]
scatters = {'Training Day':Data['Training Day'][idx],'MiceID':Data['MiceID'][idx],'Maze Type':Data['Maze Type'][idx],'Data Type':Data['data_type'][idx],
        'MAE':Data['MAE'][idx],'RMSE':Data['RMSE'][idx]}
fig = plt.figure(figsize=(8,6))
ax = Clear_Axes(axes = plt.axes(), close_spines = ['top','right'], ifyticks = True)
sns.lineplot(x = 'Training Day', y = 'RMSE', data = data, hue = 'Data Type', ax = ax, errorbar=('ci', 95))
sns.scatterplot(data=scatters,x='Training Day',y='RMSE', ax = ax)
ax.legend(facecolor = 'white', edgecolor = 'white', title = 'Data Type')
ax.axis([-0.5, 9.5, 0, 0.02])
ax.set_yticks(np.linspace(0,0.02,6))
ax.set_xlabel("Training Day")
ax.set_title("Maze 1 & 2")
SignificanceTest(data = data, ax = ax, training_day = ['Pre 2', 'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6','Day 7', 'Day 8', 'Day 9'], 
                 height = [0.016, 0.016, 0.014, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012])
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9], labels = ['Pre 2', 'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6','Day 7', 'Day 8', 'Day 9'])
plt.savefig(os.path.join(figpath, code_id+'-1-Peak Curve Regression-CrossMaze-RMSE-'+maze_type+'.png'), dpi = 600)
plt.savefig(os.path.join(figpath, code_id+'-1-Peak Curve Regression-CrossMaze-RMSE-'+maze_type+'.svg'), dpi = 600)
plt.close()


maze_type = 'Open Field'
idx = np.where((Data['Maze Type'] == maze_type))[0]
data = {'Training Day':Data['Training Day'][idx],'MiceID':Data['MiceID'][idx],'Maze Type':Data['Maze Type'][idx],'Data Type':Data['data_type'][idx],
        'MAE':Data['MAE'][idx],'RMSE':Data['RMSE'][idx]}
idx = np.where((Data['Maze Type'] == maze_type)&(Data['data_type'] == 'Experiment value'))[0]
scatters = {'Training Day':Data['Training Day'][idx],'MiceID':Data['MiceID'][idx],'Maze Type':Data['Maze Type'][idx],'Data Type':Data['data_type'][idx],
        'MAE':Data['MAE'][idx],'RMSE':Data['RMSE'][idx]}
fig = plt.figure(figsize=(8,6))
ax = Clear_Axes(axes = plt.axes(), close_spines = ['top','right'], ifyticks = True)
sns.lineplot(x = 'Training Day', y = 'RMSE', data = data, hue = 'Data Type', ax = ax, errorbar=('ci', 95))
sns.scatterplot(data=scatters,x='Training Day',y='RMSE', ax = ax)
ax.legend(facecolor = 'white', edgecolor = 'white', title = 'Data Type')
ax.axis([-0.5, 10.5, 0, 0.02])
ax.set_yticks(np.linspace(0,0.02,6))
ax.set_xlabel("Training Day")
ax.set_title("Open Field")
SignificanceTest(data = data, ax = ax, training_day = ['Pre 1','Pre 2', 'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6','Day 7', 'Day 8', 'Day 9'], 
                 height = [0.018, 0.018, 0.018, 0.016, 0.016, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014])
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9,10], labels = ['Pre 1','Pre 2', 'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6','Day 7', 'Day 8', 'Day 9'])
plt.savefig(os.path.join(figpath, code_id+'-2-Peak Curve Regression-CrossMaze-RMSE-'+maze_type+'.png'), dpi = 600)
plt.savefig(os.path.join(figpath, code_id+'-2-Peak Curve Regression-CrossMaze-RMSE-'+maze_type+'.svg'), dpi = 600)
plt.close()