# Fig0021-1, Barplot of field number
# Fig0021-2, Lineplot of field number

from mylib.statistic_test import *

code_id = '0021 - Field Number Decline'
loc = join(figpath, code_id)
mkdir(loc)

idx = np.where((f1['date'] >= 20220813)&(f1['date'] != 20220814)&(f1['MiceID'] != 11094))[0]
if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Cell','field_number','maze_type'], f = f1, function = PlaceFieldNumber_Interface, 
                              func_kwgs={'is_placecell':True}, file_idx=idx,
                              file_name = code_id, behavior_paradigm = 'CrossMaze')   
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

cell_number = Data['field_number'].shape[0]
fs = 14

def SignificanceTest(ax = None, data = None, basic_height = None, delt_h = 0.1, width = 0.8):
    training_day = ['Pre 1', 'Pre 2', 'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9']
    hue = ['Open Field 1','Maze 1','Maze 2','Open Field 2']

    height = np.array([[0,0,0.1,0.3],
                       [0,0,0,  0.2],
                       [0,0,0, 0]])

    for d in range(len(training_day)):
        for i in range(len(hue)-1):
            for j in range(i+1,len(hue)):
                idx1 = np.where((data['Training Day'] == training_day[d])&(data['maze_type'] == hue[i]))[0]
                idx2 = np.where((data['Training Day'] == training_day[d])&(data['maze_type'] == hue[j]))[0]
                if len(idx1) == 0 or len(idx2) == 0:
                    # lack of data, continue
                    continue
                # 2-side t test
                _, pvalue = scipy.stats.ttest_ind(data['field_number'][idx1], data['field_number'][idx2], equal_var=True)
                plot_star(ax = ax, left = [d + (i-2)*0.2 + 0.15], right = [d + (j-2)*0.2 + 0.05], height = [basic_height[d] * (1 + height[i,j])], delt_h = [0.05], 
                          p = [pvalue])

plt.figure(figsize=(8,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette("rocket", 3)
sns.barplot(x='Training Day', y='field_number', data = Data, hue = 'Maze Type', palette=colors, 
            errcolor='black', errwidth=1, capsize=0.1)
ax.set_yticks(np.linspace(0,7,8))
plt.savefig(join(loc, 'Field number decline.png'), dpi = 2400)
plt.savefig(join(loc, 'Field number decline.svg'), dpi = 2400)
plt.close()

# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html#matplotlib.axes.Axes.errorbar
plt.figure(figsize=(8,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette("rocket", 3)
sns.lineplot(x='Training Day', y='field_number', data = Data, hue = 'Maze Type', palette=colors, 
             err_style='bars', err_kws={'elinewidth':1, 'capsize':0.5})
ax.set_yticks(np.linspace(0,7,8))
plt.savefig(join(loc, 'Field number decline [lineplot].png'), dpi = 2400)
plt.savefig(join(loc, 'Field number decline [lineplot].svg'), dpi = 2400)
plt.close()

for d in ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9']:
    op_idx = np.where((Data['Training Day'] == d)&(Data['Maze Type'] == 'Open Field'))[0]
    m1_idx = np.where((Data['Training Day'] == d)&(Data['Maze Type'] == 'Maze 1'))[0]
    m2_idx = np.where((Data['Training Day'] == d)&(Data['Maze Type'] == 'Maze 2'))[0]
    
    print(d, '----------------------------------------')
    print("OP: M1    ", ttest_ind(Data['field_number'][op_idx], Data['field_number'][m1_idx]))
    print("OP: M2    ", ttest_ind(Data['field_number'][op_idx], Data['field_number'][m2_idx]))
    print("M1: M2    ", ttest_ind(Data['field_number'][m1_idx], Data['field_number'][m2_idx]), end='\n\n\n')

# cross day significance test.
op_d1_idx = np.where((Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Open Field'))[0]
m1_d1_idx = np.where((Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 1'))[0]
m2_d1_idx = np.where((Data['Training Day'] == 'Day 1')&(Data['Maze Type'] == 'Maze 2'))[0]
op_d9_idx = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Open Field'))[0]
m1_d9_idx = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 1'))[0]
m2_d9_idx = np.where((Data['Training Day'] == 'Day 9')&(Data['Maze Type'] == 'Maze 2'))[0]

print("Day 1 vs Day 9")
print(ttest_ind(Data['field_number'][op_d1_idx], Data['field_number'][op_d9_idx]))
print(ttest_ind(Data['field_number'][m1_d1_idx], Data['field_number'][m1_d9_idx]))
print(ttest_ind(Data['field_number'][m2_d1_idx], Data['field_number'][m2_d9_idx]))

"""
# Field Number Bar plot -------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize = (16,4))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.barplot(x = 'Training Day',y = 'field_number', data = Data, hue = 'maze_type', ax = ax, hue_order = ['Open Field 1','Maze 1','Maze 2','Open Field 2'], width = 0.8)
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze Type', loc = 'upper left', ncol = 2)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Place Field Number', fontsize = fs)
ax.set_title('Field Number ('+str(cell_number)+' cells)', fontsize = fs)
ax.axis([-0.5,10.5,0,10])
SignificanceTest(ax = ax, data = Data, basic_height = [5, 6, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5], delt_h = 0.1)
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9,10], labels = ['Pre 1','Pre 2', 'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6','Day 7', 'Day 8', 'Day 9'])
plt.tight_layout()
plt.savefig(os.path.join(figpath,code_id+'-1-Field Number-BarPlot-CrossMaze.svg'), dpi = 600)
plt.savefig(os.path.join(figpath,code_id+'-1-Field Number-BarPlot-CrossMaze.png'), dpi = 600)
plt.close()

# Field Number Line plot -------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize = (4,3))
ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
sns.lineplot(x = 'Training Day',y = 'field_number', data = Data, hue = 'maze_type', ax = ax, hue_order = ['Open Field 1','Maze 1','Maze 2','Open Field 2'])
ax.legend(edgecolor = 'white', facecolor = 'white', title = 'Maze Type', loc = 'upper right', ncol = 2)
ax.set_xlabel('Training Day', fontsize = fs)
ax.set_ylabel('Place Field Number', fontsize = fs)
ax.set_title('Field Number ('+str(cell_number)+' cells)', fontsize = fs)
ax.axis([-0.5,10.5,0,12])
ax.set_yticks(np.linspace(0,12,7))
plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9,10], labels = ['P1','P2', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6','D7', 'D8', 'D9'])
plt.tight_layout()
plt.savefig(os.path.join(figpath,code_id+'-2-Field Number-LinePlot-CrossMaze.svg'), dpi = 600)
plt.savefig(os.path.join(figpath,code_id+'-2-Field Number-LinePlot-CrossMaze.png'), dpi = 600)
plt.close()


# field number distribution ----------------------------------------------------------------------------------------------------------------------------------------
mice = '11095'
training_day = ['Pre 1', 'Pre 2', 'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9']
fig, axes = plt.subplots(ncols = 6, nrows = 6, figsize = (4*6,3*6))
# open field
for r in [0,1]:
    for l in range(6):
        if r*6 + l == 11:
            axes[r,l] = Clear_Axes(axes[r,l])
        else:
            axes[r,l] = Clear_Axes(axes[r,l], close_spines = ['top', 'right'], ifyticks = True, ifxticks=True)
            idx = np.where((Data['Training Day'] == training_day[r*6+l])&(Data['Maze Type'] == 'Open Field')&(Data['MiceID'] == mice))[0]
            axes[r,l].hist(Data['field_number'][idx], bins = 15)
            axes[r,l].set_xticks(np.linspace(0,15,4))
            axes[r,l].set_ylabel("Cell Count, "+training_day[r*6+l])
            axes[r,l].set_xlabel("Field Number, Animal #"+mice+', Open Field')
# Maze 1
for r in [2,3]:
    for l in range(6):
        if r*6 + l - 12 == 11:
            axes[r,l] = Clear_Axes(axes[r,l])
        else:
            axes[r,l] = Clear_Axes(axes[r,l], close_spines = ['top', 'right'], ifyticks = True, ifxticks=True)
            idx = np.where((Data['Training Day'] == training_day[r*6+l-12])&(Data['Maze Type'] == 'Maze 1')&(Data['MiceID'] == mice))[0]
            axes[r,l].hist(Data['field_number'][idx], bins = 15)
            axes[r,l].set_xticks(np.linspace(0,15,4))
            axes[r,l].set_ylabel("Cell Count, "+training_day[r*6+l-12])
            axes[r,l].set_xlabel("Field Number, Animal #"+mice+', Maze 1')
# Maze 2
for r in [4,5]:
    for l in range(6):
        if r*6 + l - 24 == 11:
            axes[r,l] = Clear_Axes(axes[r,l])
        else:
            axes[r,l] = Clear_Axes(axes[r,l], close_spines = ['top', 'right'], ifyticks = True, ifxticks=True)
            idx = np.where((Data['Training Day'] == training_day[r*6+l-24])&(Data['Maze Type'] == 'Maze 2')&(Data['MiceID'] == mice))[0]
            axes[r,l].hist(Data['field_number'][idx], bins = 15)
            axes[r,l].set_xticks(np.linspace(0,15,4))
            axes[r,l].set_ylabel("Cell Count, "+training_day[r*6+l-24])
            axes[r,l].set_xlabel("Field Number, Animal #"+mice+', Maze 2')
plt.tight_layout()
plt.savefig(os.path.join(figpath, code_id+'-3-Field Number Distribution-Cross Maze-'+mice+'.png'), dpi = 600)
plt.savefig(os.path.join(figpath, code_id+'-3-Field Number Distribution-Cross Maze-'+mice+'.svg'), dpi = 600)
plt.close()


# field number distribution ---------------------------------------------------------------------------------------------------------------------------------------
mice = '11092'
training_day = ['Pre 1', 'Pre 2', 'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9']
fig, axes = plt.subplots(ncols = 6, nrows = 6, figsize = (4*6,3*6))
# open field
for r in [0,1]:
    for l in range(6):
        if r*6 + l == 11:
            axes[r,l] = Clear_Axes(axes[r,l])
        else:
            axes[r,l] = Clear_Axes(axes[r,l], close_spines = ['top', 'right'], ifyticks = True, ifxticks=True)
            idx = np.where((Data['Training Day'] == training_day[r*6+l])&(Data['Maze Type'] == 'Open Field')&(Data['MiceID'] == mice))[0]
            axes[r,l].hist(Data['field_number'][idx], bins = 15)
            axes[r,l].set_xticks(np.linspace(0,20,5))
            axes[r,l].set_ylabel("Cell Count, "+training_day[r*6+l])
            axes[r,l].set_xlabel("Field Number, Animal #"+mice+', Open Field')
# Maze 1
for r in [2,3]:
    for l in range(6):
        if r*6 + l - 12 == 11:
            axes[r,l] = Clear_Axes(axes[r,l])
        else:
            axes[r,l] = Clear_Axes(axes[r,l], close_spines = ['top', 'right'], ifyticks = True, ifxticks=True)
            idx = np.where((Data['Training Day'] == training_day[r*6+l-12])&(Data['Maze Type'] == 'Maze 1')&(Data['MiceID'] == mice))[0]
            axes[r,l].hist(Data['field_number'][idx], bins = 15)
            axes[r,l].set_xticks(np.linspace(0,20,5))
            axes[r,l].set_ylabel("Cell Count, "+training_day[r*6+l-12])
            axes[r,l].set_xlabel("Field Number, Animal #"+mice+', Maze 1')
# Maze 2
for r in [4,5]:
    for l in range(6):
        if r*6 + l - 24 == 11:
            axes[r,l] = Clear_Axes(axes[r,l])
        else:
            axes[r,l] = Clear_Axes(axes[r,l], close_spines = ['top', 'right'], ifyticks = True, ifxticks=True)
            idx = np.where((Data['Training Day'] == training_day[r*6+l-24])&(Data['Maze Type'] == 'Maze 2')&(Data['MiceID'] == mice))[0]
            axes[r,l].hist(Data['field_number'][idx], bins = 15)
            axes[r,l].set_xticks(np.linspace(0,20,5))
            axes[r,l].set_ylabel("Cell Count, "+training_day[r*6+l-24])
            axes[r,l].set_xlabel("Field Number, Animal #"+mice+', Maze 2')
plt.tight_layout()
plt.savefig(os.path.join(figpath, code_id+'-3-Field Number Distribution-Cross Maze-'+mice+'.png'), dpi = 600)
plt.savefig(os.path.join(figpath, code_id+'-3-Field Number Distribution-Cross Maze-'+mice+'.svg'), dpi = 600)
plt.close()
"""