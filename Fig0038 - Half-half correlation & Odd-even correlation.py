from mylib.statistic_test import *

code_id = '0038 - Half-half or Odd-even correlation'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Half-half Correlation', 'Odd-even Correlation', 'Cell Type'], f = f1, 
                              function = InterSessionCorrelation_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

def clear_nan_value(data):
    idx = np.where(np.isnan(data))[0]
    return np.delete(data, idx)
    
   
# Compare cells half-half correlation and odd-even correlation between place cells and non-place cells.
idx = np.where((Data['Training Day'] == 'Day 4')|(Data['Training Day'] == 'Day 5')|(Data['Training Day'] == 'Day 6')|(Data['Training Day'] == 'Day 7')|(Data['Training Day'] == 'Day 8')|(Data['Training Day'] == 'Day 9'))[0]
SubData = SubDict(Data, Data.keys(), idx)

op_pc = SubData['Half-half Correlation'][np.where((SubData['Cell Type']==1)&(SubData['Maze Type'] == 'Open Field'))[0]]
op_npc = SubData['Half-half Correlation'][np.where((SubData['Cell Type']==0)&(SubData['Maze Type'] == 'Open Field'))[0]]
op_pc, op_npc = clear_nan_value(op_pc), clear_nan_value(op_npc)
var = scipy.stats.levene(op_pc, op_npc) # -> 方差非齐性
print(var)
_, p_op = scipy.stats.ttest_ind(op_pc, op_npc,equal_var=False)
print(p_op) #-> ****

m1_pc = SubData['Half-half Correlation'][np.where((SubData['Cell Type']==1)&(SubData['Maze Type'] == 'Maze 1'))[0]]
m1_npc = SubData['Half-half Correlation'][np.where((SubData['Cell Type']==0)&(SubData['Maze Type'] == 'Maze 1'))[0]]
m1_pc, m1_npc = clear_nan_value(m1_pc), clear_nan_value(m1_npc)
var = scipy.stats.levene(m1_pc, m1_npc) # -> 方差齐性
print(var)
_, p_m1 = scipy.stats.ttest_ind(m1_pc, m1_npc, equal_var=True)
print(p_m1) #-> ****

m2_pc = SubData['Half-half Correlation'][np.where((SubData['Cell Type']==1)&(SubData['Maze Type'] == 'Maze 2'))[0]]
m2_npc = SubData['Half-half Correlation'][np.where((SubData['Cell Type']==0)&(SubData['Maze Type'] == 'Maze 2'))[0]]
m2_pc, m2_npc = clear_nan_value(m2_pc), clear_nan_value(m2_npc)
var = scipy.stats.levene(m2_pc, m2_npc) # -> 方差非齐性
print(var)
_, p_m2 = scipy.stats.ttest_ind(m2_pc, m2_npc, equal_var=False)
print(p_m2) #-> ****


fig = plt.figure(figsize=(3,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette("rocket", 3)
sns.barplot(ax = ax, x = 'Cell Type', y = 'Half-half Correlation', data = SubData, palette=colors,
            width = 0.6,capsize=0.05, errorbar=("ci",95), errwidth=0.5, errcolor='black',
            hue = 'Maze Type')


ax.legend(bbox_to_anchor = (1,1), loc = 'upper left', title='Training Day', 
          title_fontsize = 8, fontsize = 8, ncol = 1, facecolor = 'white',
          edgecolor = 'white')
ax.set_ylabel("first half vs. Second half\nCorrelation")
ax.set_yticks(np.linspace(0,1,6))
ax.set_xticks([0,1],['Non-place\ncell', 'place cell'])
ax.axhline(0, color = 'black')
ax.axis([-0.5, 1.5, 0,1])
plt.tight_layout()
plt.savefig(os.path.join(loc, 'Half-half Correlation (familiar).png'), dpi=2400)
plt.savefig(os.path.join(loc, 'Half-half Correlation (familiar).svg'), dpi=2400)
plt.close()



# Compare cells odd-first correlation and odd-even correlation between place cells and non-place cells.
idx = np.where((Data['Training Day'] == 'Day 4')|(Data['Training Day'] == 'Day 5')|(Data['Training Day'] == 'Day 6')|(Data['Training Day'] == 'Day 7')|(Data['Training Day'] == 'Day 8')|(Data['Training Day'] == 'Day 9'))[0]
SubData = SubDict(Data, Data.keys(), idx)

op_pc = SubData['Odd-even Correlation'][np.where((SubData['Cell Type']==1)&(SubData['Maze Type'] == 'Open Field'))[0]]
op_npc = SubData['Odd-even Correlation'][np.where((SubData['Cell Type']==0)&(SubData['Maze Type'] == 'Open Field'))[0]]
op_pc, op_npc = clear_nan_value(op_pc), clear_nan_value(op_npc)
var = scipy.stats.levene(op_pc, op_npc) # -> 方差非齐性
print(var)
_, p_op = scipy.stats.ttest_ind(op_pc, op_npc,equal_var=False)
print(p_op) #-> ****

m1_pc = SubData['Odd-even Correlation'][np.where((SubData['Cell Type']==1)&(SubData['Maze Type'] == 'Maze 1'))[0]]
m1_npc = SubData['Odd-even Correlation'][np.where((SubData['Cell Type']==0)&(SubData['Maze Type'] == 'Maze 1'))[0]]
m1_pc, m1_npc = clear_nan_value(m1_pc), clear_nan_value(m1_npc)
var = scipy.stats.levene(m1_pc, m1_npc) # -> 方差非齐性
print(var)
_, p_m1 = scipy.stats.ttest_ind(m1_pc, m1_npc, equal_var=False)
print(p_m1) #-> ****

m2_pc = SubData['Odd-even Correlation'][np.where((SubData['Cell Type']==1)&(SubData['Maze Type'] == 'Maze 2'))[0]]
m2_npc = SubData['Odd-even Correlation'][np.where((SubData['Cell Type']==0)&(SubData['Maze Type'] == 'Maze 2'))[0]]
m2_pc, m2_npc = clear_nan_value(m2_pc), clear_nan_value(m2_npc)
var = scipy.stats.levene(m2_pc, m2_npc) # -> 方差非齐性
print(var)
_, p_m2 = scipy.stats.ttest_ind(m2_pc, m2_npc, equal_var=False)
print(p_m2) #-> ****


fig = plt.figure(figsize=(3,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette("rocket", 3)
sns.barplot(ax = ax, x = 'Cell Type', y = 'Odd-even Correlation', data = SubData, palette=colors,
            width = 0.6,capsize=0.05, errorbar=("ci",95), errwidth=0.5, errcolor='black',
            hue = 'Maze Type')


ax.legend(bbox_to_anchor = (1,1), loc = 'upper left', title='Training Day', 
          title_fontsize = 8, fontsize = 8, ncol = 1, facecolor = 'white',
          edgecolor = 'white')
ax.set_ylabel("Odd lap vs. Even lap\nCorrelation")
ax.set_yticks(np.linspace(0,1,6))
ax.set_xticks([0,1],['Non-place\ncell', 'place cell'])
ax.axhline(0, color = 'black')
ax.axis([-0.5, 1.5, 0,1])
plt.tight_layout()
plt.savefig(os.path.join(loc, 'Odd-even Correlation (familiar).png'), dpi=2400)
plt.savefig(os.path.join(loc, 'Odd-even Correlation (familiar).svg'), dpi=2400)
plt.close()