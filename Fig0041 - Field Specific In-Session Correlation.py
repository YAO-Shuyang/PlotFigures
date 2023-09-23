from mylib.statistic_test import *

code_id = "0041 - Field Specific In-Session Correlation"
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Center ID', 'Field Size', 'Center Rate', 'In-field OEC', 'In-field FSC', 'Path Type'], f = f1, 
                              function = InFieldCorrelation_Interface, 
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

Data['Field Size'] *= 4

def FieldStabilityDistribution(Data, key, **kwargs):
    x_min, x_max = np.nanmin(Data[key]), np.nanmax(Data[key])
    
    if key == 'Field Size':
        bins = 50
    else:
        bins = 100
    
    
    fig = plt.figure(figsize=(4,3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    color = sns.color_palette("rocket", 1)
    b = ax.hist(Data[key], range=(x_min, x_max+1), bins=bins, color = color, **kwargs)[0]
    y_max = np.nanmax(b)
    ax.set_xlabel(key)
    ax.set_yticks(ColorBarsTicks(y_max, is_auto=True, tick_number=8))
    ax.set_title(Data[key].shape[0])
    plt.tight_layout()
    plt.savefig(os.path.join(loc, key+'.png'), dpi = 2400)
    plt.savefig(os.path.join(loc, key+'.svg'), dpi = 2400)
    plt.close()


def co_in_ratio(Data):
    res = np.zeros((6,2), dtype=np.float64)
    days = ['Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9']
    for i, d in enumerate(days):
        in1 = len(np.where((SubData['Maze Type'] == 'Maze 1')&(SubData['Path Type'] == 0)&(Data['Training Day'] == d))[0])
        co1 = len(np.where((SubData['Maze Type'] == 'Maze 1')&(SubData['Path Type'] == 1)&(Data['Training Day'] == d))[0])
        in2 = len(np.where((SubData['Maze Type'] == 'Maze 2')&(SubData['Path Type'] == 0)&(Data['Training Day'] == d))[0])
        co2 = len(np.where((SubData['Maze Type'] == 'Maze 2')&(SubData['Path Type'] == 1)&(Data['Training Day'] == d))[0])
        res[i, 0] = co1/in1
        res[i, 1] = co2/in2
    return res


idx = np.where(((Data['Training Day']=='Day 4')|
               (Data['Training Day']=='Day 5')|
               (Data['Training Day']=='Day 6')|
               (Data['Training Day']=='Day 7')|
               (Data['Training Day']=='Day 8')|
               (Data['Training Day']=='Day 9'))&(Data['Maze Type'] != 'Open Field')&(np.isnan(Data['In-field OEC'])==False))[0]
SubData = SubDict(Data, Data.keys(), idx)

FieldStabilityDistribution(SubData, 'In-field OEC')
FieldStabilityDistribution(SubData, 'Field Size')
FieldStabilityDistribution(SubData, 'Center Rate')
FieldStabilityDistribution(SubData, 'In-field FSC')

key = 'Field Size'
x_min, x_max = np.nanmin(SubData[key]), np.nanmax(SubData[key])
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
color = sns.color_palette("rocket", 1)
b = ax.hist(SubData[key], range=(x_min, x_max+1), color = color, bins = 100)[0]
y_max = np.nanmax(b)
ax.set_xlabel(key)
ax.set_title(SubData[key].shape[0])
ax.semilogy()
plt.tight_layout()
plt.savefig(os.path.join(loc, '[semilogy] '+key+'.png'), dpi = 2400)
plt.savefig(os.path.join(loc, '[semilogy] '+key+'.svg'), dpi = 2400)
plt.close()

key = 'Center Rate'
x_min, x_max = np.nanmin(SubData[key]), np.nanmax(SubData[key])
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
color = sns.color_palette("rocket", 1)
b = ax.hist(SubData[key], range=(x_min, x_max+1), color = color, bins = 50)[0]
y_max = np.nanmax(b)
ax.set_xlabel(key)
ax.set_title(SubData[key].shape[0])
ax.semilogy()
plt.tight_layout()
plt.savefig(os.path.join(loc, '[semilogy] '+key+'.png'), dpi = 2400)
plt.savefig(os.path.join(loc, '[semilogy] '+key+'.svg'), dpi = 2400)
plt.close()

from scipy.stats import ttest_ind
in1 = np.where((SubData['Maze Type'] == 'Maze 1')&(SubData['Path Type'] == 0))[0]
co1 = np.where((SubData['Maze Type'] == 'Maze 1')&(SubData['Path Type'] == 1))[0]
in2 = np.where((SubData['Maze Type'] == 'Maze 2')&(SubData['Path Type'] == 0))[0]
co2 = np.where((SubData['Maze Type'] == 'Maze 2')&(SubData['Path Type'] == 1))[0]
_, p1 = ttest_ind(SubData['In-field OEC'][in1], SubData['In-field OEC'][co1])
_, p2 = ttest_ind(SubData['In-field OEC'][in2], SubData['In-field OEC'][co2])
print(p1,p2)

fig = plt.figure(figsize=(3,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette("rocket", 2)
sns.barplot(x='Path Type', y='In-field OEC', data=SubData, hue='Maze Type', palette=colors,
            width=0.6, capsize=0.05, errorbar=("ci",95), errwidth=0.5, errcolor='black')
ax.legend(bbox_to_anchor = (1,1), loc = 'upper left', title='Training Day', 
          title_fontsize = 8, fontsize = 8, ncol = 1, facecolor = 'white',
          edgecolor = 'white')
ax.set_ylabel("In-field OEC")
ax.set_yticks(np.linspace(0,1,6))
ax.set_xticks([0,1],['Incorrect\nPath', 'Correct\nPath'])
ax.axhline(0, color = 'black')
ax.axis([-0.5, 1.5, 0,1])
plt.tight_layout()
plt.savefig(os.path.join(loc, '[Correlation] Path Comparison (familiar).png'), dpi=2400)
plt.savefig(os.path.join(loc, '[Correlation] Path Comparison (familiar).svg'), dpi=2400)
plt.close()



from scipy.stats import ttest_1samp
res = co_in_ratio(SubData)
ctrl1 = len(CorrectPath_maze_1)/len(IncorrectPath_maze_1)
ctrl2 = len(CorrectPath_maze_2)/len(IncorrectPath_maze_2)

_, p1 = ttest_1samp(res[:, 0], ctrl1)
_, p2 = ttest_1samp(res[:, 1], ctrl2)
print(p1, p2)

fig = plt.figure(figsize=(3,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette("rocket", 2)
ax.plot(np.random.rand(6)*0.5-0.25+1, res[:, 0], 'o', color = colors[0], 
        label = 'Maze 1', markeredgewidth = 0, markersize = 3)
ax.plot(np.random.rand(6)*0.5-0.25+2, res[:, 1], 'o', color = colors[1], 
        label = 'Maze 2', markeredgewidth = 0, markersize = 3)

ax.set_ylabel("In-field OEC")
ax.plot([0.75,1.25],[np.mean(res[:,0]), np.mean(res[:,0])], '-', color = 'black', linewidth = 1)
ax.plot([1.75,2.25],[np.mean(res[:,1]), np.mean(res[:,1])], '-', color = 'black', linewidth = 1)
ax.plot([0.75,1.25], [ctrl1, ctrl1], ':', label = 'Expected Ratio\nMaze 1', color = colors[0], linewidth = 1)
ax.plot([1.75,2.25], [ctrl2, ctrl2], ':', label = 'Expected Ratio\nMaze 2', color = colors[1], linewidth = 1)
ax.legend(bbox_to_anchor = (1,1), loc = 'upper left', title='Maze Type', 
          title_fontsize = 8, fontsize = 8, ncol = 1, facecolor = 'white',
          edgecolor = 'white')
ax.set_yticks(np.linspace(0,6,4))
ax.set_xticks([1,2],['Maze 1', 'Maze 2'])
ax.axis([0.5,2.5,0,6])
plt.tight_layout()
plt.savefig(os.path.join(loc, 'Path Ratio (familiar).png'), dpi=2400)
plt.savefig(os.path.join(loc, 'Path Ratio (familiar).svg'), dpi=2400)
plt.close()
    
    
def plot_figure_maze(Data, env, mice, day, maze_type, file_name):
    idx = np.where((Data['MiceID'] == mice)&(Data['Maze Type'] == env)&(Data['Training Day'] == day))[0]
    SubData = SubDict(Data, keys=Data.keys(), idx=idx)
    
    field_center_nodes = spike_nodes_transform(SubData['Center ID'], 12)
    x = NodesReorder(field_center_nodes, maze_type=maze_type)
    l = len(CorrectPath_maze_1) if maze_type == 1 else len(CorrectPath_maze_2)
    idx = np.where(x <= l)[0]
    SubData1 = SubDict(SubData, keys=Data.keys(), idx=idx)
    x = x[idx]
    
    image = ImageBase(maze_type)
    y_max = np.nanmax(SubData['In-field OEC'])
    image.add_main_cp_top_band(y_max)
    image.add_main_ip_top_band(y_max)
      
    colors = sns.color_palette("rocket", 1)

    sns.lineplot(x=x, 
                    y = SubData1['In-field OEC'],
                    #data = SubData1, 
                    alpha = 0.5,
                    #size = SubData1['In-field OEC'],
                    #sizes=(1,2),
                    ax = image.ax2,
                    legend=False)
    image.ax2.set_yticks(ColorBarsTicks(y_max, is_auto=True, tick_number=5))
    image.ax2.axis([0, l+1, -0.2, y_max*1.1])
    image.savefig(loc, file_name=file_name)
    
plot_figure_maze(Data, 'Maze 1', '11095', 'Day 9', 1, '11095-Maze 1-Day 9')
plot_figure_maze(Data, 'Maze 2', '11095', 'Day 9', 2, '11095-Maze 2-Day 9')
plot_figure_maze(Data, 'Maze 1', '11092', 'Day 9', 1, '11092-Maze 1-Day 9')
plot_figure_maze(Data, 'Maze 2', '11092', 'Day 9', 2, '11092-Maze 2-Day 9')