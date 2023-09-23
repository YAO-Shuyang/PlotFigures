from mylib.statistic_test import *
from scipy.stats import linregress

def y(x, slope, intercepts):
    return slope*x + intercepts

code_id = "0040 - Field number vs in-session stability"
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Field Number', 'In-session OEC', 'In-session FSC'], f = f1, 
                              function = FieldNumber_InSessionStability_Interface,
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
idx = np.where(((Data['Training Day']=='Day 4')|
               (Data['Training Day']=='Day 5')|
               (Data['Training Day']=='Day 6')|
               (Data['Training Day']=='Day 7')|
               (Data['Training Day']=='Day 8')|
               (Data['Training Day']=='Day 9'))&(Data['Maze Type'] != 'Open Field'))[0]
                
SubData = SubDict(Data, Data.keys(), idx)
x_max = np.nanmax(SubData['Field Number'])

x = np.linspace(1, np.nanmax(SubData['Field Number']),1000)
maze1_idx = np.where(SubData['Maze Type'] == 'Maze 1')[0]
maze2_idx = np.where(SubData['Maze Type'] == 'Maze 2')[0]
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(SubData['Field Number'][maze1_idx], SubData['In-session OEC'][maze1_idx])
print(linregress(SubData['Field Number'][maze1_idx], SubData['In-session OEC'][maze1_idx]))
slope2, intercept2, r_value2, p_value2, std_err2 = linregress(SubData['Field Number'][maze2_idx], SubData['In-session OEC'][maze2_idx])
print(linregress(SubData['Field Number'][maze2_idx], SubData['In-session OEC'][maze2_idx]))

num = SubData['Field Number'].shape[0]
SubData['Field Number'] += np.random.rand(num)-0.5

fig = plt.figure(figsize = (5,2))
colors = sns.color_palette("rocket", 2)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.scatterplot(x='Field Number', 
                y = 'In-session OEC',
                hue = 'Maze Type',
                data = SubData, 
                alpha = 0.5,
                size = 'In-session OEC',
                sizes=(0.3,0.8),
                palette=colors)
ax.plot(x, y(x, slope1, intercept1), color = colors[0], linewidth = 0.5,
        label = f'Maze 1\nr={round(r_value1,2)}\nk={round(slope1,2)}\nb={round(intercept1,2)}\np={p_value1}')
ax.plot(x, y(x, slope2, intercept2), color = colors[1], linewidth = 0.5,
        label = f'Maze 2\nr={round(r_value2,2)}\nk={round(slope2,2)}\nb={round(intercept2,2)}\np={p_value2}')
ax.legend(bbox_to_anchor = (1,1), loc = 'upper left', title='Maze Type', 
          title_fontsize = 8, fontsize = 8, ncol = 2, facecolor = 'white',
          edgecolor = 'white')
ax.set_yticks(np.linspace(0,1,6))
ax.set_xticks([0, x_max])
ax.axis([0, x_max, 0, 1])
plt.tight_layout()
plt.savefig(os.path.join(loc, 'In-session OEC vs field number.png'), dpi = 2400)
plt.savefig(os.path.join(loc, 'In-session OEC vs field number.svg'), dpi = 2400)
plt.close()





idx = np.where(((Data['Training Day']=='Day 4')|
               (Data['Training Day']=='Day 5')|
               (Data['Training Day']=='Day 6')|
               (Data['Training Day']=='Day 7')|
               (Data['Training Day']=='Day 8')|
               (Data['Training Day']=='Day 9'))&(Data['Maze Type'] != 'Open Field'))[0]
                
SubData = SubDict(Data, Data.keys(), idx)
x_max = np.nanmax(SubData['Field Number'])

x = np.linspace(1, np.nanmax(SubData['Field Number']),1000)
maze1_idx = np.where(SubData['Maze Type'] == 'Maze 1')[0]
maze2_idx = np.where(SubData['Maze Type'] == 'Maze 2')[0]
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(SubData['Field Number'][maze1_idx], SubData['In-session FSC'][maze1_idx])
print(linregress(SubData['Field Number'][maze1_idx], SubData['In-session FSC'][maze1_idx]))
slope2, intercept2, r_value2, p_value2, std_err2 = linregress(SubData['Field Number'][maze2_idx], SubData['In-session FSC'][maze2_idx])
print(linregress(SubData['Field Number'][maze2_idx], SubData['In-session FSC'][maze2_idx]))

num = SubData['Field Number'].shape[0]
SubData['Field Number'] += np.random.rand(num)-0.5

fig = plt.figure(figsize = (5,2))
colors = sns.color_palette("rocket", 2)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.scatterplot(x='Field Number', 
                y = 'In-session FSC',
                hue = 'Maze Type',
                data = SubData, 
                alpha = 0.5,
                size = 'In-session FSC',
                sizes=(0.3,0.8),
                palette=colors)
ax.plot(x, y(x, slope1, intercept1), color = colors[0], linewidth = 0.5,
        label = f'Maze 1\nr={round(r_value1,2)}\nk={round(slope1,2)}\nb={round(intercept1,2)}\np={p_value1}')
ax.plot(x, y(x, slope2, intercept2), color = colors[1], linewidth = 0.5,
        label = f'Maze 2\nr={round(r_value2,2)}\nk={round(slope2,2)}\nb={round(intercept2,2)}\np={p_value2}')
ax.legend(bbox_to_anchor = (1,1), loc = 'upper left', title='Maze Type', 
          title_fontsize = 8, fontsize = 8, ncol = 2, facecolor = 'white',
          edgecolor = 'white')
ax.set_yticks(np.linspace(0,1,6))
ax.set_xticks([0, x_max])
ax.axis([0, x_max, 0, 1])
plt.tight_layout()
plt.savefig(os.path.join(loc, 'In-session FSC vs field number.png'), dpi = 2400)
plt.savefig(os.path.join(loc, 'In-session FSC vs field number.svg'), dpi = 2400)
plt.close()







idx = np.where(((Data['Training Day']=='Day 4')|
               (Data['Training Day']=='Day 5')|
               (Data['Training Day']=='Day 6')|
               (Data['Training Day']=='Day 7')|
               (Data['Training Day']=='Day 8')|
               (Data['Training Day']=='Day 9'))&(Data['Maze Type'] == 'Open Field'))[0]
                
SubData = SubDict(Data, Data.keys(), idx)

x = np.linspace(1, np.nanmax(SubData['Field Number']),1000)
slope, intercept, r_value, p_value, std_err = linregress(SubData['Field Number'], SubData['In-session OEC'])
print(linregress(SubData['Field Number'], SubData['In-session OEC']))

num = SubData['Field Number'].shape[0]
SubData['Field Number'] += np.random.rand(num)-0.5

fig = plt.figure(figsize = (5,2))
colors = sns.color_palette("rocket", 1)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.scatterplot(x='Field Number', 
                y = 'In-session OEC',
                hue = 'Maze Type',
                data = SubData, 
                alpha = 0.5,
                size = 'In-session OEC',
                sizes=(0.3,0.8),
                palette=colors)
ax.plot(x, y(x, slope, intercept), color = colors[0], linewidth = 0.5,
        label = f'Maze 1\nr={round(r_value,2)}\nk={round(slope,2)}\nb={round(intercept,2)}\np={p_value}')
ax.legend(bbox_to_anchor = (1,1), loc = 'upper left', title='Maze Type', 
          title_fontsize = 8, fontsize = 8, ncol = 2, facecolor = 'white',
          edgecolor = 'white')
x_max = np.nanmax(SubData['Field Number'])
ax.set_yticks(np.linspace(0,1,6))
ax.set_xticks([0, x_max])
ax.axis([0, x_max, 0, 1])
plt.tight_layout()
plt.savefig(os.path.join(loc, '[Open Field] In-session OEC vs field number.png'), dpi = 2400)
plt.savefig(os.path.join(loc, '[Open Field] In-session OEC vs field number.svg'), dpi = 2400)
plt.close()








idx = np.where(((Data['Training Day']=='Day 4')|
               (Data['Training Day']=='Day 5')|
               (Data['Training Day']=='Day 6')|
               (Data['Training Day']=='Day 7')|
               (Data['Training Day']=='Day 8')|
               (Data['Training Day']=='Day 9'))&(Data['Maze Type'] == 'Open Field'))[0]
                
SubData = SubDict(Data, Data.keys(), idx)

x = np.linspace(1, np.nanmax(SubData['Field Number']),1000)
slope, intercept, r_value, p_value, std_err = linregress(SubData['Field Number'], SubData['In-session FSC'])
print(linregress(SubData['Field Number'], SubData['In-session FSC']))

num = SubData['Field Number'].shape[0]
SubData['Field Number'] += np.random.rand(num)-0.5

fig = plt.figure(figsize = (5,2))
colors = sns.color_palette("rocket", 1)
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.scatterplot(x='Field Number', 
                y = 'In-session FSC',
                hue = 'Maze Type',
                data = SubData, 
                alpha = 0.5,
                size = 'In-session FSC',
                sizes=(0.3,0.8),
                palette=colors)
ax.plot(x, y(x, slope, intercept), color = colors[0], linewidth = 0.5,
        label = f'Maze 1\nr={round(r_value,2)}\nk={round(slope,2)}\nb={round(intercept,2)}\np={p_value}')
ax.legend(bbox_to_anchor = (1,1), loc = 'upper left', title='Maze Type', 
          title_fontsize = 8, fontsize = 8, ncol = 2, facecolor = 'white',
          edgecolor = 'white')
x_max = np.nanmax(SubData['Field Number'])
ax.set_yticks(np.linspace(0,1,6))
ax.set_xticks(ColorBarsTicks(x_max, is_auto=True, tick_number=4))
ax.axis([0, x_max, 0, 1])
plt.tight_layout()
plt.savefig(os.path.join(loc, '[Open Field] In-session FSC vs field number.png'), dpi = 2400)
plt.savefig(os.path.join(loc, '[Open Field] In-session FSC vs field number.svg'), dpi = 2400)
plt.close()