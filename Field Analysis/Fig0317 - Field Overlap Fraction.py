from mylib.statistic_test import *
from mylib.multiday.field_tracker import field_overlapping

code_id = '0317 - Field Overlap Fraction'
loc = join(figpath, code_id)
mkdir(loc)


if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = {"MiceID": np.array([], np.int64), "Maze Type": np.array([]),  "Stage": np.array([], np.int64),
            "Start Sessions": np.array([], np.int64), "Interval": np.array([], np.int64), "Overlap Fraction": np.array([], np.float64)}
    
    for i in range(len(f_CellReg_day)):
        if f_CellReg_day['include'][i] == 0 or f_CellReg_day['maze_type'][i] == 0:
            continue
        
        with open(f_CellReg_day['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
            
        fraction, start_sessions, intervals = field_overlapping(trace)
        size = fraction.shape[0]
        
        Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(int(f_CellReg_day['MiceID'][i]), size)])
        Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat("Maze "+str(int(f_CellReg_day['maze_type'][i])), size)])
        Data['Stage'] = np.concatenate([Data['Stage'], np.repeat(f_CellReg_day['Stage'][i], size)])
        Data['Start Sessions'] = np.concatenate([Data['Start Sessions'], start_sessions])
        Data['Interval'] = np.concatenate([Data['Interval'], intervals])
        Data['Overlap Fraction'] = np.concatenate([Data['Overlap Fraction'], fraction*100])
        
        del trace
        gc.collect()
        
    with open(join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
        
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)


idx = np.where((Data['MiceID'] == 10212)&(Data['Maze Type'] == 'Maze 1')&(Data['Stage'] == 'Stage 1'))[0]
SubData = SubDict(Data, Data.keys(), idx)
print(SubData)
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x='Interval', 
    y='Overlap Fraction',
    data=SubData,
    hue='Start Sessions',
    ax=ax,
)
ax.set_ylim(0, 100)
plt.show()

"""
idx = np.where(Data['Interval'] == 5)[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Start Sessions',
    y = 'Overlap Fraction',
    data=SubData,
    hue = "Maze Type",
    palette = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[3]],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 103)
ax.set_yticks(np.linspace(0, 100, 6))

plt.savefig(join(loc, 'overlap fraction.png'), dpi = 600)
plt.savefig(join(loc, 'overlap fraction.svg'), dpi = 600)
plt.show()
"""
