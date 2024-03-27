# Using NaiveBayesDecoder we can decode animal's real location from neural population data.

from mylib.statistic_test import *

code_id = '0030 - Decoding Results'
loc = os.path.join(figpath, code_id)
mkdir(os.path.join(figpath, code_id))


if os.path.exists(os.path.join(figdata, code_id+'.pkl')): 
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Error'], f = f_decode,
                              function = NeuralDecodingResults_Interface, f_member=['Lap ID'],
                              file_name = code_id, behavior_paradigm = 'decoding')

if os.path.exists(os.path.join(figdata, code_id+' [shuffle].pkl')): 
    with open(os.path.join(figdata, code_id+' [shuffle].pkl'), 'rb') as handle:
        ShuffleData = pickle.load(handle)
else:
    ShuffleData = DataFrameEstablish(variable_names = ['Error'], f = f_decode_shuffle,
                              function = NeuralDecodingResults_Interface, f_member=['Lap ID'],
                              file_name = code_id+' [shuffle]', behavior_paradigm = 'decoding')

SubData = {
    "Training Day": np.concatenate([Data['Training Day'], ShuffleData['Training Day']]),
    "Stage": np.concatenate([Data['Stage'], ShuffleData['Stage']]),
    "MiceID": np.concatenate([Data['MiceID'], ShuffleData['MiceID']]),
    "Maze Type": np.concatenate([Data['Maze Type'], ShuffleData['Maze Type']]),
    "Lap ID": np.concatenate([Data['Lap ID'], ShuffleData['Lap ID']]),
    "Error": np.concatenate([Data['Error'], ShuffleData['Error']]),
    "Data Type": np.concatenate([np.repeat("Data", len(Data['Error'])), np.repeat("Shuffle", len(ShuffleData['Error']))]),
}

SubData['hue'] = np.array([SubData['Maze Type'][i] + SubData['Data Type'][i] for i in range(len(SubData['Maze Type']))])

def create_example_video(res, save_loc: str):
    y_pred, y_test = res['y_pred'], res['y_test']
    """
    fig = plt.figure(figsize=(4,4))
    ax = Clear_Axes(plt.axes())
    ax.set_aspect("equal")
    ax.axis([-1, 48, 48, -1])
    DrawMazeProfile(axes=ax, maze_type=1, color='black', linewidth=2, nx=48)
    
    color = sns.color_palette("rocket", 3)
    
    xps, yps = (y_pred-1)%48 + np.random.rand(y_pred.shape[0])-0.5, (y_pred-1)//48 + np.random.rand(y_pred.shape[0])-0.5
    xts, yts = (y_test-1)%48 + np.random.rand(y_pred.shape[0])-0.5, (y_test-1)//48 + np.random.rand(y_pred.shape[0])-0.5    

    for i in tqdm(range(len(y_pred))):

        b = ax.plot([xts[i]], [yts[i]], 'o', markersize=5, markeredgewidth=0, color='black', label = 'Real Pos.')
        c = ax.plot(xts[:i+1], yts[:i+1], color = 'gray', linewidth = 1, label = "Real Traj.")
        a = ax.plot([xps[i]], [yps[i]], 'o', markersize=5, markeredgewidth=0, color=color[-1], label = 'Pred. Pos.')
        d = ax.plot(xps[:i+1], yps[:i+1], color = color[0], linewidth = 1, label = "Pred. Traj.") 
               
        ax.legend(ncols=2, loc='lower left', bbox_to_anchor=(0, 1, 1, 0.2), fontsize = 8, facecolor='white', edgecolor='white')
        
        plt.savefig(os.path.join(save_loc, f'Frame {i+1}.png'), dpi=600)
        
        for r in a+b+c+d:
            r.remove()
    """
    images_to_video(save_loc, length = len(y_pred))


idx = np.where((SubData['MiceID'] != 11095)&(SubData['MiceID'] != 11092))[0]
Data = SubDict(SubData, SubData.keys(), idx)
Data['x'] = np.zeros_like(Data['Lap ID'])
idx1 = np.where(Data['Lap ID'] == 1)[0]
Data['x'][idx1] = 1
idx2 = np.where(Data['Lap ID'] == 2)[0]
Data['x'][idx2] = 2
idx3 = np.where(Data['Lap ID'] == 3)[0]
Data['x'][idx3] = 3
idx4 = np.where(Data['Lap ID'] == 4)[0]
Data['x'][idx4] = 4
idx5 = np.where(Data['Lap ID'] == 5)[0]
Data['x'][idx5] = 5
idx6 = np.where(Data['Lap ID'] == 6)[0]
Data['x'][idx6] = 6
idx7 = np.where(Data['Lap ID'] == 7)[0]
Data['x'][idx7] = 7
idx8 = np.where(Data['Lap ID'] == 8)[0]
Data['x'][idx8] = 8
idx9 = np.where(Data['Lap ID'] == 9)[0]
Data['x'][idx9] = 9
idx10 = np.where(Data['Lap ID'] >= 10)[0]
Data['x'][idx10] = 10

idx = np.concatenate([np.where(Data['hue'] == hue)[0] for hue in ['Open FieldData', 'Open FieldShuffle', 'Maze 1Data', 'Maze 1Shuffle', 'Maze 2Data', 'Maze 2Shuffle']])
Data = SubDict(Data, Data.keys(), idx)

# Mean Rate
uniq_day = ['Day 1', 'Day 2', 'Day 3', 'Day 4',
            'Day 5', 'Day 6', 'Day 7', 'Day 8',
            'Day 9', '>=Day 10']
uniq_s = ['S'+str(i) for i in range(1,20)] + ['>=S20']

idx = np.concatenate([np.where(Data['Training Day'] == s)[0] for s in uniq_s] + [np.where(Data['Training Day'] == day)[0] for day in uniq_day])
Data = SubDict(Data, Data.keys(), idx)

colors = sns.color_palette("rocket", 3)
markercolors = [sns.color_palette("crest", 4)[2], sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[0]]
print("Start plot")


print("Cross Env comparison Stage 1:")
for day in uniq_day:
    print(f" {day} -------------------------")
    data_op, data_m1 = [], []
    for m in [10209, 10212, 10224, 10227]:
        data_op.append(np.nanmean(SubData['Error'][np.where((SubData['Maze Type'] == 'Open Field')&
                                                (SubData['MiceID'] == m)&
                                                (SubData['Training Day'] == day)&
                                                (SubData['Stage'] == 'Stage 1')&
                                                (SubData['Data Type'] == 'Data'))[0]]))
        data_m1.append(np.nanmean(SubData['Error'][np.where((SubData['Maze Type'] == 'Maze 1')&
                                                (SubData['MiceID'] == m)&
                                                (SubData['Training Day'] == day)&
                                                (SubData['Stage'] == 'Stage 1')&
                                                (SubData['Data Type'] == 'Data'))[0]]))
    try:
        print("     OP vs M1", ttest_rel(data_op, data_m1))
    except:
        print()
print()

print("Cross Env comparison Stage 2:")
for day in uniq_day:
    print(f" {day} -------------------------")
    data_op, data_m1, data_m2 = [], [], []
    for m in [10209, 10212, 10224, 10227]:
        data_op.append(np.nanmean(SubData['Error'][np.where((SubData['Maze Type'] == 'Open Field')&
                                                (SubData['MiceID'] == m)&
                                                (SubData['Training Day'] == day)&
                                                (SubData['Stage'] == 'Stage 2')&
                                                (SubData['Data Type'] == 'Data'))[0]]))
        data_m1.append(np.nanmean(SubData['Error'][np.where((SubData['Maze Type'] == 'Maze 1')&
                                                (SubData['MiceID'] == m)&
                                                (SubData['Training Day'] == day)&
                                                (SubData['Stage'] == 'Stage 2')&
                                                (SubData['Data Type'] == 'Data'))[0]]))
        data_m2.append(np.nanmean(SubData['Error'][np.where((SubData['Maze Type'] == 'Maze 2')&
                                                (SubData['MiceID'] == m)&
                                                (SubData['Training Day'] == day)&
                                                (SubData['Stage'] == 'Stage 2')&
                                                (SubData['Data Type'] == 'Data'))[0]]))
    
    print("     OP vs M1", ttest_rel(data_op, data_m1)[1]*2)
    print("     OP vs M2", ttest_rel(data_op, data_m2)[1]*2, end='\n\n')
print()

# Statistic Test
data_op_d1, data_m1_d1, data_m2_d1 = [], [], []
for m in [10209, 10212, 10224, 10227]:
    data_op_d1.append(np.nanmean(SubData['Error'][np.where((SubData['Maze Type'] == 'Open Field')&
                                                (SubData['MiceID'] == m)&
                                                (SubData['Training Day'] == 'Day 1')&
                                                (SubData['Stage'] == 'Stage 1')&
                                                (SubData['Data Type'] == 'Data'))[0][:1]]))
    data_m1_d1.append(np.nanmean(SubData['Error'][np.where((SubData['Maze Type'] == 'Maze 1')&
                                                (SubData['MiceID'] == m)&
                                                (SubData['Training Day'] == 'Day 1')&
                                                (SubData['Stage'] == 'Stage 1')&
                                                (SubData['Data Type'] == 'Data'))[0][:1]]))
    data_m2_d1.append(np.nanmean(SubData['Error'][np.where((SubData['Maze Type'] == 'Maze 2')&
                                                (SubData['MiceID'] == m)&
                                                (SubData['Training Day'] == 'Day 1')&
                                                (SubData['Stage'] == 'Stage 2')&
                                                (SubData['Data Type'] == 'Data'))[0][:1]]))

data_op_d10, data_m1_d10, data_m2_d10 = [], [], []
for m in [10209, 10212, 10224, 10227]:
    data_op_d10.append(np.nanmean(SubData['Error'][np.where((SubData['Maze Type'] == 'Open Field')&
                                                (SubData['MiceID'] == m)&
                                                (SubData['Training Day'] == '>=Day 10')&
                                                (SubData['Stage'] == 'Stage 1')&
                                                (SubData['Data Type'] == 'Data'))[0][-5:]]))
    data_m1_d10.append(np.nanmean(SubData['Error'][np.where((SubData['Maze Type'] == 'Maze 1')&
                                                (SubData['MiceID'] == m)&
                                                (SubData['Training Day'] == '>=Day 10')&
                                                (SubData['Stage'] == 'Stage 1')&
                                                (SubData['Data Type'] == 'Data'))[0][-5:]]))
    data_m2_d10.append(np.nanmean(SubData['Error'][np.where((SubData['Maze Type'] == 'Maze 2')&
                                                (SubData['MiceID'] == m)&
                                                (SubData['Training Day'] == '>=Day 10')&
                                                (SubData['Stage'] == 'Stage 2')&
                                                (SubData['Data Type'] == 'Data'))[0][-5:]]))


print(data_op_d1, data_m1_d1, data_m2_d1, data_op_d10, data_m1_d10, data_m2_d10, sep='\n\n')
print("Open Field: Day 1 vs Day 10")
print_estimator(data_op_d1)
print_estimator(data_op_d10)
print(ttest_rel(data_op_d1, data_op_d10), end='\n\n')
print("Maze 1: Day 1 vs Day 10")
print_estimator(data_m1_d1)
print_estimator(data_m1_d10)
print(ttest_rel(data_m1_d1, data_m1_d10), end='\n\n')
print("Maze 2: Day 1 vs Day 10")
print_estimator(data_m2_d1)
print_estimator(data_m2_d10)
print(ttest_rel(data_m2_d1, data_m2_d10), end='\n\n')


compdata = {
    "Error": np.concatenate([data_op_d1, data_m1_d1, data_m2_d1, data_op_d10, data_m1_d10, data_m2_d10]),
    "Maze Type": np.array(['Open Field']*len(data_op_d1) + ['Maze 1']*len(data_m1_d1) + ['Maze 2']*len(data_m2_d1) + ['Open Field']*len(data_op_d10) + ['Maze 1']*len(data_m1_d10) + ['Maze 2']*len(data_m2_d10)),
    "Session": np.concatenate([np.repeat("First", len(data_op_d1)+len(data_m1_d1)+len(data_m2_d1)), np.repeat("Last", len(data_op_d10)+len(data_m1_d10)+len(data_m2_d10))]), 
    "MiceID": np.array([10209, 10212, 10224, 10227] * 6)
}
print(compdata)
fig = plt.figure(figsize=(3,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x='Maze Type',
    y='Error',
    hue='Session',
    data=compdata,
    palette=['#003366', '#0099CC'],
    ax=ax,
    errcolor='black',
    errwidth=0.5,
    capsize=0.1,
    width=0.8
)

sns.stripplot(
    x='Maze Type',
    y='Error',
    hue='MiceID',
    data=compdata,
    palette=['#F2E8D4', '#8E9F85', '#C3AED6', '#A7D8DE'],
    edgecolor='black',
    size=4,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.15
)
plt.savefig(join(loc, 'Comparison of Sessions.png'), dpi=2400)
plt.savefig(join(loc, 'Comparison of Sessions.svg'), dpi=2400)
plt.close()


# Open Field Decode
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,2))
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[2], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.where((Data['Maze Type'] == 'Open Field')&(Data['Stage'] == 'PRE'))[0]
sns.lineplot(
    x=Data['Training Day'][idx],
    y=Data['Error'][idx],
    hue=Data['Data Type'][idx],
    palette='Blues',
    ax=ax1,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
print("Fig 1.1 done.")
ax1.set_ylim(0, 60)
ax1.set_yticks(np.linspace(0, 60, 6))

idx = np.where((Data['Maze Type'] == 'Open Field')&(Data['Stage'] == 'Stage 1'))[0]
sns.lineplot(
    x=Data['Training Day'][idx],
    y=Data['Error'][idx],
    hue=Data['Data Type'][idx],
    palette='Blues',
    ax=ax2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
print("Fig 2.1 done.")
ax2.set_ylim(0, 60)
ax2.set_yticks(np.linspace(0, 60, 6))

idx = np.where((Data['Maze Type'] == 'Open Field')&(Data['Stage'] == 'Stage 2'))[0]
sns.lineplot(
    x=Data['Training Day'][idx],
    y=Data['Error'][idx],
    hue=Data['Data Type'][idx],
    palette='Blues',
    ax=ax3,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax3.set_ylim(0, 60)
ax3.set_yticks(np.linspace(0, 60, 6))
print("Fig 3.1 done.")
plt.tight_layout()
plt.savefig(join(loc, 'Open Field Decode.png'), dpi=2400)
plt.savefig(join(loc, 'Open Field Decode.svg'), dpi=2400)
plt.close()


# Maze 1 Decode
colors = sns.color_palette("Blues", 2) + sns.color_palette("flare", 2)
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,2.3))
ax2 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Maze Type'] != 'Open Field'))[0]
sns.lineplot(
    x=Data['Training Day'][idx],
    y=Data['Error'][idx],
    hue=Data['hue'][idx],
    palette=colors[:2],
    ax=ax2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax2.set_ylim(0, 200)
ax2.set_yticks(np.linspace(0, 200, 9))
print("Fig 2.1 done.")

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Maze Type'] != 'Open Field'))[0]
sns.lineplot(
    x=Data['Training Day'][idx],
    y=Data['Error'][idx],
    hue=Data['hue'][idx],
    palette=colors,
    ax=ax3,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax3.set_ylim(0, 200)
ax3.set_yticks(np.linspace(0, 200, 9))
print("Fig 3.1 done.")
plt.tight_layout()
plt.savefig(join(loc, 'Maze Decode.png'), dpi=2400)
plt.savefig(join(loc, 'Maze Decode.svg'), dpi=2400)
plt.close()



colors = sns.color_palette("Blues", 2) + sns.color_palette("flare", 2)
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,1.1))
ax2 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.where((Data['Stage'] == 'Stage 1')&(Data['Maze Type'] != 'Open Field'))[0]
sns.lineplot(
    x=Data['Training Day'][idx],
    y=Data['Error'][idx],
    hue=Data['hue'][idx],
    palette=colors[:2],
    marker='o',
    ax=ax2,
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax2.set_ylim(7, 11)
ax2.set_xlim(1.5, 9.5)
ax2.set_yticks(np.linspace(7, 11, 5))
print("Fig 2.1 done.")

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Maze Type'] != 'Open Field'))[0]
sns.lineplot(
    x=Data['Training Day'][idx],
    y=Data['Error'][idx],
    hue=Data['hue'][idx],
    palette=colors,
    marker='o',
    ax=ax3,
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax3.set_ylim(5.7, 11.3)
ax3.set_xlim(0.5, 9.5)
ax3.set_yticks(np.linspace(6, 11, 6))
print("Fig 3.1 done.")
plt.tight_layout()
plt.savefig(join(loc, 'Maze Decode [zoom out2].png'), dpi=2400)
plt.savefig(join(loc, 'Maze Decode [zoom out2].svg'), dpi=2400)
plt.close()

# Maze 2 Decode
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,2))
ax2 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

idx = np.where((Data['Maze Type'] == 'Maze 2')&(Data['Stage'] == 'Stage 2'))[0]
sns.lineplot(
    x=Data['Training Day'][idx],
    y=Data['Error'][idx],
    hue=Data['Data Type'][idx],
    palette='Blues',
    marker='o',
    ax=ax3,
    markeredgecolor=None,
    markersize=2,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax3.set_ylim(0, 150)
ax3.set_yticks(np.linspace(0, 150, 7))
print("Fig 3.1 done.")
plt.tight_layout()
plt.savefig(join(loc, 'Maze 2 Decode.png'), dpi=2400)
plt.savefig(join(loc, 'Maze 2 Decode.svg'), dpi=2400)
plt.close()

D11 = Data['Error'][np.where((Data['Maze Type'] == 'Maze 1')&(Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 1))[0]]
print_estimator(D11)
S11 = ShuffleData['Error'][np.where((ShuffleData['Maze Type'] == 'Maze 1')&(ShuffleData['Stage'] == 'Stage 1')&(ShuffleData['Training Day'] == 'Day 1')&(ShuffleData['Lap ID'] == 1))[0]]
print_estimator(S11)
D13 = Data['Error'][np.where((Data['Maze Type'] == 'Maze 1')&(Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1')&(Data['Lap ID'] == 3))[0]]
print_estimator(D13)
S13 = ShuffleData['Error'][np.where((ShuffleData['Maze Type'] == 'Maze 1')&(ShuffleData['Stage'] == 'Stage 1')&(ShuffleData['Training Day'] == 'Day 1')&(ShuffleData['Lap ID'] == 3))[0]]
print_estimator(S13)
print(levene(D11, S11), ttest_ind(D11, S11, equal_var=False, alternative='less'), cohen_d(D11, S11))
print(levene(D13, S13), ttest_ind(D13, S13, equal_var=False, alternative='less'), cohen_d(D13, S13))
print(levene(D11, D13), ttest_ind(D11, D13, equal_var=False, alternative='greater'), cohen_d(D11, D13))
Data['Error'] = Data['Error'] + np.random.rand(Data['Error'].shape[0])*0.2
colors = [sns.color_palette("Blues", 2)[0], sns.color_palette("flare", 2)[0]]
colors2 = sns.color_palette("Blues", 2) + [sns.color_palette("flare", 2)[1]]
print(colors2)
idx1 = np.where((Data['Maze Type'] == 'Maze 1')&
                (Data['Stage'] == 'Stage 1')&
                (Data['Training Day'] == 'Day 1')&
                (Data['Data Type'] == 'Data'))[0]
fig = plt.figure(figsize=(1.8, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x=Data['x'][idx1],
    y=Data['Error'][idx1],
    hue=Data['Maze Type'][idx1],
    palette=colors2,
    ax=ax,
    legend=False,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax.set_yticks(np.linspace(0, 180, 10))
ax.set_ylim(0, 180)
ax.set_xticks(np.linspace(1, 6, 6))
ax.set_xlim(0.5, 6.5)
plt.tight_layout()
plt.savefig(join(loc, 'Stage 1 Day 1 Laps.png'), dpi=2400)
plt.savefig(join(loc, 'Stage 1 Day 1 Laps.svg'), dpi=2400)
plt.close()


idx1 = np.where((Data['Maze Type'] == 'Maze 1')&
                (Data['Stage'] == 'Stage 1')&
                (Data['Training Day'] == '>=Day 10')&
                (Data['Data Type'] == 'Data'))[0]
fig = plt.figure(figsize=(2.5,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x=Data['x'][idx1],
    y=Data['Error'][idx1],
    hue = Data['Maze Type'][idx1],
    palette = colors2,
    ax=ax,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax.set_ylim(0,30)
ax.set_yticks(np.linspace(0, 30, 7))
ax.set_xticks(np.linspace(1, 10, 10))
plt.tight_layout()
plt.savefig(join(loc, 'Stage 1 Day 10 Laps.png'), dpi=2400)
plt.savefig(join(loc, 'Stage 1 Day 10 Laps.svg'), dpi=2400)
plt.close()


idx1 = np.where((Data['Maze Type'] == 'Maze 2')&
                (Data['Stage'] == 'Stage 2')&
                (Data['Training Day'] == 'Day 1')&
                (Data['Data Type'] == 'Data'))[0]
fig = plt.figure(figsize=(2.5,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x=Data['x'][idx1],
    y=Data['Error'][idx1],
    hue = Data['hue'][idx1],
    palette=colors,
    ax=ax,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax.set_ylim(0, 150)
ax.set_yticks(np.linspace(0, 150, 6))
ax.set_xticks(np.linspace(1, 10, 10))

plt.tight_layout()
plt.savefig(join(loc, 'Stage 2 Day 1 Laps.png'), dpi=2400)
plt.savefig(join(loc, 'Stage 2 Day 1 Laps.svg'), dpi=2400)
plt.close()


idx1 = np.where((Data['Maze Type'] == 'Maze 2')&
                (Data['Stage'] == 'Stage 2')&
                (Data['Training Day'] == '>=Day 10')&
                (Data['Data Type'] == 'Data'))[0]
fig = plt.figure(figsize=(2.5,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x=Data['x'][idx1],
    y=Data['Error'][idx1],
    hue = Data['hue'][idx1],
    palette=colors,
    ax=ax,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
ax.set_ylim(0, 20)
ax.set_yticks(np.linspace(0, 20, 5))
ax.set_xticks(np.linspace(1, 10, 10))

plt.tight_layout()
plt.savefig(join(loc, 'Stage 2 Day 10 Laps.png'), dpi=2400)
plt.savefig(join(loc, 'Stage 2 Day 10 Laps.svg'), dpi=2400)
plt.close()

"""
# Maze 1: perfect lap and others
idx = np.where(Data['Maze Type'] == 'Maze 1')[0]
colors = sns.color_palette("Blues", 2) + sns.color_palette("flare", 2)
colors = sns.color_palette("Blues", 2) + sns.color_palette("flare", 2)
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3))
ax2 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax3 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.where((Data['Maze Type'] != 'Open Field')&(Data['Stage'] == 'Stage 1')&(Data['Maze Type'] != 'Open Field'))[0]
sns.lineplot(
    x=Data['Training Day'][idx],
    y=Data['Error'][idx],
    hue=Data['Is Perfect'][idx],
    palette=colors[:2],
    marker='o',
    ax=ax2,
    markeredgecolor=None,
    markersize=2,
    legend=False,
    estimator=np.median,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
print("Fig 2.1 done.")

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Maze Type'] != 'Open Field'))[0]
sns.lineplot(
    x=Data['Training Day'][idx],
    y=Data['Error'][idx],
    hue=Data['Is Perfect'][idx],
    palette=colors,
    marker='o',
    ax=ax3,
    markeredgecolor=None,
    markersize=2,
    legend=False,
    estimator=np.median,
    err_style='bars',
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5,
)
print("Fig 3.1 done.")
plt.tight_layout()
plt.savefig(join(loc, 'perfect lap vs others.png'), dpi=2400)
plt.savefig(join(loc, 'perfect lap vs others.png'), dpi=2400)
plt.close()
"""