from mylib.statistic_test import *

code_id = "0402 - Linearized Position"
loc = join(figpath, "Independent Field", code_id)
mkdir(loc)


if os.path.exists(os.path.join(figdata, 'Field Pool.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['FSC Stability', 'OEC Stability', 'Field Size', 'Field Length', 'Peak Rate', 'Position'], f = f1,
                              function = WithinFieldBasicInfo_Interface, func_kwgs = {'is_placecell': True},
                              file_name = 'Field Pool', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, 'Field Pool.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata, 'Field Statistics in Cell Pool.pkl')) == False:
    CellData = DataFrameEstablish(variable_names = ['Mean FSC', 'Std. FSC', 'Median FSC', 'Error FSC',
                                                'Mean OEC', 'Std. OEC', 'Median OEC', 'Error OEC',
                                                'Mean Size', 'Std. Size', 'Median Size', 'Error Size',
                                                'Mean Length', 'Std. Length', 'Median Length', 'Error Length',
                                                'Mean Rate', 'Std. Rate', 'Median Rate', 'Error Rate',
                                                'Mean Position', 'Std. Position', 'Median Position', 'Error Position',
                                                'Mean Interdistance', 'Std. Interdistance', 'Median Interdistance', 'Error Interdistance',
                                                'Cell ID', 'Field Number'], f = f1,
                              function = WithinCellFieldStatistics_Interface, func_kwgs = {'is_placecell': True},
                              file_name = 'Field Statistics in Cell Pool', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, 'Field Statistics in Cell Pool.pkl'), 'rb') as handle:
        CellData = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata, code_id + '.pkl')) == False:
    BehavData = DataFrameEstablish(variable_names = ['Error Num', 'Pass Number', 'Error Rate', 'Decision Point'], f = f1_behav,
                              function = ErrorTimesAndFieldFraction_Interface, is_behav=True,
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id + '.pkl'), 'rb') as handle:
        BehavData = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata, code_id + ' occu.pkl')) == False:
    TimeData = DataFrameEstablish(variable_names = ['Occupation Time', 'Decision Point'], f = f1_behav,
                              function = OccupationTimeAndFieldFraction_Interface, is_behav=True,
                              file_name = code_id + ' occu', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id + ' occu.pkl'), 'rb') as handle:
        TimeData = pickle.load(handle)

"""
idx = np.where(f1['maze_type'] != 0)[0]
if os.path.exists(os.path.join(figdata, code_id + ' Density.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Field Num', 'Position'], f = f1,
                              function = PlaceFieldCoveredDensity_Interface, file_idx=idx,
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, code_id + ' Density.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
"""

def CalCDF(pos, maze_type, bin_num = 300):
    xmin, xmax = np.min(pos), np.max(pos)+0.001
    steps = np.linspace(xmin, xmax, bin_num+1)
    PDF = np.zeros(bin_num+1)
    
    D = GetDMatrices(maze_type, 48)
    CP = cp.deepcopy(correct_paths[(int(maze_type), 48)])
    
    for i in range(len(steps)-1):
        field_num = len(np.where((pos >= steps[i])&(pos < steps[i+1]))[0])
        area_size = len(np.where((D[0, CP-1] >= steps[i])&(D[0, CP-1] < steps[i+1]))[0])
        if area_size == 0:
            continue
        PDF[i+1] = field_num/area_size
    
    CDF = np.cumsum(PDF)
    CDF/= CDF[-1]
    return steps, CDF
    

idx = np.where((Data['MiceID'] != 11092)&(Data['MiceID'] != 11095)&(Data['Maze Type'] != 'Open Field'))[0]
Data = SubDict(Data, Data.keys(), idx=idx)

dates = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']
Stage = np.unique(Data['Stage'])
mazes = np.unique(Data['Maze Type'])

idx = np.where((Data['Maze Type'] == 'Maze 1')&(Data['Stage'] == 'Stage 1')&(Data['Training Day'] == 'Day 1'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

colors = [sns.color_palette('Blues', 5)[1]]*18
totcolor = [sns.color_palette("Greys", 5)[4]]*2

fig = plt.figure(figsize = (3,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
DP1 = DPs[1]
D1 = GetDMatrices(1, 48)
for i in range(1, DP1.shape[0]):
    bins = np.array(Father2SonGraph[int(DP1[i])])
    lef, rig = np.min(D1[bins-1, 0]), np.max(D1[bins-1, 0])
    ax.fill_betweenx([0, 1], lef, rig, color = 'grey', alpha = 0.2, edgecolor=None)

sns.ecdfplot(
    x='Position',
    data=SubData,
    hue='MiceID',
    palette=colors,
    linewidth=.5,
)
sns.ecdfplot(
    x='Position',
    data=SubData,
    hue='Maze Type',
    palette=totcolor,
    linewidth=.5
)
"""
x09, y09 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10209)[0]], 1)
x12, y12 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10212)[0]], 1)
x24, y24 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10224)[0]], 1)
x27, y27 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10227)[0]], 1)
ax.plot(x09, y09, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x12, y12, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x24, y24, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x27, y27, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)

ax.plot(x09, (y09+y12+y24+y27)/4, color = 'black', linewidth=0.5)
"""
ax.plot([0, np.max(SubData['Position'])], [0,1], color = 'red', linewidth = 0.5)
print("Novel Maze A")
print(ks_2samp(SubData['Position'], np.random.rand(SubData['Position'].shape[0])*(np.max(SubData['Position'])-np.min(SubData['Position']))), end='\n\n')
#show and save
plt.tight_layout()
plt.savefig(join(loc, 'Position Distribution [Maze 1].png'), dpi = 600)
plt.savefig(join(loc, 'Position Distribution [Maze 1].svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize = (2.5,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
DP1 = DPs[1]
D1 = GetDMatrices(1, 48)
for i in range(1, DP1.shape[0]):
    bins = np.array(Father2SonGraph[int(DP1[i])])
    lef, rig = np.min(D1[bins-1, 0]), np.max(D1[bins-1, 0])
    ax.fill_betweenx([0.2, 0.3], lef, rig, color = 'grey', alpha = 0.2, edgecolor=None)

sns.ecdfplot(
    x='Position',
    data=SubData,
    hue='MiceID',
    palette=colors,
    linewidth=.5,
)
sns.ecdfplot(
    x='Position',
    data=SubData,
    hue='Maze Type',
    palette=totcolor,
    linewidth=.5
)
"""
x09, y09 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10209)[0]], 1)
x12, y12 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10212)[0]], 1)
x24, y24 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10224)[0]], 1)
x27, y27 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10227)[0]], 1)
ax.plot(x09, y09, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x12, y12, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x24, y24, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x27, y27, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x09, (y09+y12+y24+y27)/4, color = 'black', linewidth=0.5)
"""
ax.plot([0, np.max(SubData['Position'])], [0,1], color = 'red', linewidth = 0.5)
ax.set_ylim(0.2, 0.3)
ax.set_xlim(90, 130)
ax.set_xticks([90, 130])
#show and save
plt.tight_layout()
plt.savefig(join(loc, 'Position Distribution [Maze 1, Zoom out 1].png'), dpi = 600)
plt.savefig(join(loc, 'Position Distribution [Maze 1, Zoom out 1].svg'), dpi = 600)
plt.close()
#print
idx = np.where((Data['Maze Type'] == 'Maze 1')&(Data['Stage'] == 'Stage 2')&(Data['Training Day'] == '>=Day 10'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize = (3,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
DP1 = DPs[1]
D1 = GetDMatrices(1, 48)
for i in range(1, DP1.shape[0]):
    bins = np.array(Father2SonGraph[int(DP1[i])])
    lef, rig = np.min(D1[bins-1, 0]), np.max(D1[bins-1, 0])
    ax.fill_betweenx([0, 1], lef, rig, color = 'grey', alpha = 0.2, edgecolor=None)

sns.ecdfplot(
    x='Position',
    data=SubData,
    hue='MiceID',
    palette=colors,
    linewidth=.5,
)
sns.ecdfplot(
    x='Position',
    data=SubData,
    hue='Maze Type',
    palette=totcolor,
    linewidth=.5
)
"""
x09, y09 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10209)[0]], 1)
x12, y12 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10212)[0]], 1)
x24, y24 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10224)[0]], 1)
x27, y27 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10227)[0]], 1)
ax.plot(x09, y09, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x12, y12, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x24, y24, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x27, y27, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x09, (y09+y12+y24+y27)/4, color = 'black', linewidth=0.5)
"""
ax.plot([0, np.max(SubData['Position'])], [0,1], color = 'red', linewidth = 0.5)
print("Familiar Maze A")
print(ks_2samp(SubData['Position'], np.random.rand(SubData['Position'].shape[0])*(np.max(SubData['Position'])-np.min(SubData['Position']))), end='\n\n')
#show and save
plt.tight_layout()
plt.savefig(join(loc, 'Position Distribution [Maze 1, Familiar].png'), dpi = 600)
plt.savefig(join(loc, 'Position Distribution [Maze 1, Familiar].svg'), dpi = 600)
plt.close()

idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 2')&(Data['Training Day'] == 'Day 1'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize = (3,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
DP2 = DPs[2]
D2 = GetDMatrices(2, 48)
for i in range(DP2.shape[0]-1):
    bins = np.array(Father2SonGraph[int(DP2[i])])
    lef, rig = np.min(D2[bins-1, 0]), np.max(D2[bins-1, 0])
    ax.fill_betweenx([0, 1], lef, rig, color = 'grey', alpha = 0.2, edgecolor=None)

sns.ecdfplot(
    x='Position',
    data=SubData,
    hue='MiceID',
    palette=colors,
    linewidth=.5,
)
sns.ecdfplot(
    x='Position',
    data=SubData,
    hue='Maze Type',
    palette=totcolor,
    linewidth=.5
)
"""
x09, y09 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10209)[0]], 2)
x12, y12 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10212)[0]], 2)
x24, y24 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10224)[0]], 2)
x27, y27 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10227)[0]], 2)
ax.plot(x09, y09, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x12, y12, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x24, y24, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x27, y27, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x09, (y09+y12+y24+y27)/4, color = 'black', linewidth=0.5)
"""
ax.plot([0, np.max(SubData['Position'])], [0,1], color = 'red', linewidth = 0.5)
print("Novel Maze B")
print(ks_2samp(SubData['Position'], np.random.rand(SubData['Position'].shape[0])*(np.max(SubData['Position'])-np.min(SubData['Position']))), end='\n\n')

#show and save
plt.tight_layout()
plt.savefig(join(loc, 'Position Distribution [Maze 2].png'), dpi = 600)
plt.savefig(join(loc, 'Position Distribution [Maze 2].svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize = (2,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
DP2 = DPs[2]
D2 = GetDMatrices(2, 48)
for i in range(DP2.shape[0]-1):
    bins = np.array(Father2SonGraph[int(DP2[i])])
    lef, rig = np.min(D2[bins-1, 0]), np.max(D2[bins-1, 0])
    ax.fill_betweenx([0.21, 0.38], lef, rig, color = 'grey', alpha = 0.2, edgecolor=None)

sns.ecdfplot(
    x='Position',
    data=SubData,
    hue='MiceID',
    palette=colors,
    linewidth=.5,
)
sns.ecdfplot(
    x='Position',
    data=SubData,
    hue='Maze Type',
    palette=totcolor,
    linewidth=.5
)
"""
ax.plot(x09, y09, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x12, y12, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x24, y24, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x27, y27, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x09, (y09+y12+y24+y27)/4, color = 'black', linewidth=0.5)
"""
ax.plot([0, np.max(SubData['Position'])], [0,1], color = 'red', linewidth = 0.5)
ax.set_ylim(0.18, 0.4)
ax.set_xlim(90, 130)
ax.set_xticks([90, 130])
#show and save
plt.tight_layout()
plt.savefig(join(loc, 'Position Distribution [Maze 2, Zoom out 1].png'), dpi = 600)
plt.savefig(join(loc, 'Position Distribution [Maze 2, Zoom out 1].svg'), dpi = 600)
plt.close()


idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 2')&(Data['Training Day'] == '>=Day 10'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize = (3,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
DP2 = DPs[2]
D2 = GetDMatrices(2, 48)
for i in range(DP2.shape[0]-1):
    bins = np.array(Father2SonGraph[int(DP2[i])])
    lef, rig = np.min(D2[bins-1, 0]), np.max(D2[bins-1, 0])
    ax.fill_betweenx([0, 1], lef, rig, color = 'grey', alpha = 0.2, edgecolor=None)
  
sns.ecdfplot(
    x='Position',
    data=SubData,
    hue='MiceID',
    palette=colors,
    linewidth=.5,
)
sns.ecdfplot(
    x='Position',
    data=SubData,
    hue='Maze Type',
    palette=totcolor,
    linewidth=.5
)
"""
x09, y09 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10209)[0]], 2)
x12, y12 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10212)[0]], 2)
x24, y24 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10224)[0]], 2)
x27, y27 = CalCDF(SubData['Position'][np.where(SubData['MiceID'] == 10227)[0]], 2)
ax.plot(x09, y09, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x12, y12, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x24, y24, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x27, y27, color = sns.color_palette('Blues', 5)[1], linewidth=0.5)
ax.plot(x09, (y09+y12+y24+y27)/4, color = 'black', linewidth=0.5)
"""  
ax.plot([0, np.max(SubData['Position'])], [0,1], color = 'red', linewidth = 0.5)
print("Familiar Maze B")
print(ks_2samp(SubData['Position'], np.random.rand(SubData['Position'].shape[0])*(np.max(SubData['Position'])-np.min(SubData['Position']))), end='\n\n')

#show and save
plt.tight_layout()
plt.savefig(join(loc, 'Position Distribution [Maze 2 Familiar].png'), dpi = 600)
plt.savefig(join(loc, 'Position Distribution [Maze 2 Familiar].svg'), dpi = 600)
plt.close()

idx = np.where((Data['Maze Type'] == 'Maze 1')&(Data['Training Day'] == 'Day 1')&(Data['Stage'] == 'Stage 1'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
idx = np.where((BehavData['Maze Type'] == 'Maze 1')&(BehavData['Training Day'] == 'Day 1')&(BehavData['Stage'] == 'Stage 1'))[0]
BehavSubData = SubDict(BehavData, BehavData.keys(), idx=idx)
x = np.zeros((4,16))
y = np.zeros((4,16))
DP1 = DPs[1]
D = GetDMatrices(1, 48)
SubD = D[Correct_SonGraph1-1, 0]
for i in range(1, DP1.shape[0]):
    bins = np.array(Father2SonGraph[int(DP1[i])])
    lef, rig = np.min(D[bins-1, 0])-5, np.max(D[bins-1, 0])-3
    area_size = np.where((SubD >= lef)&(SubD <= rig))[0].shape[0]
    
    for j, m in enumerate([10209, 10212, 10224, 10227]):
        x[j, i-1] = np.sum(BehavSubData['Error Num'][np.where((BehavSubData['Decision Point'] == i+1)&(BehavSubData['MiceID'] == m))[0]])
        y[j, i-1] = len(np.where((SubData['Position'] <= rig)&(SubData['Position'] >= lef)&(SubData['MiceID'] == m))[0])/area_size

PlotData = {
    "Error Num": x.flatten(),
    "Field Density": scipy.stats.zscore(y, axis = 1).flatten(),
    "MiceID": np.repeat(["#10209", "#10212", "#10224", "#10227"], 16)
}
slope, intercept, r_value, p_value, std_err = linregress(x.flatten(), scipy.stats.zscore(y, axis = 1).flatten(), alternative='greater')
print(slope, intercept, r_value, p_value, std_err)
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.scatterplot(
    data=PlotData,
    x='Error Num',
    y='Field Density',
    size='Error Num',
    hue='MiceID',
    ax=ax,
    palette="rocket",
    edgecolor=None,
    sizes=(3,30) 
)
ax.plot([0, np.max(PlotData['Error Num'])], [intercept, intercept+slope*np.max(PlotData['Error Num'])], color = 'red', linewidth = 0.5)
ax.axis([-2, 60, -2, 3])
ax.set_xticks(np.linspace(0, 60, 7))
ax.set_yticks(np.linspace(-2, 3, 6))
plt.savefig(join(loc, 'Field Density vs Behav Errors [Maze 1].png'), dpi = 600)
plt.savefig(join(loc, 'Field Density vs Behav Errors [Maze 1].svg'), dpi = 600)
plt.close()


idx = np.where((Data['Maze Type'] == 'Maze 2')&(Data['Training Day'] == 'Day 1')&(Data['Stage'] == 'Stage 2'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
idx = np.where((BehavData['Maze Type'] == 'Maze 2')&(BehavData['Training Day'] == 'Day 1')&(BehavData['Stage'] == 'Stage 2'))[0]
BehavSubData = SubDict(BehavData, BehavData.keys(), idx=idx)
x = np.zeros((4,16))
y = np.zeros((4,16))
DP2 = DPs[2]
D = GetDMatrices(2, 48)
for i in range(DP2.shape[0]-1):
    bins = np.array(Father2SonGraph[int(DP2[i])])
    lef, rig = np.min(D[bins-1, 0])-5, np.max(D[bins-1, 0])-3
    SubD = D[Correct_SonGraph2-1, 0]
    area_size = np.where((SubD >= lef)&(SubD <= rig))[0].shape[0]
    
    for j, m in enumerate([10209, 10212, 10224, 10227]):
        x[j, i] = np.sum(BehavSubData['Error Num'][np.where((BehavSubData['Decision Point'] == i+1)&(BehavSubData['MiceID'] == m))[0]])
        y[j, i] = len(np.where((SubData['Position'] <= rig)&(SubData['Position'] >= lef)&(SubData['MiceID'] == m))[0])/area_size

PlotData = {
    "Error Num": x.flatten(),
    "Field Density": scipy.stats.zscore(y, axis = 1).flatten(),
    "MiceID": np.repeat(["#10209", "#10212", "#10224", "#10227"], 16)
}
slope, intercept, r_value, p_value, std_err = linregress(x.flatten(), scipy.stats.zscore(y, axis = 1).flatten(), alternative='greater')
print(slope, intercept, r_value, p_value, std_err)
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.scatterplot(
    data=PlotData,
    x='Error Num',
    y='Field Density',
    size='Error Num',
    hue='MiceID',
    ax=ax,
    palette="rocket",
    edgecolor=None,
    sizes=(3,30) 
)
ax.plot([0, np.max(PlotData['Error Num'])], [intercept, intercept+slope*np.max(PlotData['Error Num'])], color = 'black', linewidth = 0.5)
ax.axis([-2, 40, -2, 4])
ax.set_xticks(np.linspace(0, 40, 5))
ax.set_yticks(np.linspace(-2, 4, 7))
plt.savefig(join(loc, 'Field Density vs Behav Errors [Maze 2].png'), dpi = 600)
plt.savefig(join(loc, 'Field Density vs Behav Errors [Maze 2].svg'), dpi = 600)
plt.close()




idx = np.where((Data['Maze Type'] == 'Maze 1')&(Data['Training Day'] == 'Day 1')&(Data['Stage'] == 'Stage 1'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
idx = np.where((TimeData['Maze Type'] == 'Maze 1')&(TimeData['Training Day'] == 'Day 1')&(TimeData['Stage'] == 'Stage 1'))[0]
BehavSubData = SubDict(TimeData, TimeData.keys(), idx=idx)
x = np.zeros((4,16))
y = np.zeros((4,16))
DP1 = DPs[1]
D = GetDMatrices(1, 48)
SubD = D[Correct_SonGraph1-1, 0]
for i in range(1, DP1.shape[0]):
    bins = np.array(Father2SonGraph[int(DP1[i])])
    lef, rig = np.min(D[bins-1, 0])-5, np.max(D[bins-1, 0])-3
    area_size = np.where((SubD >= lef)&(SubD <= rig))[0].shape[0]
    
    for j, m in enumerate([10209, 10212, 10224, 10227]):
        x[j, i-1] = np.sum(BehavSubData['Occupation Time'][np.where((BehavSubData['Decision Point'] == i+1)&(BehavSubData['MiceID'] == m))[0]]/1000)
        y[j, i-1] = len(np.where((SubData['Position'] <= rig)&(SubData['Position'] >= lef)&(SubData['MiceID'] == m))[0])/area_size

PlotData = {
    "Occupation Time": x.flatten(),
    "Field Density": scipy.stats.zscore(y, axis = 1).flatten(),
    "MiceID": np.repeat(["#10209", "#10212", "#10224", "#10227"], 16)
}
slope, intercept, r_value, p_value, std_err = linregress(x.flatten(), scipy.stats.zscore(y, axis = 1).flatten(), alternative='greater')
print(slope, intercept, r_value, p_value, std_err)
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.scatterplot(
    data=PlotData,
    x='Occupation Time',
    y='Field Density',
    size='Occupation Time',
    hue='MiceID',
    ax=ax,
    palette="rocket",
    edgecolor=None,
    sizes=(3,30) 
)
ax.plot([0, np.max(PlotData['Occupation Time'])], [intercept, intercept+slope*np.max(PlotData['Occupation Time'])], color = 'red', linewidth = 0.5)
colors = sns.color_palette("rocket", 4)
ax.axis([-5, 130, -2, 3])
ax.set_xticks(np.linspace(0, 120, 7))
ax.set_yticks(np.linspace(-2, 3, 6))
plt.savefig(join(loc, 'Field Density vs OccuTime [Maze 1].png'), dpi = 600)
plt.savefig(join(loc, 'Field Density vs OccuTime [Maze 1].svg'), dpi = 600)
plt.close()


idx = np.where((Data['Maze Type'] == 'Maze 2')&(Data['Training Day'] == 'Day 1')&(Data['Stage'] == 'Stage 2'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
idx = np.where((TimeData['Maze Type'] == 'Maze 2')&(TimeData['Training Day'] == 'Day 1')&(TimeData['Stage'] == 'Stage 2'))[0]
BehavSubData = SubDict(TimeData, TimeData.keys(), idx=idx)
x = np.zeros((4,16))
y = np.zeros((4,16))
DP2 = DPs[2]
D = GetDMatrices(2, 48)
for i in range(DP2.shape[0]-1):
    bins = np.array(Father2SonGraph[int(DP2[i])])
    lef, rig = np.min(D[bins-1, 0])-5, np.max(D[bins-1, 0])-3
    SubD = D[Correct_SonGraph2-1, 0]
    area_size = np.where((SubD >= lef)&(SubD <= rig))[0].shape[0]
    
    for j, m in enumerate([10209, 10212, 10224, 10227]):
        x[j, i] = np.sum(BehavSubData['Occupation Time'][np.where((BehavSubData['Decision Point'] == i+1)&(BehavSubData['MiceID'] == m))[0]]/1000)
        y[j, i] = len(np.where((SubData['Position'] <= rig)&(SubData['Position'] >= lef)&(SubData['MiceID'] == m))[0])/area_size

PlotData = {
    "Occupation Time": x.flatten(),
    "Field Density": scipy.stats.zscore(y, axis = 1).flatten(),
    "MiceID": np.repeat(["#10209", "#10212", "#10224", "#10227"], 16)
}
slope, intercept, r_value, p_value, std_err = linregress(x.flatten(), scipy.stats.zscore(y, axis = 1).flatten(), alternative='greater')
print(slope, intercept, r_value, p_value, std_err)
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.scatterplot(
    data=PlotData,
    x='Occupation Time',
    y='Field Density',
    size='Occupation Time',
    hue='MiceID',
    ax=ax,
    palette="rocket",
    edgecolor=None,
    sizes=(3,30) 
)
ax.plot([0, np.max(PlotData['Occupation Time'])], [intercept, intercept+slope*np.max(PlotData['Occupation Time'])], color = 'black', linewidth = 0.5)
ax.axis([-3, 30, -2, 4])
ax.set_xticks(np.linspace(0, 30, 4))
ax.set_yticks(np.linspace(-2, 4, 7))
plt.savefig(join(loc, 'Field Density vs OccuTime [Maze 2].png'), dpi = 600)
plt.savefig(join(loc, 'Field Density vs OccuTime [Maze 2].svg'), dpi = 600)
plt.close()