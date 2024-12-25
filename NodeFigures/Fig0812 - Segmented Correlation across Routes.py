from mylib.statistic_test import *

code_id = '0812 - Segmented Correlation across Routes'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Segments', 'Mean PVC', 'Compare Groups', 'Routes'],
                              f=f2, file_idx=np.where(f2['MiceID'] != 10209)[0],
                              function = SegmentedCorrelationAcrossRoutes_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

idx = np.where(((Data['Routes'] == 3) | (Data['Routes'] == 6) | (Data['Routes'] == 0)) & (Data['Segments'] >= 75))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(3,2)),
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Segments',
    y = 'Mean PVC',
    data = SubData,
    hue = 'Routes',
    palette=[DSPPalette[0], DSPPalette[3], DSPPalette[6]],
    hue_order=[0, 3, 6],
    linewidth=0.5,
    ax = ax,
    err_kws={'edgecolor':None}
)
ax = plot_segments(ax, dy=0.8)
ax.set_ylim([0, 0.6])
ax.set_yticks(np.linspace(0, 0.6, 7))
ax.set_xlim(75, 112)
ax.set_xticks(np.linspace(75, 112.5, 7))
plt.savefig(join(loc, 'pvc - R4&7.svg'), dpi = 600)
plt.savefig(join(loc, 'pvc - R4&7.png'), dpi = 600)
plt.close()

idx = np.where(((Data['Routes'] == 3) | (Data['Routes'] == 6) | (Data['Routes'] == 0)) & (Data['Segments'] >= 75) & (Data['Training Day'] == 'Day 1'))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(3,2)),
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Segments',
    y = 'Mean PVC',
    data = SubData,
    hue = 'Routes',
    palette=[DSPPalette[0], DSPPalette[3], DSPPalette[6]],
    hue_order=[0, 3, 6],
    linewidth=0.5,
    ax = ax,
    err_kws={'edgecolor':None}
)
ax = plot_segments(ax, dy=0.8)
ax.set_ylim([-0.1, 0.7])
ax.set_yticks(np.linspace(-0.1, 0.7, 9))
ax.set_xlim(75, 112)
ax.set_xticks(np.linspace(75, 112.5, 7))
plt.savefig(join(loc, 'pvc - R4&7 [First Day].svg'), dpi = 600)
plt.savefig(join(loc, 'pvc - R4&7 [First Day].png'), dpi = 600)
plt.close()

idx = np.where(((Data['Routes'] == 3) | (Data['Routes'] == 6) | (Data['Routes'] == 0)) & (Data['Segments'] >= 75) & (Data['Training Day'] == 'Day 7'))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(3,2)),
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Segments',
    y = 'Mean PVC',
    data = SubData,
    hue = 'Routes',
    palette=[DSPPalette[0], DSPPalette[3], DSPPalette[6]],
    hue_order=[0, 3, 6],
    linewidth=0.5,
    ax = ax,
    err_kws={'edgecolor':None}
)
ax = plot_segments(ax, dy=0.8)
ax.set_ylim([-0.1, 0.6])
ax.set_yticks(np.linspace(-0.1, 0.6, 8))
ax.set_xlim(75, 112)
ax.set_xticks(np.linspace(75, 112.5, 7))
plt.savefig(join(loc, 'pvc - R4&7 [Last Day].svg'), dpi = 600)
plt.savefig(join(loc, 'pvc - R4&7 [Last Day].png'), dpi = 600)
plt.close()

idx = np.where(((Data['Routes'] != 3) & (Data['Routes'] != 6)))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(3,2)),
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Segments',
    y = 'Mean PVC',
    data = SubData,
    hue = 'Routes',
    palette=[DSPPalette[0], DSPPalette[1], DSPPalette[2], DSPPalette[4], DSPPalette[5]],
    hue_order=[0, 1, 2, 4, 5],
    linewidth=0.5,
    ax = ax,
    err_kws={'edgecolor':None}
)
ax = plot_segments(ax, dy=0.8)
ax.set_ylim([0, 0.8])
ax.set_yticks(np.linspace(0, 0.8, 5))
ax.set_xlim(0, 112.5)
ax.set_xticks(np.linspace(0, 112.5, 10))
plt.savefig(join(loc, 'pvc.svg'), dpi = 600)
plt.savefig(join(loc, 'pvc.png'), dpi = 600)
plt.close()


# Initialization: from 0812 - Segmented Correlation across Routes [stats].xlsx
# Route 2: 12 bin
# Route 3: 11 bin
# Route 5: 6 bin
# Route 6: 18 bins
# Statistical test
StatRes = {
    "Segments": [],
    "Comparison": [],
    "P-value": []
}
for i in range(111):
    idx0 = np.where(
        (Data['Segments'] == i) &
        (Data['Routes'] == 0)
    )[0]
    
    idx1 = np.where(
        (Data['Segments'] == i) &
        (Data['Routes'] == 1)
    )[0]
    
    idx2 = np.where(
        (Data['Segments'] == i) &
        (Data['Routes'] == 2)
    )[0]
    
    idx3 = np.where(
        (Data['Segments'] == i) &
        (Data['Routes'] == 3)
    )[0]
    
    idx4 = np.where(
        (Data['Segments'] == i) &
        (Data['Routes'] == 4)
    )[0]
    
    idx5 = np.where(
        (Data['Segments'] == i) &
        (Data['Routes'] == 5)
    )[0]
    
    idx6 = np.where(    
        (Data['Segments'] == i) &
        (Data['Routes'] == 6)
    )[0]
    
    for j, idx in enumerate([idx1, idx2, idx3, idx4, idx5, idx6]):
        _, p = ttest_ind(Data['Mean PVC'][idx0], Data['Mean PVC'][idx])
        
        StatRes["Segments"].append(i)
        StatRes["Comparison"].append(j+1)
        StatRes["P-value"].append(p)

for k in StatRes.keys():
    StatRes[k] = np.array(StatRes[k])
    
StatResD = pd.DataFrame(StatRes)
StatResD.to_excel(join(figdata, code_id+' [stats].xlsx'), index=False)