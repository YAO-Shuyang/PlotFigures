from mylib.statistic_test import *

code_id = '0812 - Segmented Correlation across Routes'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Segments', 'Mean PVC', 'Compare Groups', 'Routes'],
                              f=f2, 
                              function = SegmentedCorrelationAcrossRoutes_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

"""
fig = plt.figure(figsize=(5,2)),
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Segments',
    y = 'Mean PVC',
    data = Data,
    hue = 'Routes',
    palette=DSPPalette,
    hue_order=np.arange(7),
    linewidth=0.5,
    ax = ax,
    err_kws={'edgecolor':None}
)
ax = plot_segments(ax, dy=0.8)
ax.set_ylim([0, 0.8])
plt.savefig(join(loc, 'pvc.svg'), dpi = 600)
plt.savefig(join(loc, 'pvc.png'), dpi = 600)
plt.close()
"""

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