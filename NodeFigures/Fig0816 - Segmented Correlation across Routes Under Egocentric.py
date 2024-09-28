from mylib.statistic_test import *

code_id = '0816 - Segmented Correlation across Routes Under Egocentric'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Segments', 'Mean PVC', 'Compare Groups', 'Routes', 'Control For Route'],
                              f=f2, 
                              function = SegmentedCorrelationAcrossRoutes_Egocentric_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

StatRes = {
    "Segments": [],
    "Routes": [],
    "P-value": []
}

xlims = [np.nan, 90, 55, 30, 110, 75, 40]
for i in range(1, 7):
    idx = np.where(Data['Control For Route'] == i)[0]
    SubData = SubDict(Data, Data.keys(), idx)
    fig = plt.figure(figsize=(5,2)),
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    sns.lineplot(
        x = 'Segments',
        y = 'Mean PVC',
        data = SubData,
        hue = 'Routes',
        palette=[DSPPalette[0], DSPPalette[i]],
        linewidth=0.5,
        ax = ax,
        err_kws={'edgecolor':None}
    )
    ax.set_ylim([-0.1, 0.15])
    ax.set_xticks(np.linspace(0, 100, 9))
    ax.set_xlim([0, xlims[i]])
    plt.savefig(join(loc, f'pvc - {i}.svg'), dpi = 600)
    plt.savefig(join(loc, f'pvc - {i}.png'), dpi = 600)
    plt.close()
    
    for j in range(111):
        idx0 = np.where(
            (Data['Segments'] == j) &
            (Data['Routes'] == 0)
        )[0]
        
        idx1 = np.where(
            (Data['Segments'] == j) &
            (Data['Routes'] == i)
        )[0]
        
        _, p = ttest_ind(Data['Mean PVC'][idx0], Data['Mean PVC'][idx1], alternative="less")
        StatRes['Segments'].append(j)
        StatRes['Routes'].append(i)
        StatRes['P-value'].append(p)

for k in StatRes.keys():
    StatRes[k] = np.array(StatRes[k])

# Span: 0816 - Segmented Correlation across Routes Under Egocentric [stats]
# Route 2: 12 bins
# Route 3: 11 bins
# Route 5: 21 bins
# Route 6: 13 bins

StatResD = pd.DataFrame(StatRes)
StatResD.to_excel(join(figdata, code_id+' [stats].xlsx'), index=False)

xlims = [np.nan, 90, 55, 30, 110, 75, 40]
for i in range(1, 7):
    idx = np.where(StatRes['Routes'] == i)[0]
    SubData = SubDict(StatRes, StatRes.keys(), idx)
    fig = plt.figure(figsize=(5,2)),
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.plot(
        SubData['Segments'],
        SubData['P-value'],
        color = '#E5989B',
        linewidth = 0.5
    )
    ax.axhline(0.05, color = 'k', linewidth = 0.5)
    ax.axhline(0.01, color = 'k', linewidth = 0.5)
    ax.axhline(0.001, color = 'k', linewidth = 0.5)
    ax.axhline(0.0001, color = 'k', linewidth = 0.5)
    plt.semilogy()
    ax.set_ylim(1*10**-5, 1)
    ax.set_xticks(np.linspace(0, 100, 9))
    ax.set_xlim([0, xlims[i]])
    plt.savefig(join(loc, f'pvc - {i} [P-value].svg'), dpi = 600)
    plt.savefig(join(loc, f'pvc - {i} [P-value].png'), dpi = 600)
    plt.close()