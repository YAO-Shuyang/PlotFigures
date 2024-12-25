from mylib.statistic_test import *

code_id = '0816 - Segmented Correlation across Routes Under Egocentric'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

# Route 1 vs. remaining
if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Segments', 'Mean PVC', 'Compare Groups', 'Routes', 'Control For Route'],
                              f=f2, file_idx= np.where(f2['MiceID'] != 10209)[0],
                              function = SegmentedCorrelationAcrossRoutes_Egocentric_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
# Pairwise comparison across Route 2 to 7
if os.path.exists(os.path.join(figdata, code_id+' [Cross Routes].pkl')) == False:
    ArithmeticErrorData = DataFrameEstablish(variable_names = ['Segments', 'Mean PVC', 'Compare Groups', 'Control For Route'],
                              f=f2, file_idx= np.where(f2['MiceID'] != 10209)[0],
                              function = SegmentedCorrelationAcrossRoutes_Egocentric_DSP_Interface2, 
                              file_name = code_id+' [Cross Routes]', behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+' [Cross Routes].pkl'), 'rb') as handle:
        AData = pickle.load(handle)

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
    ax.set_ylim([-0.1, 0.1])
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
StatResD = pd.DataFrame(StatRes)
StatResD.to_excel(join(figdata, code_id+' [stats].xlsx'), index=False)

StatRes2 = {
    "Segments": [],
    "i": [],
    "j": [],
    "P-value": []
}
xlims = {
    (1, 2): 50,
    (1, 3): 30,
    (1, 7): 70,
    (1, 8): 40,
    (2, 3): 30,
    (2, 8): 40,
    (6, 1): 87.5,
    (6, 2): 50,
    (6, 3): 30,
    (6, 7): 70,
    (6, 8): 40,
    (7, 2): 50,
    (7, 3): 30,
    (7, 8): 40,
    (8, 3): 30
}
for u, i in enumerate([6, 1, 7, 2, 8, 3]):
    for v, j in enumerate([6, 1, 7, 2, 8, 3]):
        if u >= v:
            continue
        idx = np.where(AData['Compare Groups'] == f'{i}-{j}')[0]
        SubData = SubDict(AData, AData.keys(), idx)
        
        fig = plt.figure(figsize=(5,2)),
        ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        sns.lineplot(
            x = 'Segments',
            y = 'Mean PVC',
            data = SubData,
            hue='Control For Route',
            linewidth=0.5,
            ax = ax,
            err_kws={'edgecolor':None},
            palette=['#231815', '#A4C096']

        )
        ax.set_xticks(np.linspace(0, 100, 9))
        ax.set_xlim([0, xlims[(i, j)]])
        ax.set_title(f'{i}-{j}')
        
        for d in range(111):
            idx0 = np.where((SubData['Segments'] == d)&(SubData['Control For Route'] == 'Real'))[0]
            idx1 = np.where((SubData['Segments'] == d)&(SubData['Control For Route'] == 'Control'))[0]
        
            _, p = ttest_ind(SubData['Mean PVC'][idx0], SubData['Mean PVC'][idx1], alternative="greater")
            StatRes2['Segments'].append(d)
            StatRes2['i'].append(i)
            StatRes2['j'].append(j)
            StatRes2['P-value'].append(p)
        plt.savefig(join(loc, f'[Cross Route 2-7] pvc - {i}-{j}.png'), dpi = 600)
        plt.savefig(join(loc, f'[Cross Route 2-7] pvc - {i}-{j}.svg'), dpi = 600)
        plt.close()

fig = plt.figure(figsize=(2,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'],  ifxticks=True, ifyticks=True)
SubData = SubDict(AData, AData.keys(), np.where(AData['Segments'] <= 5)[0])
ax.axhline(0, ls=':', linewidth = 0.5, color='k')
box = sns.boxplot(
    x='Training Day',
    y='Mean PVC',
    data=SubData,
    hue='Control For Route',
    palette='Blues',
    ax=ax,
    linecolor='k',
    linewidth=0.5,
    fliersize=1,
    width=0.7,
    gap=0.2
)
for b in box.patches:
    b.set_linewidth(0)

ax.set_ylim(-0.2, 0.8)
for i in range(7):
    res = ttest_ind(
        SubData['Mean PVC'][(SubData['Control For Route'] == 'Real') & (SubData['Training Day'] == f"Day {i+1}")], 
        SubData['Mean PVC'][(SubData['Control For Route'] == 'Control') & (SubData['Training Day'] == f"Day {i+1}")]
    )
    print(f"Day {i+1}: {res}")
plt.savefig(join(loc, "Barplots.png"), dpi=600)
plt.savefig(join(loc, "Barplots.svg"), dpi=600)
plt.show()
    
StatResD2 = pd.DataFrame(StatRes2)
StatResD2.to_excel(join(figdata, code_id+' [stats2].xlsx'), index=False)
# Span: 0816 - Segmented Correlation across Routes Under Egocentric [stats]
# Route 2: 12 bins
# Route 3: 11 bins
# Route 5: 21 bins
# Route 6: 13 bins