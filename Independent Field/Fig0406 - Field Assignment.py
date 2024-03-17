from mylib.statistic_test import *

code_id = "0406 - Field Assignment"
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
     
if os.path.exists(join(figdata, 'Field Statistics in Cell Pool [ShuffleData].pkl')):
    with open(join(figdata, 'Field Statistics in Cell Pool [ShuffleData].pkl'), 'rb') as handle:
        ShuffleData = pickle.load(handle)
else:
    ShuffleData = {
        'Std. FSC': np.zeros_like(CellData['Std. FSC']),
        'Std. OEC': np.zeros_like(CellData['Std. OEC']),
        'Std. Size': np.zeros_like(CellData['Std. Size']),
        'Std. Length': np.zeros_like(CellData['Std. Length']),
        'Std. Rate': np.zeros_like(CellData['Std. Rate']),
        'Std. Position': np.zeros_like(CellData['Std. Position']),
        'Std. Interdistance': np.zeros_like(CellData['Std. Interdistance'])
    }

    nanidx = np.where((np.isnan(Data['FSC Stability'])==False)&(np.isnan(Data['OEC Stability'])==False))[0]
    Data = SubDict(Data, Data.keys(), nanidx)
    for i in tqdm(range(CellData['Std. FSC'].shape[0])):
        idx = np.where((Data['MiceID'] == CellData['MiceID'][i])&(Data['date'] == CellData['date'][i])&(Data['Stage'] == CellData['Stage'][i])&(Data['Maze Type'] == CellData['Maze Type'][i]))[0]
        rand_idx = np.random.choice(idx, size=int(CellData['Field Number'][i]), replace=False)
        ShuffleData['Std. FSC'][i] = np.std(Data['FSC Stability'][rand_idx])
        ShuffleData['Std. OEC'][i] = np.std(Data['OEC Stability'][rand_idx])
        ShuffleData['Std. Size'][i] = np.std(Data['Field Size'][rand_idx])
        ShuffleData['Std. Length'][i] = np.std(Data['Field Length'][rand_idx])
        ShuffleData['Std. Rate'][i] = np.std(Data['Peak Rate'][rand_idx])
        ShuffleData['Std. Position'][i] = np.std(Data['Position'][rand_idx])
        
        SortedPosition = np.sort(Data['Position'][rand_idx])
        interdistance = np.abs(np.ediff1d(SortedPosition))
        
        ShuffleData['Std. Interdistance'][i] = np.std(interdistance)

    with open(os.path.join(figdata, 'Field Statistics in Cell Pool [ShuffleData].pkl'), 'wb') as handle:
        pickle.dump(ShuffleData, handle)
        
    try:
        D = pd.DataFrame(ShuffleData)
        D.to_excel(os.path.join(figdata, 'Field Statistics in Cell Pool [ShuffleData].xlsx'), sheet_name = 'data', index = False)
    except:
        pass

if os.path.exists(os.path.join(figdata, code_id + ' [statistic test].pkl')) == False:
    StatisticData = {
        "FSC Ttest Statistics": np.zeros(len(f1), dtype=np.float64),
        "FSC Ttest P-Value": np.zeros(len(f1), dtype=np.float64),
        "FSC KS Statistics": np.zeros(len(f1), dtype=np.float64),
        "FSC KS P-Value": np.zeros(len(f1), dtype=np.float64),
        "OEC Ttest Statistics": np.zeros(len(f1), dtype=np.float64),
        "OEC Ttest P-Value": np.zeros(len(f1), dtype=np.float64),
        "OEC KS Statistics": np.zeros(len(f1), dtype=np.float64),
        "OEC KS P-Value": np.zeros(len(f1), dtype=np.float64),
        "Length Ttest Statistics": np.zeros(len(f1), dtype=np.float64),
        "Length Ttest P-Value": np.zeros(len(f1), dtype=np.float64),
        "Length KS Statistics": np.zeros(len(f1), dtype=np.float64),
        "Length KS P-Value": np.zeros(len(f1), dtype=np.float64),
        "Rate Ttest Statistics": np.zeros(len(f1), dtype=np.float64),
        "Rate Ttest P-Value": np.zeros(len(f1), dtype=np.float64),
        "Rate KS Statistics": np.zeros(len(f1), dtype=np.float64),
        "Rate KS P-Value": np.zeros(len(f1), dtype=np.float64),
    }
    
    for i in range(len(f1)):
        
        if f1['include'][i] == 0 or f1['maze_type'][i] == 0:
            for k in StatisticData.keys():
                StatisticData[k][i] = np.nan
            continue
        
        maze_type = 'Maze 1' if f1['maze_type'][i] == 1 else 'Maze 2'
        idx = np.where((CellData['Maze Type'] == maze_type)&(CellData['date'] == f1['date'][i])&(CellData['MiceID'] == f1['MiceID'][i])&(CellData['Stage'] == f1['Stage'][i]))[0]
        
        print(i, f1['MiceID'][i], f1['date'][i], f1['Stage'][i], "Maze", f1['maze_type'][i])
        StatisticData['FSC Ttest Statistics'][i], StatisticData['FSC Ttest P-Value'][i] = ttest_rel(CellData['Std. FSC'][idx], ShuffleData['Std. FSC'][idx], alternative='less')
        StatisticData['OEC Ttest Statistics'][i], StatisticData['OEC Ttest P-Value'][i] = ttest_rel(CellData['Std. OEC'][idx], ShuffleData['Std. OEC'][idx], alternative='less')
        StatisticData['Length Ttest Statistics'][i], StatisticData['Length Ttest P-Value'][i] = ttest_rel(CellData['Std. Length'][idx], ShuffleData['Std. Length'][idx], alternative='less')
        StatisticData['Rate Ttest Statistics'][i], StatisticData['Rate Ttest P-Value'][i] = ttest_rel(CellData['Std. Rate'][idx], ShuffleData['Std. Rate'][idx], alternative='less')

        StatisticData['FSC KS Statistics'][i], StatisticData['FSC KS P-Value'][i] = ks_2samp(CellData['Std. FSC'][idx], ShuffleData['Std. FSC'][idx])
        StatisticData['OEC KS Statistics'][i], StatisticData['OEC KS P-Value'][i] = ks_2samp(CellData['Std. OEC'][idx], ShuffleData['Std. OEC'][idx])
        StatisticData['Length KS Statistics'][i], StatisticData['Length KS P-Value'][i] = ks_2samp(CellData['Std. Length'][idx], ShuffleData['Std. Length'][idx])
        StatisticData['Rate KS Statistics'][i], StatisticData['Rate KS P-Value'][i] = ks_2samp(CellData['Std. Rate'][idx], ShuffleData['Std. Rate'][idx])
        
        print("FSC     ", StatisticData['FSC Ttest P-Value'][i], StatisticData['FSC Ttest Statistics'][i], StatisticData['FSC KS P-Value'][i], StatisticData['FSC KS Statistics'][i])
        print("OEC     ", StatisticData['OEC Ttest P-Value'][i], StatisticData['OEC Ttest Statistics'][i], StatisticData['OEC KS P-Value'][i], StatisticData['OEC KS Statistics'][i])
        print("Length  ", StatisticData['Length Ttest P-Value'][i], StatisticData['Length Ttest Statistics'][i], StatisticData['Length KS P-Value'][i], StatisticData['Length KS Statistics'][i])
        print("Rate    ", StatisticData['Rate Ttest P-Value'][i], StatisticData['Rate Ttest Statistics'][i], StatisticData['Rate KS P-Value'][i], StatisticData['Rate KS Statistics'][i], end='\n\n')
        
    with open(os.path.join(figdata, code_id + ' [statistic test].pkl'), 'wb') as handle:
        pickle.dump(StatisticData, handle)
        
    D = pd.DataFrame(StatisticData)
    D.to_excel(os.path.join(figdata, code_id + ' [statistic test].xlsx'), sheet_name = 'data', index = False)
else:
    with open(os.path.join(figdata, code_id + ' [statistic test].pkl'), 'rb') as handle:
        StatisticData = pickle.load(handle)
        

m1_num = np.where((np.isnan(StatisticData['FSC Ttest P-Value']) == False)&(f1['maze_type'] == 1))[0].shape[0]
m2_num = np.where((np.isnan(StatisticData['Length Ttest P-Value']) == False)&(f1['maze_type'] == 2))[0].shape[0]

m1_nodiff_fsc = np.where((StatisticData['FSC Ttest P-Value'] >= 0.05)&(f1['maze_type'] == 1))[0].shape[0]
m2_nodiff_fsc = np.where((StatisticData['FSC Ttest P-Value'] >= 0.05)&(f1['maze_type'] == 2))[0].shape[0]

m1_nodiff_size = np.where((StatisticData['Length Ttest P-Value'] >= 0.05)&(f1['maze_type'] == 1))[0].shape[0]
m2_nodiff_size = np.where((StatisticData['Length Ttest P-Value'] >= 0.05)&(f1['maze_type'] == 2))[0].shape[0]


print(m1_num, m2_num)
print(m1_nodiff_fsc, m2_nodiff_fsc)
print(m1_nodiff_size, m2_nodiff_size)


idx = np.where((CellData['Maze Type'] == 'Maze 1')&(CellData['date'] == 20230930)&(CellData['MiceID'] == 10227)&(CellData['Stage'] == 'Stage 2'))[0]
fig = plt.figure(figsize=(3, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(CellData['Std. FSC'][idx], bins=20, range=(0,1), alpha=0.5)
ax.hist(ShuffleData['Std. FSC'][idx], bins=20, range=(0,1), alpha=0.5)
plt.savefig(os.path.join(loc, '10227-20230930-Maze 1-FSC Distribution.png'), dpi = 600)
plt.savefig(os.path.join(loc, '10227-20230930-Maze 1-FSC Distribution.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(3, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(CellData['Std. Length'][idx], bins=20, range=(0,60), alpha=0.5)
ax.hist(ShuffleData['Std. Length'][idx], bins=20, range=(0,60), alpha=0.5)
plt.savefig(os.path.join(loc, '10227-20230930-Maze 1-Field Length Distribution.png'), dpi = 600)
plt.savefig(os.path.join(loc, '10227-20230930-Maze 1-Field Length Distribution.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(3, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(CellData['Std. Rate'][idx], bins=20, range=(0,4), alpha=0.5)
ax.hist(ShuffleData['Std. Rate'][idx], bins=20, range=(0,4), alpha=0.5)
plt.savefig(os.path.join(loc, '10227-20230930-Maze 1-Field Rate Distribution.png'), dpi = 600)
plt.savefig(os.path.join(loc, '10227-20230930-Maze 1-Field Rate Distribution.svg'), dpi = 600)
plt.close()

idx = np.where((CellData['Maze Type'] == 'Maze 2')&(CellData['date'] == 20230930)&(CellData['MiceID'] == 10227)&(CellData['Stage'] == 'Stage 2'))[0]
fig = plt.figure(figsize=(3, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(CellData['Std. FSC'][idx], bins=20, range=(0,1), alpha=0.5)
ax.hist(ShuffleData['Std. FSC'][idx], bins=20, range=(0,1), alpha=0.5)
plt.savefig(os.path.join(loc, '10227-20230930-Maze 2-FSC Distribution.png'), dpi = 600)
plt.savefig(os.path.join(loc, '10227-20230930-Maze 2-FSC Distribution.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(3, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(CellData['Std. Length'][idx], bins=20, range=(0,60), alpha=0.5)
ax.hist(ShuffleData['Std. Length'][idx], bins=20, range=(0,60), alpha=0.5)
plt.savefig(os.path.join(loc, '10227-20230930-Maze 2-Field Length Distribution.png'), dpi = 600)
plt.savefig(os.path.join(loc, '10227-20230930-Maze 2-Field Length Distribution.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(3, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(StatisticData['Length Ttest P-Value'][np.where(f1['maze_type'] == 1)[0]], 
        StatisticData['FSC Ttest P-Value'][np.where(f1['maze_type'] == 1)[0]], 'o', markeredgewidth=0, markersize=3, label='Maze 1')
ax.plot(StatisticData['Length Ttest P-Value'][np.where(f1['maze_type'] == 2)[0]],
        StatisticData['FSC Ttest P-Value'][np.where(f1['maze_type'] == 2)[0]], 'o', markeredgewidth=0, markersize=3, label='Maze 2')
ax.axis([0.0001, 1.05, 0.0001, 1.05])
ax.semilogx()
ax.semilogy()
ax.legend()
ax.axvline(0.05/191, color='red', ls=':', linewidth=0.5)
ax.axhline(0.05/191, color='red', ls=':', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(loc, 'Ttest P-Value [Length Stability].png'), dpi = 600)
plt.savefig(os.path.join(loc, 'Ttest P-Value [Length Stability].svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(3, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(StatisticData['Length KS P-Value'][np.where(f1['maze_type'] == 1)[0]], 
        StatisticData['FSC KS P-Value'][np.where(f1['maze_type'] == 1)[0]], 'o', markeredgewidth=0, markersize=3, label='Maze 1')
ax.plot(StatisticData['Length KS P-Value'][np.where(f1['maze_type'] == 2)[0]],
        StatisticData['FSC KS P-Value'][np.where(f1['maze_type'] == 2)[0]], 'o', markeredgewidth=0, markersize=3, label='Maze 2')
ax.axis([0.0001, 1.05, 0.0001, 1.05])
ax.semilogx()
ax.semilogy()
ax.legend()
ax.axvline(0.05/191, color='red', ls=':', linewidth=0.5)
ax.axhline(0.05/191, color='red', ls=':', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(loc, 'KS P-Value [Length Stability].png'), dpi = 600)
plt.savefig(os.path.join(loc, 'KS P-Value [Length Stability].svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(3, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(StatisticData['Rate Ttest P-Value'][np.where(f1['maze_type'] == 1)[0]], 
        StatisticData['Length Ttest P-Value'][np.where(f1['maze_type'] == 1)[0]], 'o', markeredgewidth=0, markersize=3, label='Maze 1')
ax.plot(StatisticData['Rate Ttest P-Value'][np.where(f1['maze_type'] == 2)[0]],
        StatisticData['Length Ttest P-Value'][np.where(f1['maze_type'] == 2)[0]], 'o', markeredgewidth=0, markersize=3, label='Maze 2')
ax.axis([0.0001, 1.05, 0.0001, 1.05])
ax.semilogx()
ax.semilogy()
ax.legend()
ax.axvline(0.05/191, color='red', ls=':', linewidth=0.5)
ax.axhline(0.05/191, color='red', ls=':', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(loc, 'Ttest P-Value [Length Rate].png'), dpi = 600)
plt.savefig(os.path.join(loc, 'Ttest P-Value [Length Rate].svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(3, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(StatisticData['Rate KS P-Value'][np.where(f1['maze_type'] == 1)[0]], 
        StatisticData['Length KS P-Value'][np.where(f1['maze_type'] == 1)[0]], 'o', markeredgewidth=0, markersize=3, label='Maze 1')
ax.plot(StatisticData['Rate KS P-Value'][np.where(f1['maze_type'] == 2)[0]],
        StatisticData['Length KS P-Value'][np.where(f1['maze_type'] == 2)[0]], 'o', markeredgewidth=0, markersize=3, label='Maze 2')
ax.axis([0.0001, 1.05, 0.0001, 1.05])
ax.semilogx()
ax.semilogy()
ax.axvline(0.05/191, color='red', ls=':', linewidth=0.5)
ax.axhline(0.05/191, color='red', ls=':', linewidth=0.5)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(loc, 'KS P-Value [Length Rate].png'), dpi = 600)
plt.savefig(os.path.join(loc, 'KS P-Value [Length Rate].svg'), dpi = 600)
plt.close()