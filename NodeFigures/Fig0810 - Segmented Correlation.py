from mylib.statistic_test import *
from mylib.calcium.dsp_ms import calc_pvc

code_id = '0810 - Segmented Correlation'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    SData = DataFrameEstablish(variable_names = ['Segments', 'Mean PVC', 'Compare Groups'],
                              f = f2, 
                              function = MazeSegmentsPVC_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        SData = pickle.load(handle)

segments = np.concatenate([seg1, seg2, seg3, seg4, seg5, seg6, seg7])
if os.path.exists(os.path.join(figdata, code_id+' [cross-day].pkl')) == False:
    Data = {
        "MiceID": [],
        "Training Day": [],
        "Interval": [],
        "Segments": [],
        "Mean PVC": []
    }
    
    for i, mouse in enumerate([10209, 10212, 10224, 10227]):
        index_map = ReadCellReg(f_CellReg_dsp['cellreg_folder'][i])
        
        file_indices = np.where(f2['MiceID'] == mouse)[0]
        D = GetDMatrices(1, 48)
        print(f"Mouse {mouse}")
        for j in tqdm(range(len(file_indices)-1)):
            for dt in range(1, len(file_indices)-j):
                with open(f2['Trace File'][file_indices[j]], 'rb') as handle:
                    trace1 = pickle.load(handle)
                
                with open(f2['Trace File'][file_indices[j+dt]], 'rb') as handle:
                    trace2 = pickle.load(handle)
                
                cell_pairs = np.where(
                    (index_map[j, :] != 0) &
                    (index_map[j+dt, :] != 0)
                )[0]
            
                indexes = index_map[:, cell_pairs].astype(np.int64)
                indexes = indexes[[j, j+dt], :]
            

                is_placecells = np.where(
                    (trace1['node 9']['is_placecell'][indexes[0, :]-1] == 1) |
                    (trace2['node 0']['is_placecell'][indexes[1, :]-1] == 1)
                )[0]

                indexes = indexes[:, is_placecells]

                son_segments = get_son_area(segments)-1
                segments_pvc = np.zeros(len(son_segments))
                for k in range(len(son_segments)):
                    segments_pvc[k], _ = pearsonr(
                        trace1['node 9']['smooth_map_all'][indexes[0, :]-1, son_segments[k]], 
                        trace2['node 0']['smooth_map_all'][indexes[1, :]-1, son_segments[k]]
                    )
                
                Data['MiceID'].append(np.repeat(mouse, len(segments_pvc)))
                Data['Training Day'].append(np.repeat(j, len(segments_pvc)))
                Data['Interval'].append(np.repeat(dt, len(segments_pvc)))
                Data['Segments'].append(D[son_segments, 0])
                Data['Mean PVC'].append(segments_pvc)
    
    for k in Data.keys():
        Data[k] = np.concatenate(Data[k])
        
    with open(os.path.join(figdata, code_id+' [cross-day].pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
        
    D = pd.DataFrame(Data)
    D.to_excel(os.path.join(figdata, code_id+' [cross-day].xlsx'), index = False)
else:
    with open(os.path.join(figdata, code_id+' [cross-day].pkl'), 'rb') as handle:
        Data = pickle.load(handle)

if os.path.exists(os.path.join(figdata, code_id+' [control].pkl')) == False:
    CtrlData = {
        "MiceID": [],
        "Segments": [],
        "Mean PVC": [],
    }
    dir_names = [
        r"E:\Data\Cross_maze\10209\20230521\session 2\trace.pkl",
        r"E:\Data\Cross_maze\10212\20230521\session 2\trace.pkl",
        r"E:\Data\Cross_maze\10224\20230930\session 2\trace.pkl",
        r"E:\Data\Cross_maze\10227\20230930\session 2\trace.pkl"
    ]
    MiceID = [10209, 10212, 10224, 10227]
    
    D = GetDMatrices(1, 48)
    for i in tqdm(range(len(dir_names))):
        with open(dir_names[i], 'rb') as handle:
            trace = pickle.load(handle)

            son_segments = get_son_area(segments)-1
            segments_pvc = np.zeros(len(son_segments))
            for k in range(len(son_segments)):
                segments_pvc[k], _ = pearsonr(
                    trace['smooth_map_fir'][:, son_segments[k]], 
                    trace['smooth_map_sec'][:, son_segments[k]]
                )
                
            CtrlData['MiceID'].append(np.repeat(MiceID[i], segments_pvc.shape[0]))
            CtrlData['Segments'].append(D[son_segments, 0])
            CtrlData['Mean PVC'].append(segments_pvc)

    for k in CtrlData.keys():
        CtrlData[k] = np.concatenate(CtrlData[k])
            
    with open(os.path.join(figdata, code_id+' [control].pkl'), 'wb') as handle:
        pickle.dump(CtrlData, handle)
        
    D = pd.DataFrame(CtrlData)
    D.to_excel(os.path.join(figdata, code_id+' [control].xlsx'), index = False)
else:
    with open(os.path.join(figdata, code_id+' [control].pkl'), 'rb') as handle:
        CtrlData = pickle.load(handle)

Dist = Data['Segments'] / np.max(Data['Segments']) * 111
Dist = Dist // 1
Data['X'] = Dist.astype(np.int64)

SDist = SData['Segments'] / np.max(SData['Segments']) * 111
SDist = SDist // 1
SData['X'] = SDist.astype(np.int64)

CtrlDist = CtrlData['Segments'] / np.max(CtrlData['Segments']) * 111
CtrlDist = CtrlDist // 1
CtrlData['X'] = CtrlDist.astype(np.int64)

idx = np.where((Data['Interval'] == 1)&(Data['MiceID'] != 10209))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize = (4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'X',
    y = 'Mean PVC',
    data = SubData,
    ax = ax
)
ax.set_ylim([0, 1])
ax = plot_segments(ax, dy=0.1)
plt.show()


fig = plt.figure(figsize = (4, 3))
idx = np.where(SData['Compare Groups'] == '0-9')[0]
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'X',
    y = 'Mean PVC',
    data = SData,
    ax = ax
)
sns.lineplot(
    x = 'X',
    y = 'Mean PVC',
    data = CtrlData,
    ax = ax
)
ax = plot_segments(ax, dy=0.1)
ax.set_ylim([0, 1])
plt.show()