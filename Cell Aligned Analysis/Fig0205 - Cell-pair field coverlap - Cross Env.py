from mylib.statistic_test import *

code_id = '0205 - Cross Env field coverlap'
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = {
        "MiceID": np.array([]),
        "date": np.array([]),
        "Training Day": np.array([]),
        "Stage": np.array([]),
        "Overlap A-B": np.array([]),
        "Overlap B-A": np.array([])
    }
    
    for i in tqdm(range(len(f_CellReg_env))):
        if f_CellReg_env['include'][i] == 0 or f_CellReg_env['Stage'][i] != 'Stage 2':
            continue
        else:
            index_map = ReadCellReg(join(f_CellReg_env['cellreg_folder'][i], 'cellregistered.mat'))
            
            idx = np.where((f1['MiceID'] == f_CellReg_env['MiceID'][i])&(f1['date'] == f_CellReg_env['date'][i]))[0]
            
            if len(idx)!= 4:
                print(idx)
                assert False
                
            with open(f1['Trace File'][idx[1]], 'rb') as handle:
                trace1 = pickle.load(handle)
                
            with open(f1['Trace File'][idx[2]], 'rb') as handle:
                trace2 = pickle.load(handle)
                
            temp = np.where(index_map!= 0, 1, 0)
            cellpair_idx = np.where((temp[1, :] == 1) & (temp[2, :] == 1))[0]
            index_map = index_map[:, cellpair_idx].astype(np.int64)
            
            cellpair_idx = np.where((trace1['LA']['place_field_num'][index_map[1, :]-1] > 0)&
                                    (trace2['LA']['place_field_num'][index_map[2, :]-1] > 0))[0]
            index_map = index_map[:, cellpair_idx]
            index_map = index_map[1:3, :]
            
            overlap_num1, overlap_num2 = 0, 0
            field_num1, field_num2 = 0, 0
            
            for ce in range(index_map.shape[1]):
                for j, k in enumerate(trace1['LA']['place_field_all'][index_map[0, ce]-1]):
                    field1 = trace1['LA']['place_field_all'][index_map[0, ce]-1][k]
                    field_num1 += 1
                    for k2 in trace2['LA']['place_field_all'][index_map[1, ce]-1]:
                        field2 = trace2['LA']['place_field_all'][index_map[1, ce]-1][k2]
                        if len(np.intersect1d(field1, field2))/len(field1) >= 0.6 or len(np.intersect1d(field1, field2))/len(field2) >= 0.6:
                            overlap_num1 += 1
                            break
                        
                for j, k in enumerate(trace2['LA']['place_field_all'][index_map[1, ce]-1]):
                    field2 = trace2['LA']['place_field_all'][index_map[1, ce]-1][k]
                    field_num2 += 1
                    for k1 in trace1['LA']['place_field_all'][index_map[0, ce]-1]:
                        field1 = trace1['LA']['place_field_all'][index_map[0, ce]-1][k1]
                        if len(np.intersect1d(field1, field2))/len(field1) >= 0.6 or len(np.intersect1d(field1, field2))/len(field2) >= 0.6:
                            overlap_num2 += 1
                            break
            
            Data['Overlap A-B'] = np.concatenate([Data['Overlap A-B'], [overlap_num1/field_num1]])
            Data['Overlap B-A'] = np.concatenate([Data['Overlap B-A'], [overlap_num2/field_num2]])

            Data['Stage'] = np.concatenate([Data['Stage'], np.repeat(f_CellReg_env['Stage'][i], 1)])
            Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(f_CellReg_env['MiceID'][i], 1)])
            Data['date'] = np.concatenate([Data['date'], np.repeat(f_CellReg_env['date'][i], 1)])
            Data['Training Day'] = np.concatenate([Data['Training Day'], np.repeat(f_CellReg_env['Training Day'][i], 1)])
            
            
    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
            
    D = pd.DataFrame(Data)
    D.to_excel(os.path.join(figdata, code_id+'.xlsx'), index=False)
        
        
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

idx = np.where((Data['MiceID'] != 11095)&(Data['MiceID'] != 11092))[0]
Data = SubDict(Data, Data.keys(), idx=idx)

print_estimator(Data['Overlap A-B'])
print_estimator(Data['Overlap B-A'])