# Using NaiveBayesDecoder we can decode animal's real location from neural population data.

from mylib.statistic_test import *

code_id = '0030 - Decoding Results'
mkdir(os.path.join(figpath, code_id))

shuffle_name = '2'
f3 = pd.read_excel(r'E:\Data\Simulation_pc\Decoding_Test_Records.xlsx', sheet_name = 'Decoding_Test'+shuffle_name)
idx = np.where((f3['Comparison Figure'] == 2)&(f3['date'] >= 20220813)&(f3['date'] != 20220814))[0]
print(idx)
print(idx.shape)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')): 
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['RMSE c.l.', 'MAE c.l.', 'abHit c.l.', 'geHit c.l.'], f = f3.iloc[idx], 
                              function = NeuralDecodingResults_Interface, 
                              file_name = code_id, behavior_paradigm = 'decoding', file_idx = idx, 
                              f_member = ['RMSE', 'MAE', 'general Accuracy','absolute Accuracy'])

# Transform data to a proper state
def TransformData(data:dict, mice:list = ['11095', '11092']):
    idx = np.concatenate([np.where(data['MiceID'] == m)[0] for m in mice])
    subdata = {'RMSE':np.concatenate([data['RMSE'][idx], data['RMSE c.l.'][idx]]), 'MAE':np.concatenate([data['MAE'][idx], data['MAE c.l.'][idx]]), 
               'general Accuracy':np.concatenate([data['general Accuracy'][idx]*100, data['geHit c.l.'][idx]*100]), 
               'absolute Accuracy':np.concatenate([data['absolute Accuracy'][idx]*100, data['abHit c.l.'][idx]*100]),
               'Data Type':np.concatenate([np.repeat('Real data', len(data['RMSE'][idx])), np.repeat("Chance level", len(data['RMSE'][idx]))])}
    for k in data.keys():
        if k not in ['RMSE c.l.', 'MAE c.l.', 'abHit c.l.', 'geHit c.l.', 'RMSE', 'MAE', 'general Accuracy','absolute Accuracy']:
            subdata[k] = np.concatenate([data[k][idx], data[k][idx]])

    return subdata
        
print(Data)