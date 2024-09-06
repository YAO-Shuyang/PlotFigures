from mylib.statistic_test import *
from mylib.dsp.route_decode import compute_accuracy

code_id = '0820 - Decode Routes From Neural Activity'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = {
        "X": [],
        "Route": [],
        "Accuracy": [],
        "MiceID": [],
        "Training Day": [],
        "Date": [],
        "Data Type": [] # Experimental or Control
    }
    
    for i in tqdm(range(len(f2))):
        with open(join(f2['Path'][i], "route_decode.pkl"), 'rb') as handle:
            res = pickle.load(handle)
        
        for seg in range(7):
            x_test_converted = spike_nodes_transform(res['x_test'], nx = 12)
            idx = np.where((res['decode_type'] == 0)&(res['segments'] == seg))[0] # Real data
            accuracy = compute_accuracy(res['y_pred'][idx], res['y_test'][idx], x_test_converted[idx])
            idx_ctrl = np.where((res['decode_type'] == 1)&(res['segments'] == seg))[0]
            accuracy_ctrl = compute_accuracy(
                res['y_pred'][np.random.permutation(idx_ctrl)], 
                res['y_test'][idx_ctrl], x_test_converted[idx_ctrl]
            )
        
            for j in range(7):
                Data['X'].append(np.arange(accuracy.shape[1]))
                Data['Route'].append(np.repeat(j, accuracy.shape[1]))
                Data['Accuracy'].append(accuracy[j, :])
                Data['MiceID'].append(np.repeat(f2['MiceID'][i], accuracy.shape[1]))
                Data['Training Day'].append(np.repeat(f2['training_day'][i], accuracy.shape[1])) 
                Data['Date'].append(np.repeat(f2['date'][i], accuracy.shape[1]))
                Data['Data Type'].append(np.repeat("Exp.", accuracy.shape[1]))
           
                Data['X'].append(np.arange(accuracy_ctrl.shape[1]))
                Data['Route'].append(np.repeat(j, accuracy_ctrl.shape[1]))
                Data['Accuracy'].append(accuracy_ctrl[j, :])
                Data['MiceID'].append(np.repeat(f2['MiceID'][i], accuracy_ctrl.shape[1]))
                Data['Training Day'].append(np.repeat(f2['training_day'][i], accuracy_ctrl.shape[1])) 
                Data['Date'].append(np.repeat(f2['date'][i], accuracy_ctrl.shape[1]))
                Data['Data Type'].append(np.repeat("Ctrl.", accuracy_ctrl.shape[1]))
    
    for k in Data.keys():
        Data[k] = np.concatenate(Data[k])
    
    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
    
    D = pd.DataFrame(Data)
    D.to_excel(os.path.join(figdata, code_id+'.xlsx'), index=False)
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
      
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.where(Data['Data Type'] == 'Exp.')[0]
SubData = SubDict(Data, Data.keys(), idx)
    
sns.lineplot(
    x = 'X',
    y = 'Accuracy',
    hue = 'Route',
    palette= DSPPalette,
    data = SubData,
    linewidth=0.5,
    err_kws={'edgecolor':None},
    ax = ax
)
    
ax.set_xticks(np.linspace(0, 110, 12))
ax.set_xlim([0, 111])
ax.set_ylim([0, 1.03])
ax.set_yticks(np.linspace(0, 1, 11))
    
plt.savefig(join(loc, f'accuracy [All].svg'), dpi = 600)
plt.savefig(join(loc, f'accuracy [All].png'), dpi = 600)
ax.clear()
        
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
for i in range(7):
    idx = np.where(Data['Route'] == i)[0]
    SubData = SubDict(Data, Data.keys(), idx)
    
    sns.lineplot(
        x = 'X',
        y = 'Accuracy',
        hue = 'Data Type',
        palette= [DSPPalette[i], '#003366'],
        data = SubData,
        linewidth=0.5,
        err_kws={'edgecolor':None},
        ax = ax
    )
    
    ax.set_xticks(np.linspace(0, 110, 12))
    ax.set_xlim([0, 110])
    ax.set_ylim([0, 1])
    ax.set_yticks(np.linspace(0, 1, 11))
    
    plt.savefig(join(loc, f'accuracy - Route {i+1}.svg'), dpi = 600)
    plt.savefig(join(loc, f'accuracy - Route {i+1}.png'), dpi = 600)
    ax.clear()

plt.close()