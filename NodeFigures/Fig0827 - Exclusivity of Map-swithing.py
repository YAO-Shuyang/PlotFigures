from mylib.statistic_test import *

code_id = '0827 - Exclusivity of Map-swithing'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ["X", "Y", "P"],
                              f=f2, 
                              function = Exclusivity_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
        
fig = plt.figure(figsize = (3, 3))
ax = Clear_Axes(plt.axes())
P = np.zeros((10, 10))

for i in range(10):
    for j in range(10):
        idx = np.where((Data['X'] == i) & (Data['Y'] == j))[0]
        if len(idx) > 0:
            P[i, j] = np.nanmean(Data['P'][idx])
            
P[(np.arange(10), np.arange(10))] = np.nan            
sns.heatmap(
    P, ax=ax, vmin=0.3, vmax=0.8
)
ax.set_aspect("equal")
plt.savefig(os.path.join(loc, f'allmice.svg'), dpi=600)
plt.savefig(os.path.join(loc, f'allmice.png'), dpi=600)
plt.close()

idx1 = np.where((Data['X'] == 3) & (Data['Y'] != 3) & (Data['Y'] != 8))[0]
print_estimator(Data['P'][idx1])
idx2 = np.where((Data['X'] == 8) & (Data['Y'] != 3) & (Data['Y'] != 8))[0]
print_estimator(Data['P'][idx2])

idx3 = np.where((Data['X'] != 3) & (Data['X'] != 8) & (Data['Y'] == 3))[0]
print_estimator(Data['P'][idx3])
idx4 = np.where((Data['X'] != 3) & (Data['X'] != 8) & (Data['Y'] == 8))[0]
print_estimator(Data['P'][idx4])

idx5 = np.where((Data['X'] != 3) & (Data['X'] != 8) & (Data['Y'] != 3) & (Data['Y'] != 8))[0]
print_estimator(Data['P'][idx5])

print(f"Route 4 vs others: {ttest_ind(np.concatenate([Data['P'][idx1], Data['P'][idx3]]), Data['P'][idx5])}")
print(f"Route 7 vs others: {ttest_ind(np.concatenate([Data['P'][idx2], Data['P'][idx4]]), Data['P'][idx5])}")