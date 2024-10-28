#from mylib.statistic_test import *
#from mylib.field.tracker_v2 import Tracker2d
#from mylib.field.sfer import get_surface, get_data, fit_kww, fit_reci
from mylib.statistic_test import *

code_id = '0348 - Same position Field Coordination'
loc = join(figpath, code_id)
mkdir(loc)


def get_roi_center_dist_matrix(dir_name: str):
    files = os.listdir(dir_name)
    print(dir_name)
    dates = []
    for file in files:
        if 'SFP' in file:
            dates.append(int(file[3:11]))
    
    dates = np.array(dates)
    idx = np.argsort(dates)
    
    x_ratio = 700/250
    y_ratio = 450/160
    
    sfps = []
    for i in tqdm(idx):
        path = os.path.join(dir_name, f"SFP{dates[i]}.mat")
        with h5py.File(path, 'r') as handle:
            sfp = np.array(handle['SFP'])
            pos = np.zeros((sfp.shape[2], 2))
            for i in range(sfp.shape[2]):
                max_strength = np.max(sfp[:, :, i])
                pos[i, 0], pos[i, 1] = np.where(sfp[:, :, i] == max_strength)
            
            pos[:, 0] = pos[:, 0] * x_ratio
            pos[:, 1] = pos[:, 1] * y_ratio
            
            mat = np.zeros((sfp.shape[2], sfp.shape[2]))
            for i in range(sfp.shape[2]-1):
                for j in range(i+1, sfp.shape[2]):
                    mat[i, j] = np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2)
                    mat[j, i] = mat[i, j]
            
            sfps.append(mat)
    
    return sfps

        
idx = np.where((f_CellReg_modi['Type'] == 'Real') & (f_CellReg_modi['maze_type'] != 0))[0]
if exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(
        variable_names = ['Training Session', 'Chi-Square Statistic', 'Paradigm'],
        f = f_CellReg_modi, file_idx=idx,
        function = FieldSamePosition_Evolution_Interface, 
        file_name = code_id, 
        behavior_paradigm = 'CrossMaze'
    )

if os.path.exists(join(figdata, code_id+' [residual matrix].pkl')):
    with open(join(figdata, code_id+' [residual matrix].pkl'), 'rb') as handle:
        CData = pickle.load(handle)
else:
    CData = DataFrameEstablish(
        variable_names = ['Training Session', 'delta-P', 'Pair Type',
                        'Paradigm', 'X'],
        f = f_CellReg_modi, 
        function = compute_coordination_on_position_Interface, 
        file_name = code_id+' [residual matrix]', file_idx=idx,
        behavior_paradigm = 'CrossMaze'
    )

length = len(Data['Training Session'])
idx = np.concatenate([np.arange(0, length, 2), np.arange(1, length, 2)])
Data = SubDict(Data, Data.keys(), idx)
Data['Field Type'] = np.concatenate([np.repeat("Same Position", int(length/2)), np.repeat("Different Position", int(length/2))])

fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.boxplot(
    x="Paradigm", 
    y="Chi-Square Statistic",
    hue="Field Type",
    data=Data,
    palette=['#457B9D', '#A8DADC'],
    ax = ax,
    linecolor='black',
    linewidth=0.5,
    gap=0.2,
    flierprops={'markersize': 0.5},
)
plt.semilogy()
plt.savefig(join(loc, f'coordination.png'), dpi=600)
plt.savefig(join(loc, f'coordination.svg'), dpi=600)
plt.close()

print("Coordination")
for i in np.unique(Data['Paradigm']):
    idx1 = np.where((Data['Field Type'] == 'Same Position') & (Data['Paradigm'] == i))[0]
    idx2 = np.where((Data['Field Type'] == 'Different Position') & (Data['Paradigm'] == i))[0]
    print(i, ttest_rel(Data['Chi-Square Statistic'][idx1], Data['Chi-Square Statistic'][idx2]))
print()

idx = np.where((np.isnan(CData['delta-P']) == False))[0]
SubData = SubDict(CData, CData.keys(), idx=idx)

SubData['hue'] = np.array([SubData['Paradigm'][i]+SubData['Pair Type'][i] for i in idx])
colors = sns.color_palette("Reds", 2) + sns.color_palette("Oranges", 2) + sns.color_palette("Wistia", 2) + sns.color_palette("Greens", 2) + sns.color_palette("Blues", 2) + sns.color_palette("Purples", 2)
fig = plt.figure(figsize=(10, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.axhline(y = 0, color = 'black', linewidth = 0.5, ls=':')
box = sns.boxplot(
    x="X", 
    y="delta-P",
    hue="hue",
    data=SubData,
    palette=colors,
    ax = ax,
    linecolor='black',
    linewidth=0.5,
    gap=0.2,
    flierprops={'markersize': 0.5},
)
for line in box.patches:
    line.set_linewidth(0)  # Remove the outer box line
ax.set_xlim([-0.5, 2.5])
ax.set_ylim([-0.01, 0.01])
ax.set_yticks(np.linspace(-0.04, 0.04, 9))
plt.savefig(join(loc, '[dim=2] delt_P All Paradigm.png'), dpi = 600)
plt.savefig(join(loc, '[dim=2] delt_P All Paradigm.svg'), dpi = 600)
plt.close()

print()
for x in np.unique(SubData['X']):
    print("------------------------")
    for y in np.unique(SubData['Paradigm']):
        idx1 = np.where((SubData['X'] == x) & (SubData['Paradigm'] == y) & (SubData['Pair Type'] == 'Sibling'))[0]
        idx2 = np.where((SubData['X'] == x) & (SubData['Paradigm'] == y) & (SubData['Pair Type'] == 'Non-sibling'))[0]
        print(f"{y} - {x}:\n    {ttest_rel(SubData['delta-P'][idx1], SubData['delta-P'][idx2])}")
        print(f"    Same Pos vs 0: {ttest_1samp(SubData['delta-P'][idx1], 0)}")
        print(f"    Diff Pos vs 0: {ttest_1samp(SubData['delta-P'][idx2], 0)}", end='\n\n')

if __name__ == '__main__':
    pass
    """
    from mylib.local_path import f_CellReg_modi
    
    for i in range(len(f_CellReg_modi)):
        if f_CellReg_modi['Type'][i] != 'Real' or f_CellReg_modi['maze_type'][i] == 0:
            continue
        
        with open(f_CellReg_modi['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)  
              
        distances = get_roi_center_dist_matrix(f_CellReg_modi['sfp_folder'][i])
        
        trace['ROI_Distance'] = distances
        
        with open(f_CellReg_modi['Trace File'][i], 'wb') as handle:
            pickle.dump(trace, handle)
    """