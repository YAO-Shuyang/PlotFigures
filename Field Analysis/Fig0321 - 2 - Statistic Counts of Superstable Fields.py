from mylib.statistic_test import *

code_id = '0321 - Fraction of Superstable Fields changes over time'
loc = join(figpath, code_id)
mkdir(loc)

f = pd.read_excel(join(figdata, "0300 - Kinetic Model Simulation.xlsx"))

idx = np.where((f_CellReg_modi['paradigm'] == 'CrossMaze')&(f_CellReg_modi['maze_type'] != 0))[0]
print(idx)
if os.path.exists(join(figdata, code_id+' [real data].pkl')):
    with open(join(figdata, code_id+' [real data].pkl'), 'rb') as handle:
        RData = pickle.load(handle)
else:
    RData = DataFrameEstablish(variable_names = [
                        'Duration', 'Superstable Frac.', 'Threshold', 'Drift Model'], 
                        f = f_CellReg_modi, file_idx=idx,
                        function = Superstable_Fraction_Data_Interface, 
                        file_name = code_id + ' [real data]', 
                        behavior_paradigm = 'CrossMaze')
    
def EDM_based_breakpoints(x, r):
    return np.power(r, x)

def CDM_based_breakpoints(x, k, b):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        d = np.arange(1, x[i])
        y[i] = np.prod(1-np.exp(-k*(d-b)))
        
    return y

def CDM_based_breakpoints_with_bottom(x, k, b, c):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        d = np.arange(1, x[i])
        y[i] = np.prod(1-np.exp(-k*(d-b))-c)
        
    return y

def compute_micewise_residual(RData):
    res = {"MiceID": [], "Maze Type": [], "R2": [], "Model Fitted": []}
    for mice in [10209, 10212, 10224, 10227]:
        for maze in ['Maze 1', 'Maze 2']:
            idx = np.where((RData['MiceID'] == mice)&
                           (RData['Maze Type'] == maze)&
                           (RData['Threshold'] - RData['Duration'] == 0)&
                           (np.isnan(RData['Superstable Frac.']) == False))[0]
            print(mice, maze, '-------------------------------------------------------')
            params_edm1, coe1 = curve_fit(EDM_based_breakpoints, RData['Duration'][idx], RData['Superstable Frac.'][idx])
            print(params_edm1, coe1)
            params_cdm, coe2 = curve_fit(CDM_based_breakpoints, RData['Duration'][idx], RData['Superstable Frac.'][idx])
            print(params_cdm, coe2)
            params_cdm2, coe3 = curve_fit(CDM_based_breakpoints_with_bottom, RData['Duration'][idx], RData['Superstable Frac.'][idx], bounds=[[0, -np.inf, 0], [np.inf, 1, 1]])
            print(params_cdm2, coe3)
            x = RData['Duration'][idx]
            y_edm = EDM_based_breakpoints(x, params_edm1[0])
            y_cdm = CDM_based_breakpoints(x, params_cdm[0], params_cdm[1])
            y_cdm2 = CDM_based_breakpoints_with_bottom(x, params_cdm2[0], params_cdm2[1], params_cdm2[2])
            print("EDM:", 1-np.sum((RData['Superstable Frac.'][idx] - y_edm)**2) / np.sum((np.mean(RData['Superstable Frac.'][idx]) - y_edm)**2))
            print("CDM:", 1-np.sum((RData['Superstable Frac.'][idx] - y_cdm)**2) / np.sum((np.mean(RData['Superstable Frac.'][idx]) - y_cdm)**2))
            print("CDM2:", 1-np.sum((RData['Superstable Frac.'][idx] - y_cdm2)**2) / np.sum((np.mean(RData['Superstable Frac.'][idx]) - y_cdm2)**2))
            
            res['Maze Type'] += [maze]*3
            res['MiceID'] += [mice]*3
            res['R2'] += [1-np.sum((RData['Superstable Frac.'][idx] - y_edm)**2) / np.sum((np.mean(RData['Superstable Frac.'][idx]) - y_edm)**2),
                          1-np.sum((RData['Superstable Frac.'][idx] - y_cdm)**2) / np.sum((np.mean(RData['Superstable Frac.'][idx]) - y_cdm)**2),
                          1-np.sum((RData['Superstable Frac.'][idx] - y_cdm2)**2) / np.sum((np.mean(RData['Superstable Frac.'][idx]) - y_cdm2)**2)]
            res['Model Fitted'] += ['EDM', 'CDM', 'Mixed']
            print()
    
    for k in res.keys():
        res[k] = np.array(res[k])
    return res

res = compute_micewise_residual(RData)

fig = plt.figure(figsize=(3,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x = 'Model Fitted',
    y = 'R2',
    data=res,
    ax=ax
)
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, 'R2 of Breakpoints Fitted.png'), dpi = 600)
plt.savefig(join(loc, 'R2 of Breakpoints Fitted.svg'), dpi = 600)
plt.show()

idx1 = np.where((RData['Maze Type'] == 'Maze 1')&(RData['Threshold'] - RData['Duration'] == 0)&(RData['Duration'] <= 13))[0]
idx2 = np.where((RData['Maze Type'] == 'Maze 2')&(RData['Threshold'] - RData['Duration'] == 0))[0]

params_edm1, _ = curve_fit(EDM_based_breakpoints, RData['Duration'][idx1], RData['Superstable Frac.'][idx1])
print(params_edm1)
params_cdm, _ = curve_fit(CDM_based_breakpoints, RData['Duration'][idx1], RData['Superstable Frac.'][idx1])
print(params_cdm)
params_cdm2, _ = curve_fit(CDM_based_breakpoints_with_bottom, RData['Duration'][idx1], RData['Superstable Frac.'][idx1])
print(params_cdm2)

x = np.arange(3, 27)
y_edm = EDM_based_breakpoints(x, params_edm1[0])
y_cdm = CDM_based_breakpoints(x, params_cdm[0], params_cdm[1])
y_cdm2 = CDM_based_breakpoints_with_bottom(x, params_cdm2[0], params_cdm2[1], params_cdm2[2])

fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x, y_edm, linewidth = 0.5)
ax.plot(x, y_cdm, linewidth = 0.5)
ax.plot(x, y_cdm2, linewidth = 0.5)
sns.lineplot(
    x=RData['Duration'][idx1],
    y=RData['Superstable Frac.'][idx1],
    ax=ax,
    hue=RData['Maze Type'][idx1],
    palette='dark',
    legend=False
)
plt.show()