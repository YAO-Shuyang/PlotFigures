from mylib.statistic_test import *

code_id = '0321 - Fraction of Superstable Fields changes over time'
loc = join(figpath, code_id)
mkdir(loc)

f = pd.read_excel(join(figdata, "0300 - Kinetic Model Simulation.xlsx"))

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = [
                        'Duration', 'Superstable Frac.', 'Threshold', 'Drift Model'], 
                        f = f,
                        function = Superstable_Fraction_Interface, 
                        file_name = code_id, 
                        behavior_paradigm = 'CrossMaze')

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

def CDMSupSBP(theta: int):
    # The tendency of breakpoints.
    assert theta >= 2
    x = np.arange(1, theta)
    return np.prod(1-np.exp(-0.3187*(x+3.369-2.7))-0.01)
    

x = np.arange(2,27)
y = np.zeros_like(x, dtype=np.float64)
for i in range(x.shape[0]):
    y[i] = CDMSupSBP(x[i])
    
# Real Data

 

# Real Data&(RData['MiceID'] != 10224)
idx = np.where((RData['Maze Type'] == 'Maze 1')&(RData['Threshold'] <= 13)&(RData['Duration'] <= 13))[0]
SubData = SubDict(RData, Data.keys(), idx)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
#ax.plot(x, y, 'k--', linewidth = 0.5)
idx = np.where(SubData['Duration'] - SubData['Threshold'] == 0)[0]
sns.lineplot(
    x=SubData['Duration'][idx],
    y=SubData['Superstable Frac.'][idx],
    ax=ax,
    palette='dark',
    linewidth = 0.5,
    err_kws={'edgecolor': None},    
)
sns.lineplot(
    x = 'Duration',
    y = 'Superstable Frac.',
    data=SubData,
    hue = "Threshold",
    palette='rainbow',
    linewidth = 0.5,
    err_kws={'edgecolor': None},
    ax=ax
)
ax.set_xlim(0, 13)
ax.set_xticks(np.arange(1, 14))
ax.set_ylim(-0.03, 0.75)
ax.set_yticks(np.linspace(0, 0.75, 6))
plt.savefig(join(loc, "[Maze 1] Real Superstable Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "[Maze 1] Real Superstable Frac. over time.svg"), dpi=600)
plt.show()

idx = np.where((RData['Maze Type'] == 'Maze 2')&(RData['Threshold'] <= 13))[0]
SubData = SubDict(RData, Data.keys(), idx)
idx = np.where(SubData['Duration'] - SubData['Threshold'] == 0)[0]
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
#ax.plot(x, y, 'k--', linewidth = 0.5)
sns.lineplot(
    x=SubData['Duration'][idx],
    y=SubData['Superstable Frac.'][idx],
    ax=ax,
    palette='dark',
    linewidth = 0.5,
    err_kws={'edgecolor': None},    
)
sns.lineplot(
    x = 'Duration',
    y = 'Superstable Frac.',
    data=SubData,
    hue = "Threshold",
    palette='rainbow',
    linewidth = 0.5,
    err_kws={'edgecolor': None},
    ax=ax
)
ax.set_xlim(0, 14)
ax.set_xticks(np.arange(1, 14))
ax.set_ylim(-0.02, 0.6)
ax.set_yticks(np.linspace(0, 0.6, 7))
plt.savefig(join(loc, "[Maze 2] Real Superstable Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "[Maze 2] Real Superstable Frac. over time.svg"), dpi=600)
plt.show()   

idx = np.where((Data['Drift Model'] == 'converged')&(Data['Threshold'] <= 26))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x, y, 'k--', linewidth = 0.5)
sns.lineplot(
    x = 'Duration',
    y = 'Superstable Frac.',
    data=SubData,
    hue = "Threshold",
    palette='rainbow',
    linewidth = 0.5,
    err_kws={'edgecolor': None},
    ax=ax
)
ax.set_xlim(0, 27)
ax.set_xticks(np.arange(1, 26))
ax.set_ylim(-0.03, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, "Converged Superstable Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "Converged Superstable Frac. over time.svg"), dpi=600)
plt.close()


# Equal-rate Drift Model
idx = np.where(Data['Drift Model'] == 'equal-rate')[0]
SubData = SubDict(Data, Data.keys(), idx)
fig, axes = plt.subplots(ncols=5, nrows=1, figsize=(20, 2))
thre = [0.5, 0.6, 0.7, 0.8, 0.9]
for i in range(5):
    ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((SubData['MiceID'] > 50+i*10)&(SubData['MiceID'] <= 60+i*10)&(SubData['Threshold'] <= 26))[0]
    x = np.arange(0,27)
    y = np.power(thre[i], x)
    ax.plot(x+1, y, 'k--', linewidth = 0.5)
    sns.lineplot(
        x=SubData['Duration'][idx], 
        y=SubData['Superstable Frac.'][idx],
        hue=SubData['Threshold'][idx],
        palette="rainbow",
        linewidth = 0.5,
        err_kws={'edgecolor': None},
        ax=ax
    )
    ax.set_xlim(0, 27)
    ax.set_xticks(np.arange(1, 26))
    ax.set_ylim(-0.03, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, "Equal-rate SupStable Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "Equal-rate SupStable Frac. over time.svg"), dpi=600)
plt.close()