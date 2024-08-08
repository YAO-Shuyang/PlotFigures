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
                        'Duration', 'Superstable Frac.', 'Threshold'], 
                        f = f,
                        function = Superstable_Fraction_Interface, 
                        file_name = code_id, f_member=['Drift Type'],
                        behavior_paradigm = 'CrossMaze')

if os.path.exists(join(figdata, code_id+' [real data].pkl')):
    with open(join(figdata, code_id+' [real data].pkl'), 'rb') as handle:
        RData = pickle.load(handle)
else:
    RData = DataFrameEstablish(variable_names = [
                        'Duration', 'Superstable Frac.', 'Threshold', 'Paradigm'], 
                        f = f_CellReg_modi, f_member=['Type'],
                        function = Superstable_Fraction_Data_Interface, 
                        file_name = code_id + ' [real data]', 
                        behavior_paradigm = 'CrossMaze')

def CDMSupSBP(theta: int):
    # The tendency of breakpoints.
    assert theta >= 2
    x = np.arange(1, theta)
    return np.prod(1-np.exp(-0.18752*(x+3.5082)))
    
def CDMSupSBP_Poly(theta: int):
    assert theta >= 2
    x = np.arange(1, theta)
    return np.prod(0.9903-1/(0.9967*x+1.06453))

x = np.arange(2,27)
y = np.zeros_like(x, dtype=np.float64)
for i in range(x.shape[0]):
    y[i] = CDMSupSBP(x[i])

y2 = np.zeros_like(x, dtype=np.float64)
for i in range(x.shape[0]):
    y2[i] = CDMSupSBP_Poly(x[i])
    
# Real Data
idx = np.where((RData['Maze Type'] == 'Maze 1')&
               (RData['MiceID'] == 10227)&
               (RData['Type'] == 'Real')&
               (RData['Paradigm'] == 'CrossMaze'))[0]
SubData = SubDict(RData, RData.keys(), idx)
idx = np.where(SubData['Duration'] - SubData['Threshold'] == 0)[0]
print(SubData)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x, SubData['Superstable Frac.'][idx], ls=':', color='k', linewidth=0.5)
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
ax.set_xlim(0, 26)
ax.set_xticks(np.arange(1, 27))
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, "[Maze 1 10227] Real Superstable Frac.over time.png"), dpi=600)
plt.savefig(join(loc, "[Maze 1 10227] Real Superstable Frac.over time.svg"), dpi=600)
plt.show()

idx = np.where((RData['Maze Type'] == 'Maze 1')&
               (RData['MiceID'] == 10224)&
               (RData['Type'] == 'Real')&
               (RData['Paradigm'] == 'CrossMaze'))[0]
SubData = SubDict(RData, RData.keys(), idx)
idx = np.where(SubData['Duration'] - SubData['Threshold'] == 0)[0]
print(SubData)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x, SubData['Superstable Frac.'][idx], ls=':', color='k', linewidth=0.5)
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
ax.set_xlim(0, 26)
ax.set_xticks(np.arange(1, 27))
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, "[Maze 1 10224] Real Superstable Frac.over time.png"), dpi=600)
plt.savefig(join(loc, "[Maze 1 10224] Real Superstable Frac.over time.svg"), dpi=600)
plt.show()

"""
# Real Data&(RData['MiceID'] != 10224)
idx = np.where(
    (RData['Maze Type'] == 'Maze 1')&
    (RData['Threshold'] <= 13)&
    (RData['Duration'] <= 13)&
    (RData['Type'] == 'Real')&
    (RData['Paradigm'] == 'CrossMaze')
)[0]
SubData = SubDict(RData, RData.keys(), idx)
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
ax.set_ylim(-0.03, 0.8)
ax.set_yticks(np.linspace(0, 0.8, 5))
plt.savefig(join(loc, "[Maze 1] Real Superstable Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "[Maze 1] Real Superstable Frac. over time.svg"), dpi=600)
plt.show()

idx = np.where(
    (RData['Maze Type'] == 'Maze 2')&
    (RData['Threshold'] <= 13)&
    (RData['Duration'] <= 13)&
    (RData['Type'] == 'Real')&
    (RData['Paradigm'] == 'CrossMaze'))[0]
SubData = SubDict(RData, RData.keys(), idx)
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
ax.set_ylim(-0.03, 0.8)
ax.set_yticks(np.linspace(0, 0.8, 5))
plt.savefig(join(loc, "[Maze 2] Real Superstable Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "[Maze 2] Real Superstable Frac. over time.svg"), dpi=600)
plt.show()   

idx = np.where((Data['Drift Type'] == 'Converged')&(Data['Threshold'] <= 13) & (Data['Duration'] <= 13))[0]
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
ax.set_xlim(0, 14)
ax.set_xticks(np.arange(1, 14))
ax.set_ylim(-0.03, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, "Converged Superstable Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "Converged Superstable Frac. over time.svg"), dpi=600)
plt.close()
"""
idx = np.where((Data['Drift Type'] == 'Converged Poly')&(Data['Threshold'] <= 13) & (Data['Duration'] <= 13))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x, y2, 'k--', linewidth = 0.5)
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
ax.set_ylim(-0.03, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, "Converged Poly Superstable Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "Converged Poly Superstable Frac. over time.svg"), dpi=600)
plt.close()


# Equal-rate Drift Model
idx = np.where(Data['Drift Type'] == 'Equal-rate')[0]
SubData = SubDict(Data, Data.keys(), idx)
fig, axes = plt.subplots(ncols=5, nrows=1, figsize=(20, 2))
thre = [0.5, 0.6, 0.7, 0.8, 0.9]
for i in range(5):
    ax = Clear_Axes(axes[i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((SubData['MiceID'] > 50+i*10)&(SubData['MiceID'] <= 60+i*10)&(SubData['Threshold'] <= 13) & (SubData['Duration'] <= 13))[0]
    print(len(idx))
    x = np.arange(0,14)
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
    ax.set_xlim(0, 14)
    ax.set_xticks(np.arange(1, 14))
    ax.set_ylim(-0.03, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, "Equal-rate SupStable Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "Equal-rate SupStable Frac. over time.svg"), dpi=600)
plt.close()