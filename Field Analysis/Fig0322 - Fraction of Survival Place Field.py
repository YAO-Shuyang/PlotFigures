from mylib.statistic_test import *

code_id = '0322 - Fraction of Survival Place Field'
loc = join(figpath, code_id)
mkdir(loc)

f = pd.read_excel(join(figdata, "0300 - Kinetic Model Simulation.xlsx"))

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = [
                        'Session Interval', 'Start Session', 'Survival Frac.'], 
                        f = f,
                        function = SurvivalField_Fraction_Interface, 
                        file_name = code_id, f_member=['Drift Type'],
                        behavior_paradigm = 'CrossMaze')

if os.path.exists(join(figdata, code_id+' [real data].pkl')):
    with open(join(figdata, code_id+' [real data].pkl'), 'rb') as handle:
        RData = pickle.load(handle)
else:
    RData = DataFrameEstablish(variable_names = [
                        'Session Interval', 'Start Session', 'Survival Frac.', 'Paradigm'], 
                        f = f_CellReg_modi, f_member=['Type'],
                        function = SurvivalField_Fraction_Data_Interface, 
                        file_name = code_id + ' [real data]', 
                        behavior_paradigm = 'CrossMaze')

idx = np.where((RData['Maze Type'] == 'Maze 1')&
               (RData['MiceID'] == 10227)&
               (RData['Type'] == 'Real')&
               (RData['Paradigm'] == 'CrossMaze'))[0]
SubData = SubDict(RData, RData.keys(), idx)
print(SubData)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Session Interval',
    y = 'Survival Frac.',
    data=SubData,
    hue = "Start Session",
    palette='rainbow',
    linewidth = 0.5,
    err_kws={'edgecolor': None},
    ax=ax
)
ax.set_xlim(0, 26)
ax.set_xticks(np.arange(1, 27))
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, "[Maze 1 10227] Real Survival Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "[Maze 1 10227] Real Survival Frac. over time.svg"), dpi=600)
plt.show()

idx = np.where((RData['Maze Type'] == 'Maze 1')&
               (RData['MiceID'] == 10224)&
               (RData['Type'] == 'Real')&
               (RData['Paradigm'] == 'CrossMaze'))[0]
SubData = SubDict(RData, RData.keys(), idx)
print(SubData)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Session Interval',
    y = 'Survival Frac.',
    data=SubData,
    hue = "Start Session",
    palette='rainbow',
    linewidth = 0.5,
    err_kws={'edgecolor': None},
    ax=ax
)
ax.set_xlim(0, 26)
ax.set_xticks(np.arange(1, 27))
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, "[Maze 1 10224] Real Survival Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "[Maze 1 10224] Real Survival Frac. over time.svg"), dpi=600)
plt.show()

# Real Data&(RData['MiceID'] != 10224)
idx = np.where(
    (RData['Maze Type'] == 'Maze 1')&
    (RData['Session Interval'] + RData['Start Session'] <= 14)&
    (RData['Type'] == 'Real')&
    (RData['Paradigm'] == 'CrossMaze')
)[0]
SubData = SubDict(RData, RData.keys(), idx)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Session Interval',
    y = 'Survival Frac.',
    data=SubData,
    hue = "Start Session",
    palette='rainbow',
    linewidth = 0.5,
    err_kws={'edgecolor': None},
    ax=ax
)
ax.set_xlim(0, 13)
ax.set_xticks(np.arange(1, 14))
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, "[Maze 1] Real Survival Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "[Maze 1] Real Survival Frac. over time.svg"), dpi=600)
plt.show()

idx = np.where(
    (RData['Maze Type'] == 'Maze 2')&
    (RData['Session Interval'] + RData['Start Session'] <= 14)&
    (RData['Type'] == 'Real')&
    (RData['Paradigm'] == 'CrossMaze'))[0]
SubData = SubDict(RData, RData.keys(), idx)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Session Interval',
    y = 'Survival Frac.',
    data=SubData,
    hue = "Start Session",
    palette='rainbow',
    linewidth = 0.5,
    err_kws={'edgecolor': None},
    ax=ax
)
ax.set_xlim(0, 14)
ax.set_xticks(np.arange(1, 14))
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, "[Maze 2] Real Survival Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "[Maze 2] Real Survival Frac. over time.svg"), dpi=600)
plt.show()


idx = np.where((Data['Drift Type'] == 'Converged')&(Data['Session Interval'] + Data['Start Session'] <= 26))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Session Interval',
    y = 'Survival Frac.',
    data=SubData,
    hue = "Start Session",
    palette='rainbow',
    linewidth = 0.5,
    err_kws={'edgecolor': None},
    ax=ax
)
ax.set_xlim(0, 27)
ax.set_xticks(np.arange(1, 26))
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, "Converged Survival Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "Converged Survival Frac. over time.svg"), dpi=600)
plt.show()

idx = np.where((Data['Drift Type'] == 'Converged Poly')&(Data['Session Interval'] + Data['Start Session'] <= 26))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Session Interval',
    y = 'Survival Frac.',
    data=SubData,
    hue = "Start Session",
    palette='rainbow',
    linewidth = 0.5,
    err_kws={'edgecolor': None},
    ax=ax
)
ax.set_xlim(0, 27)
ax.set_xticks(np.arange(1, 26))
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(join(loc, "Converged Poly Survival Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "Converged Poly Survival Frac. over time.svg"), dpi=600)
plt.show()

# Equal-rate Drift Model
idx = np.where(Data['Drift Type'] == 'Equal-rate')[0]
SubData = SubDict(Data, Data.keys(), idx)
fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(20, 4))
thre = [0.5, 0.6, 0.7, 0.8, 0.9]
for i in range(5):
    ax = Clear_Axes(axes[0, i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((SubData['MiceID'] > 50+i*10)&(SubData['MiceID'] <= 60+i*10)&(SubData['Session Interval'] + SubData['Start Session'] <= 26))[0]
    sns.lineplot(
        x=SubData['Session Interval'][idx], 
        y=SubData['Survival Frac.'][idx],
        hue=SubData['Start Session'][idx],
        palette="rainbow",
        linewidth = 0.5,
        err_kws={'edgecolor': None},
        ax=ax
    )
    ax.set_xlim(0, 27)
    ax.set_xticks(np.arange(1, 26))
    ax.set_ylim(-0.03, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    
    ax = Clear_Axes(axes[1, i], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((SubData['MiceID'] > 50+i*10)&(SubData['MiceID'] <= 60+i*10)&(SubData['Session Interval'] + SubData['Start Session'] <= 26))[0]
    sns.lineplot(
        x=SubData['Session Interval'][idx], 
        y=SubData['Survival Frac.'][idx],
        hue=SubData['Start Session'][idx],
        palette="rainbow",
        linewidth = 0.5,
        err_kws={'edgecolor': None},
        ax=ax
    )
    ax.set_xlim(0, 27)
    ax.set_xticks(np.arange(1, 26))
    ax.semilogy()
    ax.set_ylim(0.0001, 1)
plt.savefig(join(loc, "Equal-rate Survival Frac. over time.png"), dpi=600)
plt.savefig(join(loc, "Equal-rate Survival Frac. over time.svg"), dpi=600)
plt.close()