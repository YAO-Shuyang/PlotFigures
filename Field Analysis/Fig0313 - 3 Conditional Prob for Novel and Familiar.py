from mylib.statistic_test import *
from mylib.statistic_test import GetMultidayIndexmap, ReadCellReg
from mylib.multiday.core import MultiDayCore

code_id = "0313 - Conditional Probability for Field Maintain"
loc = join(figpath, code_id, 'Novelty')
mkdir(loc)

if os.path.exists(join(figdata, code_id+' [Novelty].pkl')):
    with open(join(figdata, code_id+' [Novelty].pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = [
                             'Duration', 'Init Session', 'Conditional Prob.', 'Conditional Recover Prob.',
                             'Paradigm', 'On-Next Num', 'Off-Next Num'], 
                             f_member=['Type'], 
                             f = f_CellReg_modi, function = ConditionalProb_Interface_NovelFalimiar, 
                             file_name = code_id+' [Novelty]', behavior_paradigm = 'CrossMaze'
           )


# Statistical Analysis using one-way ANOVA
# Define a function to perform one-way ANOVA
def one_way_anova(data: dict, init_session_max: int, duration: int, key: str) -> None:
    
    groups = []
    for i in range(1, init_session_max+1):
        idx = np.where(
            (data['Duration'] == duration) &
            (data['Init Session'] == i)
        )[0]
        if len(idx) > 0:
            groups += [data[key][idx]]
    f_value, p_value = scipy.stats.f_oneway(*groups)
    return f"  F-value: {f_value:.4f}, p-value: {p_value}, len: {len(groups)}"

colors = sns.color_palette("rocket", 3)[1:]
markercolors = [sns.color_palette("Blues", 3)[1], sns.color_palette("Blues", 3)[2]]
chancecolors = ['#D4C9A8', '#8E9F85', '#C3AED6', '#FED7D7']

#Data['hue'] = np.array([Data['Papadigm'][i] + ' ' + Data['Maze Type'][i] for i in range(Data['Duration'].shape[0])])
idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Conditional Prob.']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Type'] == 'Real'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)

idx1 = np.where(SubData['Maze Type'] == 'Maze 1')[0]
idx2 = np.where(SubData['Maze Type'] == 'Maze 2')[0]

fig = plt.figure(figsize=(8, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Prob.',
    data=SubDict(SubData, SubData.keys(), idx=idx1),
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=2,
    linewidth=0.1,
    alpha=0.8,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 103)
ax.set_yticks(np.linspace(0, 100, 6))

plt.savefig(join(loc, 'conditional prob [Maze 1].png'), dpi = 600)
plt.savefig(join(loc, 'conditional prob [Maze 1].svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(5, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Prob.',
    data=SubDict(SubData, SubData.keys(), idx=idx2),
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=2,
    linewidth=0.1,
    alpha=0.8,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 103)
ax.set_yticks(np.linspace(0, 100, 6))

plt.savefig(join(loc, 'conditional prob [Maze 2].png'), dpi = 600)
plt.savefig(join(loc, 'conditional prob [Maze 2].svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(10, 2.5))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'On-Next Num',
    data=SubDict(SubData, SubData.keys(), idx=idx1),
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=3,
    linewidth=0.1,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.semilogy()
ax.set_ylim(1, 50000)
plt.savefig(join(loc, 'on-next num [Maze 1].png'), dpi = 600)
plt.savefig(join(loc, 'on-next num [Maze 1].svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(5, 2.5))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'On-Next Num',
    data=SubDict(SubData, SubData.keys(), idx=idx2),
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=3,
    linewidth=0.1,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.semilogy()
ax.set_ylim(1, 50000)
plt.savefig(join(loc, 'on-next num [Maze 2].png'), dpi = 600)
plt.savefig(join(loc, 'on-next num [Maze 2].svg'), dpi = 600)
plt.close()

# Statistical Test
print("Maze A - Conditional Prob.:")
for dur in range(1, 12):
    print(f"  {dur} - "+one_way_anova(SubDict(SubData, SubData.keys(), idx=idx1), duration = dur, init_session_max=12, key = 'Conditional Prob.'))
print()
print("Maze B - Conditional Prob.:")
for dur in range(1, 12):
    print(f"  {dur} - "+one_way_anova(SubDict(SubData, SubData.keys(), idx=idx2), duration = dur, init_session_max=12, key = 'Conditional Prob.'))
print()

fig = plt.figure(figsize=(4,2))
idx = np.where((Data['Paradigm'] == 'CrossMaze')&
               (np.isnan(Data['Conditional Recover Prob.']) == False)&
               (Data['Maze Type'] != 'Open Field')&
               (Data['Type'] == 'Real'))[0]

SubData = SubDict(Data, Data.keys(), idx=idx)
idx1 = np.where(SubData['Maze Type'] == 'Maze 1')[0]
idx2 = np.where(SubData['Maze Type'] == 'Maze 2')[0]


fig = plt.figure(figsize=(10, 2.5))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Recover Prob.',
    data=SubDict(SubData, SubData.keys(), idx=idx1),
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(-1, 40)
ax.set_yticks(np.linspace(0, 40, 9))

plt.savefig(join(loc, 'Conditional recover prob [Maze 1].png'), dpi = 600)
plt.savefig(join(loc, 'Conditional recover prob [Maze 1].svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(10, 2.5))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Off-Next Num',
    data=SubDict(SubData, SubData.keys(), idx=idx1),
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=3,
    linewidth=0.1,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.semilogy()
ax.set_ylim(1, 40000)
plt.savefig(join(loc, 'off-next num [Maze 1].png'), dpi = 600)
plt.savefig(join(loc, 'off-next num [Maze 1].svg'), dpi = 600)
plt.close()


fig = plt.figure(figsize=(5, 2.5))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Recover Prob.',
    data=SubDict(SubData, SubData.keys(), idx=idx2),
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(-1, 40)
ax.set_yticks(np.linspace(0, 40, 9))

plt.savefig(join(loc, 'Conditional recover prob [Maze 2].png'), dpi = 600)
plt.savefig(join(loc, 'Conditional recover prob [Maze 2].svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(5, 2.5))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Off-Next Num',
    data=SubDict(SubData, SubData.keys(), idx=idx2),
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=3,
    linewidth=0.1,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.semilogy()
ax.set_ylim(1, 40000)
plt.savefig(join(loc, 'off-next num [Maze 2].png'), dpi = 600)
plt.savefig(join(loc, 'off-next num [Maze 2].svg'), dpi = 600)
plt.close()

# Statistical Test
print("Maze A - Conditional Recover Prob.:")
for dur in range(1, 12):
    print(f"  {dur} - "+one_way_anova(SubDict(SubData, SubData.keys(), idx=idx1), duration = dur, init_session_max=12, key = 'Conditional Recover Prob.'))
print()
print("Maze B - Conditional Recover Prob.:")
for dur in range(1, 11):
    print(f"  {dur} - "+one_way_anova(SubDict(SubData, SubData.keys(), idx=idx2), duration = dur, init_session_max=12, key = 'Conditional Recover Prob.'))
print()

# MAf ----------------------------------------------------------------------------
idx = np.where(
    (Data['Paradigm'] == 'ReverseMaze cis') &
    (np.isnan(Data['Conditional Prob.']) == False)&
    (Data['Type'] == 'Real')
)[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Prob.',
    data=SubData,
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=2,
    linewidth=0.1,
    alpha=0.8,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 103)
ax.set_yticks(np.linspace(0, 100, 6))

plt.savefig(join(loc, 'conditional prob [MAf].png'), dpi = 600)
plt.savefig(join(loc, 'conditional prob [MAf].svg'), dpi = 600)
plt.close()

print("MAf - Conditional Prob.:")
for dur in range(1, 6):
    print(f"  {dur} - "+one_way_anova(SubData, duration = dur, init_session_max=6, key = 'Conditional Prob.'))
print()

idx = np.where(
    (Data['Paradigm'] == 'ReverseMaze cis') &
    (np.isnan(Data['Conditional Recover Prob.']) == False)&
    (Data['Type'] == 'Real')
)[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Recover Prob.',
    data=SubData,
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(-1, 40)
ax.set_yticks(np.linspace(0, 40, 9))
plt.savefig(join(loc, 'Conditional recover prob [MAf].png'), dpi = 600)
plt.savefig(join(loc, 'Conditional recover prob [MAf].svg'), dpi = 600)
plt.close()
print("MAf - Conditional Recover Prob.:")
for dur in range(1, 5):
    print(f"  {dur} - "+one_way_anova(SubData, duration = dur, init_session_max=5, key = 'Conditional Recover Prob.'))
print()

# MAb ----------------------------------------------------------------------------
idx = np.where(
    (Data['Paradigm'] == 'ReverseMaze trs') &
    (np.isnan(Data['Conditional Prob.']) == False)&
    (Data['Type'] == 'Real')
)[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Prob.',
    data=SubData,
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=2,
    linewidth=0.1,
    alpha=0.8,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 103)
ax.set_yticks(np.linspace(0, 100, 6))

plt.savefig(join(loc, 'conditional prob [MAb].png'), dpi = 600)
plt.savefig(join(loc, 'conditional prob [MAb].svg'), dpi = 600)
plt.close()

print("MAb - Conditional Prob.:")
for dur in range(1, 6):
    print(f"  {dur} - "+one_way_anova(SubData, duration = dur, init_session_max=6, key = 'Conditional Prob.'))
print()

idx = np.where(
    (Data['Paradigm'] == 'ReverseMaze trs') &
    (np.isnan(Data['Conditional Recover Prob.']) == False)&
    (Data['Type'] == 'Real')
)[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(4, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Recover Prob.',
    data=SubData,
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(-1, 40)
ax.set_yticks(np.linspace(0, 40, 9))
plt.savefig(join(loc, 'Conditional recover prob [MAb].png'), dpi = 600)
plt.savefig(join(loc, 'Conditional recover prob [MAb].svg'), dpi = 600)
plt.close()
print("MAb - Conditional Recover Prob.:")
for dur in range(1, 5):
    print(f"  {dur} - "+one_way_anova(SubData, duration = dur, init_session_max=5, key = 'Conditional Recover Prob.'))
print()


# HPf ----------------------------------------------------------------------------
idx = np.where(
    (Data['Paradigm'] == 'HairpinMaze cis') &
    (np.isnan(Data['Conditional Prob.']) == False)&
    (Data['Type'] == 'Real')
)[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(2, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Prob.',
    data=SubData,
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=2,
    linewidth=0.1,
    alpha=0.8,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 103)
ax.set_yticks(np.linspace(0, 100, 6))

plt.savefig(join(loc, 'conditional prob [HPf].png'), dpi = 600)
plt.savefig(join(loc, 'conditional prob [HPf].svg'), dpi = 600)
plt.close()

print("HPf - Conditional Prob.:")
for dur in range(1, 6):
    print(f"  {dur} - "+one_way_anova(SubData, duration = dur, init_session_max=6, key = 'Conditional Prob.'))
print()

idx = np.where(
    (Data['Paradigm'] == 'HairpinMaze cis') &
    (np.isnan(Data['Conditional Recover Prob.']) == False)&
    (Data['Type'] == 'Real')
)[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(2, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Recover Prob.',
    data=SubData,
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(-1, 40)
ax.set_yticks(np.linspace(0, 40, 9))
plt.savefig(join(loc, 'Conditional recover prob [HPf].png'), dpi = 600)
plt.savefig(join(loc, 'Conditional recover prob [HPf].svg'), dpi = 600)
plt.close()
print("HPf - Conditional Recover Prob.:")
for dur in range(1, 5):
    print(f"  {dur} - "+one_way_anova(SubData, duration = dur, init_session_max=5, key = 'Conditional Recover Prob.'))
print()

# HPb ----------------------------------------------------------------------------
idx = np.where(
    (Data['Paradigm'] == 'HairpinMaze trs') &
    (np.isnan(Data['Conditional Prob.']) == False)&
    (Data['Type'] == 'Real')
)[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(2, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Prob.',
    data=SubData,
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=2,
    linewidth=0.1,
    alpha=0.8,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 103)
ax.set_yticks(np.linspace(0, 100, 6))

plt.savefig(join(loc, 'conditional prob [HPb].png'), dpi = 600)
plt.savefig(join(loc, 'conditional prob [HPb].svg'), dpi = 600)
plt.close()

print("HPb - Conditional Prob.:")
for dur in range(1, 6):
    print(f"  {dur} - "+one_way_anova(SubData, duration = dur, init_session_max=6, key = 'Conditional Prob.'))
print()

idx = np.where(
    (Data['Paradigm'] == 'HairpinMaze trs') &
    (np.isnan(Data['Conditional Recover Prob.']) == False)&
    (Data['Type'] == 'Real')
)[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
fig = plt.figure(figsize=(2, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Recover Prob.',
    data=SubData,
    hue = "Init Session",
    palette = 'Spectral',
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(-1, 40)
ax.set_yticks(np.linspace(0, 40, 9))
plt.savefig(join(loc, 'Conditional recover prob [HPb].png'), dpi = 600)
plt.savefig(join(loc, 'Conditional recover prob [HPb].svg'), dpi = 600)
plt.close()
print("HPb - Conditional Recover Prob.:")
for dur in range(1, 5):
    print(f"  {dur} - "+one_way_anova(SubData, duration = dur, init_session_max=5, key = 'Conditional Recover Prob.'))
print()