from mylib.statistic_test import *
from mylib.multiday.field_tracker import conditional_prob, conditional_prob_jumpnan

code_id = "0313 - Conditional Probability for Field Maintain"
loc = join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = {"MiceID": np.array([], np.int64), "Maze Type": np.array([]), 
            "Duration": np.array([], np.int64), "Conditional Prob.": np.array([], np.float64), "No Detect Prob.": np.array([], np.float64),
            "Recovered Prob.": np.array([], np.float64), "Cumulative Prob.": np.array([], np.float64),
            "Re-detect Active Prob.": np.array([], np.float64), "Re-detect Prob.": np.array([], np.float64)}
    
    for i in range(len(f_CellReg_day)):
        if f_CellReg_day['include'][i] == 0 or f_CellReg_day['maze_type'][i] == 0:
            continue
        
        with open(f_CellReg_day['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
            
        retained_dur, prob, nodetect_prob, recover_prob, redetect_prob, redetect_frac, on_next_prob = conditional_prob(trace)
        
        
        Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(int(f_CellReg_day['MiceID'][i]), prob.shape[0])])
        Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat("Maze "+str(int(f_CellReg_day['maze_type'][i])), prob.shape[0])])
        Data['Duration'] = np.concatenate([Data['Duration'], retained_dur])
        Data['Conditional Prob.'] = np.concatenate([Data['Conditional Prob.'], prob*100])
        Data['No Detect Prob.'] = np.concatenate([Data['No Detect Prob.'], nodetect_prob*100])
        Data['Recovered Prob.'] = np.concatenate([Data['Recovered Prob.'], recover_prob*100])
        Data['Re-detect Active Prob.'] = np.concatenate([Data['Re-detect Active Prob.'], redetect_prob*100])
        Data['Re-detect Prob.'] = np.concatenate([Data['Re-detect Prob.'], redetect_frac*100])
        res = np.nancumprod(prob)*100
        res[np.isnan(prob)] = np.nan
        res[0] = 100
        Data['Cumulative Prob.'] = np.concatenate([Data['Cumulative Prob.'], res])
        
        del trace
        gc.collect()
        
    with open(join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
        
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)

"""

retain_m1 = np.array([13.0892299996294, 17.61975629, 18.29082998, 21.31898831, 17.12948351])
retain_m2 = np.array([19.59862026, 30.588844])

print_estimator(retain_m1)
print_estimator(retain_m2)

disappear_m1 = np.array([66.18479, 68.99102, 66.02225, 55.75556, 71.85714, ])
disappear_m2 = np.array([67.46818, 70.17974, 72.8004, 68.35095])

print_estimator(disappear_m1)
print_estimator(disappear_m2)

idx = np.where((Data['Duration'] == 1)&(Data['Maze Type'] == "Maze 1"))[0]
print_estimator(Data['Conditional Prob.'][idx])
idx = np.where((Data['Duration'] == 1)&(Data['Maze Type'] == "Maze 2"))[0]
print_estimator(Data['Conditional Prob.'][idx])

fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.where((Data['Maze Type'] == "Maze 1")&(np.isnan(Data['Conditional Prob.']) == False))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
a1, b1 = SigmoidFit(SubData['Duration'], SubData['Conditional Prob.']/100)
x1 = np.linspace(1, 23, 26001)
y1 = Sigmoid(x1, a1, b1)
print(f"Maze 1: {a1:.3f}, {b1:.3f}")
ax.plot(x1, y1*100, color=sns.color_palette("rocket", 3)[1], linewidth=0.5)
idx = np.where((Data['Maze Type'] == "Maze 2")&(np.isnan(Data['Conditional Prob.']) == False))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
a2, b2 = SigmoidFit(SubData['Duration'], SubData['Conditional Prob.']/100)
x2 = np.linspace(1, 12, 26001)
y2 = Sigmoid(x2, a2, b2)
print(f"Maze 2: {a2:.3f}, {b2:.3f}")
ax.plot(x2, y2*100, color=sns.color_palette("rocket", 3)[2], linewidth=0.5)
sns.stripplot(
    x = 'Duration',
    y = 'Conditional Prob.',
    data=Data,
    hue = "Maze Type",
    palette = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[3]],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 103)
ax.set_yticks(np.linspace(0, 100, 6))

plt.savefig(join(loc, 'conditional prob.png'), dpi = 600)
plt.savefig(join(loc, 'conditional prob.svg'), dpi = 600)
plt.close()


fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
print(np.cumprod(y1), y2)
x1 = np.linspace(1, 23, 23)
y1 = Sigmoid(x1, a1, b1)
x2 = np.linspace(1, 12, 12)
y2 = Sigmoid(x2, a2, b2)
y3 = np.concatenate([[100], np.cumprod(y1)*100])
y4 = np.concatenate([[100], np.cumprod(y2)*100])
ax.plot(np.linspace(0, 23, 24), y3, color=sns.color_palette("rocket", 3)[1], linewidth=0.5)
ax.plot(np.linspace(0, 12, 13), y4, color=sns.color_palette("rocket", 3)[2], linewidth=0.5)
x = np.linspace(1, 1000, 1000)
y1 = np.cumprod(Sigmoid(x, a1, b1))
y2 = np.cumprod(Sigmoid(x, a2, b2))
print(y1[-1], y2[-1])
sns.stripplot(
    x = 'Duration',
    y = 'Cumulative Prob.',
    data=Data,
    hue = "Maze Type",
    palette = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[3]],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 103)
ax.set_yticks(np.linspace(0, 100, 6))

plt.savefig(join(loc, 'cumulative conditional prob.png'), dpi = 600)
plt.savefig(join(loc, 'cumulative conditional prob.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(4,2))
Data['Recovered Prob.'][Data['Recovered Prob.'] == 0] = np.nan
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Duration',
    y = 'Recovered Prob.',
    data=Data,
    hue = "Maze Type",
    palette=sns.color_palette("rocket", 3)[1:],
    ax=ax,
    markeredgecolor=None,
    errorbar='se',
    legend=False,
    err_kws={'edgecolor':None},
    linewidth=0.5,
)
sns.stripplot(
    x = 'Duration',
    y = 'Recovered Prob.',
    data=Data,
    hue = "Maze Type",
    palette = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[3]],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(-3, 30)
ax.set_yticks(np.linspace(0, 30, 7))

plt.savefig(join(loc, 'recover prob.png'), dpi = 600)
plt.savefig(join(loc, 'recover prob.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(4,2))
Data['Recovered Prob.'][Data['Recovered Prob.'] == 0] = np.nan
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Duration',
    y = 'Recovered Prob.',
    data=Data,
    hue = "Maze Type",
    palette=sns.color_palette("rocket", 3)[1:],
    ax=ax,

    markeredgecolor=None,
    errorbar='se',
    legend=False,
    err_kws={'edgecolor':None},
    linewidth=0.5,
)
sns.stripplot(
    x = 'Duration',
    y = 'Recovered Prob.',
    data=Data,
    hue = "Maze Type",
    palette = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[3]],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0.001, 30)
ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 20, 30])
ax.semilogy()

plt.savefig(join(loc, 'recover prob semilog.png'), dpi = 600)
plt.savefig(join(loc, 'recover prob semilog.svg'), dpi = 600)
plt.close()

fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'No Detect Prob.',
    data=Data,
    hue = "Maze Type",
    palette = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[3]],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(0, 100)
ax.set_yticks(np.linspace(0, 100, 6))
plt.close()

fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.stripplot(
    x = 'Duration',
    y = 'Re-detect Active Prob.',
    data=Data,
    hue = "Maze Type",
    palette = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[3]],
    edgecolor='black',
    size=3,
    linewidth=0.15,
    ax = ax,
    dodge=True,
    jitter=0.1
)
ax.set_ylim(-3, 70)
ax.set_yticks(np.linspace(0, 70, 8))
ax.set_xlabel("Not detected duration / session")
plt.show()
"""