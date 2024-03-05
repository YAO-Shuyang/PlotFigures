from mylib.statistic_test import *
from mylib.field.field_tracker import conditional_prob, conditional_prob_jumpnan

code_id = "0313 - Conditional Probability for Field Maintain"
loc = join(figpath, code_id)
mkdir(loc)
    
Data = {"MiceID": np.array([], np.int64), "Maze Type": np.array([]), "Data Type": np.array([]),
            "Duration": np.array([], np.int64), "Conditional Prob.": np.array([], np.float64), "No Detect Prob.": np.array([], np.float64),
            "Conditional Recovered Prob.": np.array([], np.float64), "Cumulative Prob.": np.array([], np.float64),
            "Re-detect Active Prob.": np.array([], np.float64), "Re-detect Prob.": np.array([], np.float64)}


#with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\trace_mdays.pkl", 'rb') as handle:
#with open(r"E:\Data\Cross_maze\10224\Super Long-term Maze 1\trace_mdays.pkl", 'rb') as handle:
#with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10227\Maze2-footprint\trace_mdays_conc.pkl", 'rb') as handle:
#with open(r"E:\Data\FigData\PermenantFieldAnalysis\mouse_1_converged_10000fields_50days.pkl", 'rb') as handle:
with open(r"E:\Data\FigData\PermenantFieldAnalysis\mouse_91_equal_rate_0.9_10000fields_50days.pkl", 'rb') as handle:
    trace = pickle.load(handle)
    trace['field_reg'] = trace['field_reg'].T[:30, :]
    #trace['field_reg'] = trace['field_reg'][:, :]

field_num_mat = np.where(np.isnan(trace['field_reg']), 0, 1)[:, :]
num = np.count_nonzero(field_num_mat, axis=0)
idx = np.where(num >= 8)[0]
trace['field_reg'] = trace['field_reg'][:, idx] # [equal rate]
"""    
with open(r"E:\Data\FigData\PermenantFieldAnalysis\Mouse 1 [equal rate].pkl", 'rb') as handle: # [equal rate]
    trace = pickle.load(handle)
    
"""
print(trace['field_reg'].shape)

supstable_frac = calculate_superstable_fraction(trace['field_reg'])
training_day = np.arange(supstable_frac.shape[0])
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(training_day, supstable_frac, linewidth=0.8)
ax.set_xlabel("Training day")
ax.set_ylabel("Superstable Fields %")
# ax.set_ylim(0, 0.15)
plt.show()

survival_frac, start_sessions, training_day = calculate_survival_fraction(trace['field_reg'])
SData = {"survival_frac": survival_frac.flatten(), "start_sessions": start_sessions.flatten(), "training_day": training_day.flatten()}
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = "training_day",
    y = "survival_frac",
    data = SData,
    hue="start_sessions",
    palette="rainbow",
    linewidth = 0.8,
    markers='o',
    ax = ax,
    legend=False
)
ax.set_ylim(0, 0.6)
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_xlabel("Training day")
ax.set_ylabel("Survival Fields %")
plt.show()



retained_dur, prob, nodetect_prob, conditional_recover_prob, global_recover_prob, redetect_prob, redetect_frac, on_next_prob1, off_next_prob1 = conditional_prob(trace)
on_next_prob1[1:, 3] = on_next_prob1[:-1, 3]
on_next_prob1[0, 3] = 0
on_next_prob1[0, 1] = np.sum(on_next_prob1[1, :])

#Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(int(trace['MiceID']), prob.shape[0])])
#Data['Data Type'] = np.concatenate([Data['Data Type'], np.repeat("Post", prob.shape[0])])
#Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat("Maze "+str(int(trace['maze_type'])), prob.shape[0])])
Data['Duration'] = np.concatenate([Data['Duration'], retained_dur])
Data['Conditional Prob.'] = np.concatenate([Data['Conditional Prob.'], prob*100])
Data['No Detect Prob.'] = np.concatenate([Data['No Detect Prob.'], nodetect_prob*100])
Data['Conditional Recovered Prob.'] = np.concatenate([Data['Conditional Recovered Prob.'], conditional_recover_prob*100])
Data['Re-detect Active Prob.'] = np.concatenate([Data['Re-detect Active Prob.'], redetect_prob*100])
Data['Re-detect Prob.'] = np.concatenate([Data['Re-detect Prob.'], redetect_frac*100])
res = np.nancumprod(prob)*100
res[np.isnan(prob)] = np.nan
res[0] = 100
Data['Cumulative Prob.'] = np.concatenate([Data['Cumulative Prob.'], res])

fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
plt.plot(Data['Duration'], Data['Conditional Prob.'], 'o-')
plt.xlabel('Duration (s)', fontsize=12)
plt.ylabel('Cumulative Probability (%)', fontsize=12)
plt.show()


# with open(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\trace_mdays.pkl", 'rb') as handle:
#     trace = pickle.load(handle)
    
retained_dur, prob, nodetect_prob, recover_prob, redetect_prob, redetect_frac, on_next_prob2 = conditional_prob(trace)
on_next_prob2[1:, 3] = on_next_prob2[:-1, 3]
on_next_prob2[0, 3] = 0
on_next_prob2[0, 1] = np.sum(on_next_prob2[1, :])
print(on_next_prob2[:])
#Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(int(trace['MiceID']), prob.shape[0])])
##Data['Data Type'] = np.concatenate([Data['Data Type'], np.repeat("Pre", prob.shape[0])])
#Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat("Maze "+str(int(trace['maze_type'])), prob.shape[0])])
Data['Duration'] = np.concatenate([Data['Duration'], retained_dur])
Data['Conditional Prob.'] = np.concatenate([Data['Conditional Prob.'], prob*100])
Data['No Detect Prob.'] = np.concatenate([Data['No Detect Prob.'], nodetect_prob*100])
Data['Conditional Recovered Prob.'] = np.concatenate([Data['Conditional Recovered Prob.'], recover_prob*100])
Data['Re-detect Active Prob.'] = np.concatenate([Data['Re-detect Active Prob.'], redetect_prob*100])
Data['Re-detect Prob.'] = np.concatenate([Data['Re-detect Prob.'],redetect_frac*100])
res = np.nancumprod(prob)*100
res[np.isnan(prob)] = np.nan
res[0] = 100
Data['Cumulative Prob.'] = np.concatenate([Data['Cumulative Prob.'], res])

fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
plt.plot(Data['Duration'], Data['Cumulative Prob.'], 'o-')
plt.xlabel('Duration (s)', fontsize=12)
plt.ylabel('Cumulative Probability (%)', fontsize=12)
plt.show()
