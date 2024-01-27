from mylib.statistic_test import *
from mylib.multiday.field_tracker import conditional_prob, conditional_prob_jumpnan

code_id = "0313 - Conditional Probability for Field Maintain"
loc = join(figpath, code_id)
mkdir(loc)

class Counter:
    def __init__(self, stat: int) -> None:
        if np.isnan(stat):
            stat = 0
        else:
            stat = int(stat)
            
        self._stat = stat
        self._hist = 1
        
    @property
    def stat(self):
        return self._stat
    
    @property
    def hist(self):
        return self._hist
    
    def update(self, stat: int):
        if np.isnan(stat):
            stat = 0
        else:
            stat = int(stat)
            
        if self._stat == 1:
            if stat == 1:
                self._hist += 1
            else:
                self._hist = 1
                
        elif self._stat == 0:
            if stat == 1:
                self._hist = 1
            else:
                self._hist += 1
        else:
            raise ValueError(f"The stat should be 0 or 1! instead of {stat}")
        
        self._stat = stat
        
        
def count_superstable_fields(fields: list[Counter]):
    n_superstable = 0
    for field in fields:
        if field.stat == 1 and field.hist >= 15:
            n_superstable += 1
    return n_superstable   
 
def update_counters(fields: list[Counter], index_line):
    for i, field in enumerate(fields):
        field.update(index_line[i])
    return fields

def calculate_superstable_fraction(field_reg: np.ndarray):
    fields = [Counter(i) for i in field_reg[0, :]]
    superstable_frac = np.zeros(field_reg.shape[0])
    
    n_fields = np.mean(np.nansum(field_reg, axis=1))
    
    for i in tqdm(range(1, field_reg.shape[0])):
        fields = update_counters(fields, field_reg[i, :])
        superstable_frac[i] = count_superstable_fields(fields) / n_fields
    return superstable_frac

def calculate_survival_fraction(field_reg: np.ndarray):
    survival_frac = np.full((field_reg.shape[0], field_reg.shape[0]), np.nan)
    start_sessions = np.full((field_reg.shape[0], field_reg.shape[0]), np.nan)
    training_day = np.full((field_reg.shape[0], field_reg.shape[0]), np.nan)
    for i in tqdm(range(field_reg.shape[0])):
        # start session
        n_ori = np.where(field_reg[i, :] == 1)[0].shape[0]
        for j in range(i, field_reg.shape[0]):
            # compare session
            n_retain = np.where(np.nansum(field_reg[i:j+1, :], axis=0) == j-i+1)[0].shape[0]
            survival_frac[i, j] = n_retain / n_ori
            start_sessions[i, j] = i+1
            training_day[i, j] = j-i+1
    return survival_frac, start_sessions, training_day
    
Data = {"MiceID": np.array([], np.int64), "Maze Type": np.array([]), "Data Type": np.array([]),
            "Duration": np.array([], np.int64), "Conditional Prob.": np.array([], np.float64), "No Detect Prob.": np.array([], np.float64),
            "Recovered Prob.": np.array([], np.float64), "Cumulative Prob.": np.array([], np.float64),
            "Re-detect Active Prob.": np.array([], np.float64), "Re-detect Prob.": np.array([], np.float64)}


#with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\trace_mdays.pkl", 'rb') as handle:
#with open(r"E:\Data\Cross_maze\10224\Super Long-term Maze 1\trace_mdays.pkl", 'rb') as handle:
with open(r"E:\Data\Cross_maze\10212\Maze-2-footprint\trace_mdays.pkl", 'rb') as handle:
    trace = pickle.load(handle)

field_num_mat = np.where(np.isnan(trace['field_reg']), 0, 1)[:, :]
num = np.count_nonzero(field_num_mat, axis=0)
#idx = np.where(num >= 26)[0]
#trace['field_reg'] = trace['field_reg'][:, idx] # [equal rate]
"""    
with open(r"E:\Data\FigData\PermenantFieldAnalysis\Mouse 1 [equal rate].pkl", 'rb') as handle: # [equal rate]
    trace = pickle.load(handle)
    trace['field_reg'] = trace['field_reg'].T[:24, :]
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



retained_dur, prob, nodetect_prob, recover_prob, redetect_prob, redetect_frac, on_next_prob1 = conditional_prob(trace)
on_next_prob1[1:, 3] = on_next_prob1[:-1, 3]
on_next_prob1[0, 3] = 0
on_next_prob1[0, 1] = np.sum(on_next_prob1[1, :])

Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(int(trace['MiceID']), prob.shape[0])])
Data['Data Type'] = np.concatenate([Data['Data Type'], np.repeat("Post", prob.shape[0])])
Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat("Maze "+str(int(trace['maze_type'])), prob.shape[0])])
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
Data['MiceID'] = np.concatenate([Data['MiceID'], np.repeat(int(trace['MiceID']), prob.shape[0])])
Data['Data Type'] = np.concatenate([Data['Data Type'], np.repeat("Pre", prob.shape[0])])
Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat("Maze "+str(int(trace['maze_type'])), prob.shape[0])])
Data['Duration'] = np.concatenate([Data['Duration'], retained_dur])
Data['Conditional Prob.'] = np.concatenate([Data['Conditional Prob.'], prob*100])
Data['No Detect Prob.'] = np.concatenate([Data['No Detect Prob.'], nodetect_prob*100])
Data['Recovered Prob.'] = np.concatenate([Data['Recovered Prob.'], recover_prob*100])
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
