from mylib.statistic_test import *

code_id = '0063 - Identify RS phase'
loc = os.path.join(figpath, code_id)
mkdir(loc)
    
def generate_improvement_curve(
    LapwiseTime: dict,
    correct_rate: dict,
    lapwise_distance: dict,
    lapwise_velocity: dict
) -> dict:
    Data = {
        "MiceID": np.array([], np.int64),
        "date": np.array([], np.int64),
        "Training Day": np.array([]),
        "Stage": np.array([]),
        "Maze Type": np.array([]),
        "Median Time": np.array([], np.float64),
        "Mean Correct Rate": np.array([], np.float64),
        "Median Distance": np.array([], np.float64),
        "Median Speed": np.array([], np.float64),
    }
    
    days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7", "Day 8", "Day 9", ">=Day 10"]
    mice = [10209, 10212, 10224, 10227]
    # Stage 1, Maze 1
    for m in mice:
        for d in days:
            idx = np.where((LapwiseTime['Training Day'] == d)&(LapwiseTime['Stage'] == 'Stage 1')&(LapwiseTime['Maze Type'] == 'Maze 1')&(LapwiseTime['MiceID'] == m))[0]
            if len(idx) != 0:
            
                Data['date'] = np.append(Data['date'], LapwiseTime['date'][idx[0]])
                Data['MiceID'] = np.append(Data['MiceID'], LapwiseTime['MiceID'][idx[0]])
                Data['Training Day'] = np.append(Data['Training Day'], LapwiseTime['Training Day'][idx[0]])
                Data['Stage'] = np.append(Data['Stage'], LapwiseTime['Stage'][idx[0]])
                Data['Maze Type'] = np.append(Data['Maze Type'], LapwiseTime['Maze Type'][idx[0]])
        
                Data['Median Time'] = np.append(Data['Median Time'], np.nanmedian(LapwiseTime['Lap-wise time cost'][idx]))
            
                idx = np.where((correct_rate['Training Day'] == d)&
                           (correct_rate['Stage'] == 'Stage 1')&
                           (correct_rate['Maze Type'] == 'Maze 1')&
                           (correct_rate['MiceID'] == m))[0]
                Data['Mean Correct Rate'] = np.append(Data['Mean Correct Rate'], np.nanmean(correct_rate['Correct Rate'][idx]))
        
                idx = np.where((lapwise_distance['Training Day'] == d)&
                           (lapwise_distance['Stage'] == 'Stage 1')&
                           (lapwise_distance['Maze Type'] == 'Maze 1')&
                           (lapwise_distance['MiceID'] == m))[0]
                Data['Median Distance'] = np.append(Data['Median Distance'], np.nanmedian(lapwise_distance['Lap-wise Distance'][idx]))
            
                idx = np.where((lapwise_velocity['Training Day'] == d)&
                           (lapwise_velocity['Stage'] == 'Stage 1')&
                           (lapwise_velocity['Maze Type'] == 'Maze 1')&
                           (lapwise_velocity['MiceID'] == m))[0]
                Data['Median Speed'] = np.append(Data['Median Speed'], np.nanmean(lapwise_velocity['Lap-wise Average Velocity'][idx]))
        
        
            idx = np.where((LapwiseTime['Training Day'] == d)&(LapwiseTime['Stage'] == 'Stage 2')&(LapwiseTime['Maze Type'] == 'Maze 1')&(LapwiseTime['MiceID'] == m))[0]
            if len(idx) != 0:
            
                Data['date'] = np.append(Data['date'], LapwiseTime['date'][idx[0]])
                Data['MiceID'] = np.append(Data['MiceID'], LapwiseTime['MiceID'][idx[0]])
                Data['Training Day'] = np.append(Data['Training Day'], LapwiseTime['Training Day'][idx[0]])
                Data['Stage'] = np.append(Data['Stage'], LapwiseTime['Stage'][idx[0]])
                Data['Maze Type'] = np.append(Data['Maze Type'], LapwiseTime['Maze Type'][idx[0]])
        
                Data['Median Time'] = np.append(Data['Median Time'], np.nanmedian(LapwiseTime['Lap-wise time cost'][idx]))
            
                idx = np.where((correct_rate['Training Day'] == d)&
                           (correct_rate['Stage'] == 'Stage 2')&
                           (correct_rate['Maze Type'] == 'Maze 1')&
                           (correct_rate['MiceID'] == m))[0]
                Data['Mean Correct Rate'] = np.append(Data['Mean Correct Rate'], np.nanmean(correct_rate['Correct Rate'][idx]))
        
                idx = np.where((lapwise_distance['Training Day'] == d)&
                           (lapwise_distance['Stage'] == 'Stage 2')&
                           (lapwise_distance['Maze Type'] == 'Maze 1')&
                           (lapwise_distance['MiceID'] == m))[0]
                Data['Median Distance'] = np.append(Data['Median Distance'], np.nanmedian(lapwise_distance['Lap-wise Distance'][idx]))
            
                idx = np.where((lapwise_velocity['Training Day'] == d)&
                           (lapwise_velocity['Stage'] == 'Stage 2')&
                           (lapwise_velocity['Maze Type'] == 'Maze 1')&
                           (lapwise_velocity['MiceID'] == m))[0]
                Data['Median Speed'] = np.append(Data['Median Speed'], np.nanmedian(lapwise_velocity['Lap-wise Average Velocity'][idx]))
        
        idx = np.where((Data['Maze Type'] == 'Maze 1')&(Data['MiceID'] == m))[0]
        t_min = np.min(Data['Median Time'][idx])
        t_max = np.max(Data['Median Time'][idx])
        Data['Median Time'][idx] = 1 - (Data['Median Time'][idx] - t_min) / (t_max - t_min)
    
        cr_max = np.max(Data['Mean Correct Rate'][idx])
        cr_min = np.min(Data['Mean Correct Rate'][idx])
        Data['Mean Correct Rate'][idx] = (Data['Mean Correct Rate'][idx] - cr_min) / (cr_max - cr_min)
    
        d_min = np.min(Data['Median Distance'][idx])
        d_max = np.max(Data['Median Distance'][idx])
        Data['Median Distance'][idx] = 1 - (Data['Median Distance'][idx] - d_min) / (d_max - d_min)
    
        v_min = np.min(Data['Median Speed'][idx])
        v_max = np.max(Data['Median Speed'][idx])
        Data['Median Speed'][idx] = (Data['Median Speed'][idx] - v_min) / (v_max - v_min)

    # Stage 2, Maze 2    
    for m in mice:
        for d in days:
            idx = np.where((LapwiseTime['Training Day'] == d)&(LapwiseTime['Stage'] == 'Stage 2')&(LapwiseTime['Maze Type'] == 'Maze 2')&(LapwiseTime['MiceID'] == m))[0]
            if len(idx) == 0:
                continue
            
            Data['date'] = np.append(Data['date'], LapwiseTime['date'][idx[0]])
            Data['MiceID'] = np.append(Data['MiceID'], LapwiseTime['MiceID'][idx[0]])
            Data['Training Day'] = np.append(Data['Training Day'], LapwiseTime['Training Day'][idx[0]])
            Data['Stage'] = np.append(Data['Stage'], LapwiseTime['Stage'][idx[0]])
            Data['Maze Type'] = np.append(Data['Maze Type'], LapwiseTime['Maze Type'][idx[0]])
        
            Data['Median Time'] = np.append(Data['Median Time'], np.nanmedian(LapwiseTime['Lap-wise time cost'][idx]))
            
            idx = np.where((correct_rate['Training Day'] == d)&
                           (correct_rate['Stage'] == 'Stage 1')&
                           (correct_rate['Maze Type'] == 'Maze 1')&
                           (correct_rate['MiceID'] == m))[0]
            Data['Mean Correct Rate'] = np.append(Data['Mean Correct Rate'], np.nanmean(correct_rate['Correct Rate'][idx]))
        
            idx = np.where((lapwise_distance['Training Day'] == d)&
                           (lapwise_distance['Stage'] == 'Stage 1')&
                           (lapwise_distance['Maze Type'] == 'Maze 1')&
                           (lapwise_distance['MiceID'] == m))[0]
            Data['Median Distance'] = np.append(Data['Median Distance'], np.nanmedian(lapwise_distance['Lap-wise Distance'][idx]))
            
            idx = np.where((lapwise_velocity['Training Day'] == d)&
                           (lapwise_velocity['Stage'] == 'Stage 1')&
                           (lapwise_velocity['Maze Type'] == 'Maze 1')&
                           (lapwise_velocity['MiceID'] == m))[0]
            Data['Median Speed'] = np.append(Data['Median Speed'], np.nanmedian(lapwise_velocity['Lap-wise Average Velocity'][idx]))
        
        idx = np.where((Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 2')&(Data['MiceID'] == m))[0]
        t_min = np.min(Data['Median Time'][idx])
        t_max = np.max(Data['Median Time'][idx])
        Data['Median Time'][idx] = 1 - (Data['Median Time'][idx] - t_min) / (t_max - t_min)
    
        cr_max = np.max(Data['Mean Correct Rate'][idx])
        cr_min = np.min(Data['Mean Correct Rate'][idx])
        Data['Mean Correct Rate'][idx] = (Data['Mean Correct Rate'][idx] - cr_min) / (cr_max - cr_min)
    
        d_min = np.min(Data['Median Distance'][idx])
        d_max = np.max(Data['Median Distance'][idx])
        Data['Median Distance'][idx] = 1 - (Data['Median Distance'][idx] - d_min) / (d_max - d_min)
    
        v_min = np.min(Data['Median Speed'][idx])
        v_max = np.max(Data['Median Speed'][idx])
        Data['Median Speed'][idx] = (Data['Median Speed'][idx] - v_min) / (v_max - v_min)

    return Data


if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    maze_indices = np.where(f_pure_behav['maze_type'] != 0)[0]
    if os.path.exists(join(figdata, '0020 - learning curve'+'.pkl')):
        with open(join(figdata, '0020 - learning curve'+'.pkl'), 'rb') as handle:
            LapwiseTime = pickle.load(handle)
    else:
        LapwiseTime = DataFrameEstablish(variable_names = ['Lap ID', 'Lap-wise time cost'], is_behav=True,
                              file_idx=maze_indices,
                              f = f_pure_behav, function = LearningCurve_Interface, 
                              file_name = '0020 - learning curve', behavior_paradigm = 'CrossMaze')

    if os.path.exists(join(figdata,'0020 - learning curve'+'-2.pkl')):
        with open(join(figdata,'0020 - learning curve'+'-2.pkl'), 'rb') as handle:
            CorrectRate = pickle.load(handle)
    else:
        CorrectRate = DataFrameEstablish(variable_names = ['Correct Rate', 'Pass Number', 'Error Number', 'Pure Guess Correct Rate'], is_behav=True,
                              file_idx=maze_indices,
                              f = f_pure_behav, function = LearningCurveBehavioralScore_Interface, 
                              file_name ='0020 - learning curve'+'-2', behavior_paradigm = 'CrossMaze')
    
    if os.path.exists(os.path.join(figdata,'0050 lap-wise traveling distance'+'.pkl')) == False:
        LapwiseDistance = DataFrameEstablish(variable_names = ['Lap ID', 'Lap-wise Distance'], f = f_pure_behav, 
                              function = LapwiseTravelDistance_Interface, is_behav = True,
                              file_name ='0050 lap-wise traveling distance', behavior_paradigm = 'CrossMaze', func_kwgs = {'is_placecell':False})
    else:
        with open(os.path.join(figdata,'0050 lap-wise traveling distance'+'.pkl'), 'rb') as handle:
            LapwiseDistance = pickle.load(handle)
        
        
    if os.path.exists(join(figdata, '0055 - lap-wise average velocity'+'.pkl')):
        with open(join(figdata, '0055 - lap-wise average velocity'+'.pkl'), 'rb') as handle:
            LapwiseVelocity = pickle.load(handle)
    else:
        LapwiseVelocity = DataFrameEstablish(variable_names = ['Lap-wise Average Velocity', 'Lap ID'], is_behav=True,
                              f = f_pure_behav, function = LapwiseAverageVelocity_Interface, 
                              file_name = '0055 - lap-wise average velocity', behavior_paradigm = 'CrossMaze')
    
    Data = generate_improvement_curve(LapwiseTime, CorrectRate, LapwiseDistance, LapwiseVelocity)
    with open(join(figdata, code_id+'.pkl'), 'wb') as handle:
        pickle.dump(Data, handle)
        
    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)

idx = np.where((Data['MiceID'] != 11095)&(Data['MiceID'] != 11092)&(Data['MiceID'] != 11096))[0]
Data = SubDict(Data, Data.keys(), idx)

uniq_day = ['Stage 1Day '+str(i) for i in range(1, 10)] + ['Stage 1>=Day 10'] + ['Stage 2Day '+str(i) for i in range(1, 10)] + ['Stage 2>=Day 10']
uniq_day2 = ['Day '+str(i) for i in range(1, 10)] + ['>=Day 10']

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(9,2), gridspec_kw={'width_ratios': [2, 1]})
ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

idx = np.where(Data['Maze Type'] == 'Maze 1')[0]
colors = sns.color_palette("rocket", 4)
x = np.array([Data['Stage'][i] + Data['Training Day'][i] for i in idx])
SubData = SubDict(Data, Data.keys(), idx=idx)

SubData['x'] = x
orderidx = np.concatenate([np.where(SubData['x'] == x)[0] for x in uniq_day])
SubData = SubDict(SubData, SubData.keys(), orderidx)
sns.lineplot(
    ax=ax1,
    x='x',
    y='Median Time',
    hue='Maze Type',
    data=SubData,
    palette=[colors[0]],
    err_kws={'edgecolor':None},
    linewidth=0.5
)
sns.lineplot(
    ax=ax1, 
    x='x',
    y='Mean Correct Rate',
    data=SubData,
    hue='Maze Type',
    palette=[colors[1]],
    err_kws={'edgecolor':None},
    linewidth=0.5
)
sns.lineplot(
    ax=ax1,
    x='x',
    y='Median Distance',
    data=SubData,
    hue='Maze Type',
    palette=[colors[2]],
    err_kws={'edgecolor':None},
    linewidth=0.5
)

ax1.set_ylim(0, 1)
ax1.set_yticks(np.linspace(0, 1, 6))
ax1.axhline(y=0.7, color='k', linestyle='--', linewidth=0.5)

idx = np.where(Data['Maze Type'] == 'Maze 2')[0]
colors = sns.color_palette("rocket", 4)
SubData = SubDict(Data, Data.keys(), idx=idx)
orderidx = np.concatenate([np.where(SubData['Training Day'] == x)[0] for x in uniq_day2])
SubData = SubDict(SubData, SubData.keys(), orderidx)
sns.lineplot(
    ax=ax2,
    x='Training Day',
    y='Median Time',
    hue='Maze Type',
    data=SubData,
    palette=[colors[0]],
    err_kws={'edgecolor':None},
    linewidth=0.5
    
)
sns.lineplot(
    ax=ax2, 
    x='Training Day',
    y='Mean Correct Rate',
    hue='Maze Type',
    data=SubData,
    palette=[colors[1]],
    err_kws={'edgecolor':None},
    linewidth=0.5
)
sns.lineplot(
    ax=ax2,
    x='Training Day',
    y='Median Distance',
    hue='Maze Type',
    data=SubData,
    palette=[colors[2]],
    err_kws={'edgecolor':None},
    linewidth=0.5

)
ax2.set_ylim(0, 1)
ax2.set_yticks(np.linspace(0, 1, 6))
ax2.axhline(y=0.7, color='k', linestyle='--', linewidth=0.5)

plt.savefig(join(loc, "RS phase.png"), dpi=600)
plt.savefig(join(loc, "RS phase.svg"), dpi=600)
plt.close()