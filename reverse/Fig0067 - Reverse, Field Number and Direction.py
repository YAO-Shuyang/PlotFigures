from mylib.statistic_test import *

code_id = '0067 - Reverse, Field Number and Direction'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Field Number', 'Direction'], 
                              f = f3, function = PlaceFieldNumberPerDirection_Reverse_Interface, 
                              file_name = code_id, behavior_paradigm = 'ReverseMaze')
    
if os.path.exists(join(figdata, code_id+' [Hairpin FieldNumber].pkl')):
    with open(join(figdata, code_id+' [Hairpin FieldNumber].pkl'), 'rb') as handle:
        HPData = pickle.load(handle)
else:
    HPData = DataFrameEstablish(variable_names = ['Field Number', 'Direction'], 
                              f = f4, function = PlaceFieldNumberPerDirection_Reverse_Interface, 
                              file_name = code_id + ' [Hairpin FieldNumber]', behavior_paradigm = 'HairpinMaze')
    
if os.path.exists(join(figdata, code_id+' Correlation.pkl')):
    with open(join(figdata, code_id+' Correlation.pkl'), 'rb') as handle:
        CorrData = pickle.load(handle)
else:
    CorrData = DataFrameEstablish(variable_names = ['Corr', 'Shuffle'], 
                              f = f3, function = PlaceFieldNumberPerDirectionCorr_Reverse_Interface, 
                              file_name = code_id + ' Correlation', behavior_paradigm = 'ReverseMaze')

colors = [sns.color_palette('Blues', 9)[5], sns.color_palette('YlOrRd', 9)[5]]
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x="Training Day",
    y="Field Number",
    hue="Direction",
    data=Data,
    style='MiceID',
    err_style='bars',
    palette=colors,
    ax=ax, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
    linewidth=0.5
)
ax.set_ylim(0, 7)
ax.set_yticks(np.linspace(0, 7, 8))


print(ttest_rel(CorrData['Corr'], CorrData['Shuffle']))
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(CorrData['Corr'], bins=20, range=(-0.2, 0.6), alpha=0.5)
ax.hist(CorrData['Shuffle'], bins=20, range=(-0.2, 0.6), alpha=0.5)
ax.set_ylim(0, 15)
ax.set_yticks(np.linspace(0, 15, 6))
plt.savefig(os.path.join(loc, 'Field Number Correlation.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Field Number Correlation.svg'), dpi=600)
plt.close()

fig = plt.figure(figsize=(3,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.set_aspect('equal')
idx_cis = np.where((Data['MiceID'] == 10209)&(Data['Training Day'] == 'Day 5')&(Data['Direction'] == 'Cis'))[0]
idx_trs = np.where((Data['MiceID'] == 10209)&(Data['Training Day'] == 'Day 5')&(Data['Direction'] == 'Trs'))[0]
ax.plot([0, 15], [0, 15], color='red', linewidth=0.5, ls=':')
ax.plot(
    Data['Field Number'][idx_cis] + np.random.rand(idx_cis.shape[0])*0.6-0.3, 
    Data['Field Number'][idx_trs] + np.random.rand(idx_trs.shape[0])*0.6-0.3,
    'o', color = 'black', markersize=2, markeredgewidth=0
)
ax.axis([0, 15, 0, 15])
ax.set_xticks(np.linspace(0, 15, 6))
ax.set_yticks(np.linspace(0, 15, 6))
plt.savefig(os.path.join(loc, 'Example.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Example.svg'), dpi=600)
plt.close()

fig = plt.figure(figsize=(3,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.set_aspect('equal')
idx_cis = np.where((HPData['MiceID'] == 10227)&(HPData['Training Day'] == 'Day 4')&(HPData['Direction'] == 'Cis'))[0]
idx_trs = np.where((HPData['MiceID'] == 10227)&(HPData['Training Day'] == 'Day 4')&(HPData['Direction'] == 'Trs'))[0]
ax.plot([0, 15], [0, 15], color='red', linewidth=0.5, ls=':')
ax.plot(
    HPData['Field Number'][idx_cis] + np.random.rand(idx_cis.shape[0])*0.6-0.3, 
    HPData['Field Number'][idx_trs] + np.random.rand(idx_trs.shape[0])*0.6-0.3,
    'o', color = 'black', markersize=2, markeredgewidth=0
)
ax.axis([0, 15, 0, 15])
ax.set_xticks(np.linspace(0, 15, 6))
ax.set_yticks(np.linspace(0, 15, 6))
ax.set_title("0.437")
plt.savefig(os.path.join(loc, 'Example Hairpin.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Example Hairpin.svg'), dpi=600)
plt.close()

"""
for i in range(len(f3)):
    with open(f3['Trace File'][i], 'rb') as handle:
        trace = pickle.load(handle)
    
    print(i, int(f3['MiceID'][i]), int(f3['date'][i]), int(f3['session'][i]))
    trace['cis']['place_field_all'] = place_field(trace['cis'], 2, 0.4)
    trace['cis'] = count_field_number(trace['cis'])
    trace['trs']['place_field_all'] = place_field(trace['trs'], 2, 0.4)
    trace['trs'] = count_field_number(trace['trs'])
    
    with open(f3['Trace File'][i], 'wb') as handle:
        pickle.dump(trace, handle)
"""