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

    
if os.path.exists(join(figdata, code_id+' Correlation [HP].pkl')):
    with open(join(figdata, code_id+' Correlation [HP].pkl'), 'rb') as handle:
        CorrHPData = pickle.load(handle)
else:
    CorrHPData = DataFrameEstablish(variable_names = ['Corr', 'Shuffle'], 
                              f = f4, function = PlaceFieldNumberPerDirectionCorr_Reverse_Interface, 
                              file_name = code_id + ' Correlation [HP]', behavior_paradigm = 'HairpinMaze')


print_estimator(CorrData['Corr'])
print(len(CorrData['Corr']))
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(CorrData['Corr'], bins=8, range=(0.35, 0.75), alpha=0.5)
ax.set_ylim(0, 10)
ax.set_yticks(np.linspace(0, 10, 6))
plt.savefig(os.path.join(loc, 'Field Number Correlation.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Field Number Correlation.svg'), dpi=600)
plt.close()

print_estimator(CorrHPData['Corr'])
print(len(CorrHPData['Corr']))
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(CorrHPData['Corr'], bins=12, range=(0.15, 0.75), alpha=0.5)
ax.set_ylim(0, 10)
ax.set_yticks(np.linspace(0, 10, 6))
plt.savefig(os.path.join(loc, 'Field Number Correlation [Hairpin].png'), dpi=600)
plt.savefig(os.path.join(loc, 'Field Number Correlation [Hairpin].svg'), dpi=600)
plt.close()


fig = plt.figure(figsize=(3,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.set_aspect('equal')
idx_cis = np.where((Data['MiceID'] == 10209)&(Data['Training Day'] == 'Day 5')&(Data['Direction'] == 'cis'))[0]
idx_trs = np.where((Data['MiceID'] == 10209)&(Data['Training Day'] == 'Day 5')&(Data['Direction'] == 'trs'))[0]
print('Reversed Maze Example: ', pearsonr(Data['Field Number'][idx_cis], Data['Field Number'][idx_trs]))
ax.plot([0, 20], [0, 20], color='red', linewidth=0.5, ls=':')
ax.plot(
    Data['Field Number'][idx_cis] + np.random.rand(idx_cis.shape[0])*0.6-0.3, 
    Data['Field Number'][idx_trs] + np.random.rand(idx_trs.shape[0])*0.6-0.3,
    'o', color = 'black', markersize=2, markeredgewidth=0
)
ax.axis([0, 20, 0, 20])
ax.set_xticks(np.linspace(0, 20, 5))
ax.set_yticks(np.linspace(0, 20, 5))
plt.savefig(os.path.join(loc, 'Example.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Example.svg'), dpi=600)
plt.close()

fig = plt.figure(figsize=(3,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.set_aspect('equal')
idx_cis = np.where((HPData['MiceID'] == 10227)&(HPData['Training Day'] == 'Day 4')&(HPData['Direction'] == 'cis'))[0]
idx_trs = np.where((HPData['MiceID'] == 10227)&(HPData['Training Day'] == 'Day 4')&(HPData['Direction'] == 'trs'))[0]
print("Hairpin example:", pearsonr(HPData['Field Number'][idx_cis], HPData['Field Number'][idx_trs]))
ax.plot([0, 25], [0, 25], color='red', linewidth=0.5, ls=':')
ax.plot(
    HPData['Field Number'][idx_cis] + np.random.rand(idx_cis.shape[0])*0.6-0.3, 
    HPData['Field Number'][idx_trs] + np.random.rand(idx_trs.shape[0])*0.6-0.3,
    'o', color = 'black', markersize=2, markeredgewidth=0
)
ax.axis([0, 25, 0, 25])
ax.set_xticks(np.linspace(0, 25, 6))
ax.set_yticks(np.linspace(0, 25, 6))
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