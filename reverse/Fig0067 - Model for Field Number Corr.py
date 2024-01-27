from mylib.statistic_test import *

code_id = '0067 - Reverse, Field Number and Direction'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, '0068 - Reverse, Bidirectional Poisson Test.pkl')):
    with open(join(figdata, '0068 - Reverse, Bidirectional Poisson Test.pkl'), 'rb') as handle:
        RData = pickle.load(handle)
else:
    RData = DataFrameEstablish(variable_names = ['lam', 'KS Statistics', 'KS P-Value',
                                                                            'r', 'p', 'KS Gamma', 'KS Gamma P-value',
                                                                            'Direction'], 
                              f = f3, function = PoissonTest_Reverse_Interface, 
                              file_name = '0068 - Reverse, Bidirectional Poisson Test', behavior_paradigm = 'ReverseMaze')
    
if os.path.exists(join(figdata, '0067 - Reverse, Field Number and Direction Correlation.pkl')):
    with open(join(figdata, '0067 - Reverse, Field Number and Direction Correlation.pkl'), 'rb') as handle:
        CorrData = pickle.load(handle)
else:
    CorrData = DataFrameEstablish(variable_names = ['Corr', 'Shuffle'], 
                              f = f3, function = PlaceFieldNumberPerDirectionCorr_Reverse_Interface, 
                              file_name = '0067 - Reverse, Field Number and Direction Correlation', behavior_paradigm = 'ReverseMaze')

if os.path.exists(join(figdata, code_id+' [model].pkl')):
    with open(join(figdata, code_id+' [model].pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Corr', 'Data Type'], 
                              f = f3, function = ModelPlaceFieldNumberPerDirection_Reverse_Interface, 
                              file_name = code_id+' [model]', behavior_paradigm = 'ReverseMaze')

if os.path.exists(join(figdata, code_id+' [hairpin].pkl')):
    with open(join(figdata, code_id+' [hairpin].pkl'), 'rb') as handle:
        HData = pickle.load(handle)
else:
    HData = DataFrameEstablish(variable_names = ['Corr', 'Data Type'], 
                              f = f4, function = ModelPlaceFieldNumberPerDirection_Reverse_Interface, 
                              file_name = code_id+' [hairpin]', behavior_paradigm = 'HairpinMaze')



print_estimator(Data['Corr'][np.where(Data['Data Type'] == 'Data')])
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(Data['Corr'][np.where(Data['Data Type'] == 'Data')], bins=12, range=(-0, 0.6), color='gray')
ax.set_ylim(0, 12)
ax.set_yticks(np.linspace(0, 12, 7))
ax.set_xticks(np.linspace(0, 0.6, 7))
plt.savefig(os.path.join(loc, 'Field Number Correlation.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Field Number Correlation.svg'), dpi=600)
plt.close()

print_estimator(HData['Corr'][np.where(HData['Data Type'] == 'Data')])
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(HData['Corr'][np.where(HData['Data Type'] == 'Data')], bins=14, range=(-0.1, 0.6), color='gray')
ax.set_ylim(0, 6)
ax.set_yticks(np.linspace(0, 6, 4))
ax.set_xticks(np.linspace(-0.1, 0.6, 8))
plt.savefig(os.path.join(loc, 'Field Number Correlation [hairpin].png'), dpi=600)
plt.savefig(os.path.join(loc, 'Field Number Correlation [hairpin].svg'), dpi=600)
plt.close()
