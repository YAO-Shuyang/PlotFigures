from mylib.statistic_test import *
from scipy.stats import binned_statistic_2d, binned_statistic

code_id = "0408 - Conditional Indipendent Test"
loc = os.path.join(figpath, code_id)
mkdir(loc)

idx = np.where((f1['maze_type'] != 0))[0]
if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Chi2 Statistic', 'Mutual Information', 'Field Pair Type', 'Variable', 'Pair Num'], f = f1,
                              function = FieldPropertyIndependence_Chi2_MI_DoubleCheck_Interface, file_idx=idx,
                              file_name = code_id, behavior_paradigm = 'CrossMaze')
    
    with open(join(figdata, code_id+'.pkl'), 'wb') as f:
        pickle.dump(Data, f)


bar_colors = [sns.color_palette("rocket", 3)[0], sns.color_palette("Blues", 3)[0], 
              sns.color_palette("rocket", 3)[1], sns.color_palette("Blues", 3)[1], 
              sns.color_palette("rocket", 3)[2], sns.color_palette("Blues", 3)[2]]

# Hex codes for the dot colors (lighter shades of the bar colors)
dot_colors = [
    "#AED581",  # Light Green
    "#FFCC80",  # Light Orange
    "#9FA8DA",  # Light Indigo
    "#EF9A9A",  # Light Red
    "#CE93D8",  # Light Purple
    "#80CBC4",  # Light Teal
]

idx = np.where((Data['MiceID'] != 11095) & (Data['MiceID'] != 11092))
Data = SubDict(Data, Data.keys(), idx)
Data['hue'] = np.array([Data['Maze Type'][i] + Data['Field Pair Type'][i] for i in range(Data['Maze Type'].shape[0])])
    
# Plot an example to showcase the P(x, y) - P(x)P(y) table
stab_sib, stab_non = [], []
size_sib, size_non = [], []
rate_sib, rate_non = [], []
"""
with open(r"E:\Data\Cross_maze\10227\20230930\session 2\trace.pkl", "rb") as handle:
    trace = pickle.load(handle)
field_reg = trace['field_reg']
for i in range(field_reg.shape[0]):
    for j in range(field_reg.shape[0]):
        if i == j:
            continue
            
        if field_reg[i, 0] == field_reg[j, 0]:
            stab_sib.append([field_reg[i, 5], field_reg[j, 5]])
            size_sib.append([field_reg[i, 3], field_reg[j, 3]])
            rate_sib.append([field_reg[i, 4], field_reg[j, 4]])
        else:
            stab_non.append([field_reg[i, 5], field_reg[j, 5]])
            size_non.append([field_reg[i, 3], field_reg[j, 3]])
            rate_non.append([field_reg[i, 4], field_reg[j, 4]])
                
stab_sib = np.array(stab_sib, np.float64)
stab_non = np.array(stab_non, np.float64)[np.random.randint(0, len(stab_non), len(stab_sib)), :]
size_sib = np.array(size_sib, np.int64)
size_non = np.array(size_non, np.int64)[np.random.randint(0, len(size_non), len(size_sib)), :]
rate_sib = np.array(rate_sib, np.float64)
rate_non = np.array(rate_non, np.float64)[np.random.randint(0, len(rate_non), len(rate_sib)), :]

def indept_field_properties(
    real_distribution: np.ndarray,
    X_pairs: np.ndarray,
    Y_pairs: np.ndarray,
    n_bin = 20
) -> np.ndarray:
    real_distribution = real_distribution[np.isnan(real_distribution) == False]
    X_pairs = X_pairs[np.where((np.isnan(X_pairs[:, 0]) == False) & (np.isnan(X_pairs[:, 1]) == False))[0], :]
    Y_pairs = Y_pairs[np.where((np.isnan(Y_pairs[:, 0]) == False) & (np.isnan(Y_pairs[:, 1]) == False))[0], :]
    
    _range = [np.min(real_distribution), np.max(real_distribution)+0.0001]
    P, _, _ = binned_statistic(
        x=real_distribution,
        values=None,
        statistic="count",
        bins=n_bin,
        range=_range
    )
    P = P / real_distribution.shape[0] # Calculate the probability mass function
    
    observed_joint_freq_X, _, _, binnumber_X = binned_statistic_2d(
        x=X_pairs[:, 0],
        y=X_pairs[:, 1],
        values=None,
        statistic="count",
        bins=[n_bin, n_bin],
        range=[_range, _range],
        expand_binnumbers=True
    )
    
    observed_joint_freq_Y, _, _, binnumber_Y = binned_statistic_2d(
        x=Y_pairs[:, 0],
        y=Y_pairs[:, 1],
        values=None,
        statistic="count",
        bins=[n_bin, n_bin],
        range=[_range, _range],
        expand_binnumbers=True
    )
    
    joint_p = np.outer(P, P)
    expected_joint_freq_X = joint_p*len(X_pairs) + 0.001
    expected_joint_freq_Y = joint_p*len(Y_pairs) + 0.001
    
    observed_joint_freq_X = observed_joint_freq_X + 0.001
    observed_joint_freq_Y = observed_joint_freq_Y + 0.001
    
    return observed_joint_freq_X, observed_joint_freq_Y, expected_joint_freq_Y


X, Y, E = indept_field_properties(real_distribution=field_reg[:, 3], X_pairs=size_sib, Y_pairs=size_non, n_bin=20)
res_X = X / np.sum(X) - E / np.sum(E)
res_Y = Y / np.sum(Y) - E / np.sum(E)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6), gridspec_kw={'height_ratios': [1, 3], 'width_ratios': [2, 2]})
ax1 = Clear_Axes(axes[0, 0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax2, ax3, ax4 = Clear_Axes(axes[0, 1]), Clear_Axes(axes[1, 0]), Clear_Axes(axes[1, 1])
ax1.hist(field_reg[:, 3], bins=20, density=True, color='lightgrey', range=(np.nanmin(field_reg[:, 3]), np.nanmax(field_reg[:, 3])+0.0001))
ax1.set_xticks(np.linspace(np.nanmin(field_reg[:, 3]), np.nanmax(field_reg[:, 3])+0.0001, 21))
_vmin, _vmax = min(np.min(res_X), np.min(res_Y)), max(np.max(res_X), np.max(res_Y))
_bound = np.max(np.abs([_vmin, _vmax]))
sns.heatmap(res_X, cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True), vmin=-_bound, vmax=_bound, ax = ax3)
sns.heatmap(res_Y, cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True), vmin=-_bound, vmax=_bound, ax = ax4)
ax3.set_aspect('equal')
ax4.set_aspect('equal')
plt.savefig(join(loc, "Example for Chi square test.png"), dpi=600)
plt.savefig(join(loc, "Example for Chi square test.svg"), dpi=600)
plt.close()
"""

# Statistic comparison of sibling field pairs with non-sibling field pairs
# Chi-square test for P(X, Y) with P(X)P(Y)
# 1. Nubmer of sibling field pairs used for analysis
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
idx = np.where((Data['Variable'] == 'Stability')&(Data['Field Pair Type'] == 'Sibling'))[0]
SubData = SubDict(Data, Data.keys(), idx=idx)
sns.barplot(
    x='MiceID',
    y='Pair Num',
    data = SubData,
    hue = 'Maze Type',
    palette=['#003366', '#66CCCC', '#D4C9A8'],
    ax = ax,
    errwidth=0.5,
    capsize=0.15,
    errcolor='black',
    width = 0.8
)
"""
sns.stripplot(
    x='MiceID',
    y='Pair Num',
    data = SubData,
    hue = 'Maze Type',
    ax = ax,
    palette=sns.color_palette("Blues", 3),
    edgecolor='black',
    size=2,
    linewidth=0.1,
    jitter=0.1,
    dodge=True
)"""
ax.set_ylim(0, 50000)
ax.set_yticks(np.linspace(0, 50000, 6))
plt.savefig(join(loc, "Number of Field Pairs used for statistic test.png"), dpi=600)
plt.savefig(join(loc, "Number of Field Pairs used for statistic test.svg"), dpi=600)
plt.close()

# Chi-square test for P(X, Y) with P(X)P(Y)
# 2. Compare of Chi-square statistics.
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x='Variable',
    y='Chi2 Statistic',
    data = Data,
    hue = 'hue',
    palette=['#003366', '#0099CC','#66CCCC', '#99CCFF'], #  , '#D4C9A8', '#8E9F85'
    hue_order=['Maze 1Sibling', 'Maze 1Non-sibling', 'Maze 2Sibling', 'Maze 2Non-sibling'], # 'Open FieldSibling', 'Open FieldNon-sibling', 
    ax = ax,
    errwidth=0.5,
    capsize=0.08,
    errcolor='black',
    width = 0.8
)

sns.stripplot(
    x='Variable',
    y='Chi2 Statistic',
    data = Data,
    hue = 'hue',
    hue_order=['Maze 1Sibling', 'Maze 1Non-sibling', 'Maze 2Sibling', 'Maze 2Non-sibling'], # 'Open FieldSibling', 'Open FieldNon-sibling',
    ax = ax,
    palette='Blues',
    edgecolor='black',
    size=1,
    linewidth=0.05,
    jitter=0.1,
    dodge=True
)
ax.set_ylim(0, 3000)
ax.set_yticks(np.linspace(0, 3000, 7))
plt.savefig(join(loc, "Chi2 Statistic.png"), dpi=600)
plt.savefig(join(loc, "Chi2 Statistic.svg"), dpi=600)
plt.close()

print("Open Field ----------------------------------------------------------------")
idx_sib = np.where((Data['Variable'] == 'Stability')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Open Field'))[0]
idx_non = np.where((Data['Variable'] == 'Stability')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Open Field'))[0]
print("  [Stability] T Test for Chi2 Statistics: ", ttest_ind(Data['Chi2 Statistic'][idx_sib], Data['Chi2 Statistic'][idx_non]))
idx_sib = np.where((Data['Variable'] == 'Size')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Open Field'))[0]
idx_non = np.where((Data['Variable'] == 'Size')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Open Field'))[0]
print("  [Size] T Test for Chi2 Statistics: ", ttest_ind(Data['Chi2 Statistic'][idx_sib], Data['Chi2 Statistic'][idx_non]))
idx_sib = np.where((Data['Variable'] == 'Rate')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Open Field'))[0]
idx_non = np.where((Data['Variable'] == 'Rate')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Open Field'))[0]
print("  [Rate] T Test for Chi2 Statistics: ", ttest_ind(Data['Chi2 Statistic'][idx_sib], Data['Chi2 Statistic'][idx_non]), end='\n\n')

print("Maze 1 ----------------------------------------------------------------")
idx_sib = np.where((Data['Variable'] == 'Stability')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Maze 1'))[0]
idx_non = np.where((Data['Variable'] == 'Stability')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Maze 1'))[0]
print("  [Stability] T Test for Chi2 Statistics: ", ttest_ind(Data['Chi2 Statistic'][idx_sib], Data['Chi2 Statistic'][idx_non]))
idx_sib = np.where((Data['Variable'] == 'Size')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Maze 1'))[0]
idx_non = np.where((Data['Variable'] == 'Size')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Maze 1'))[0]
print("  [Size] T Test for Chi2 Statistics: ", ttest_ind(Data['Chi2 Statistic'][idx_sib], Data['Chi2 Statistic'][idx_non]))
idx_sib = np.where((Data['Variable'] == 'Rate')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Maze 1'))[0]
idx_non = np.where((Data['Variable'] == 'Rate')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Maze 1'))[0]
print("  [Rate] T Test for Chi2 Statistics: ", ttest_ind(Data['Chi2 Statistic'][idx_sib], Data['Chi2 Statistic'][idx_non]), end='\n\n')

print("Maze 2 ----------------------------------------------------------------")
idx_sib = np.where((Data['Variable'] == 'Stability')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Maze 2'))[0]
idx_non = np.where((Data['Variable'] == 'Stability')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Maze 2'))[0]
print("  [Stability] T Test for Chi2 Statistics: ", ttest_ind(Data['Chi2 Statistic'][idx_sib], Data['Chi2 Statistic'][idx_non]))
idx_sib = np.where((Data['Variable'] == 'Size')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Maze 2'))[0]
idx_non = np.where((Data['Variable'] == 'Size')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Maze 2'))[0]
print("  [Size] T Test for Chi2 Statistics: ", ttest_ind(Data['Chi2 Statistic'][idx_sib], Data['Chi2 Statistic'][idx_non]))
idx_sib = np.where((Data['Variable'] == 'Rate')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Maze 2'))[0]
idx_non = np.where((Data['Variable'] == 'Rate')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Maze 2'))[0]
print("  [Rate] T Test for Chi2 Statistics: ", ttest_ind(Data['Chi2 Statistic'][idx_sib], Data['Chi2 Statistic'][idx_non]), end='\n\n')

# Mutual Information
# 3. Compare the Mutual information between each pair of field
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.barplot(
    x='Variable',
    y='Mutual Information',
    data = Data,
    hue = 'hue',
    hue_order=['Maze 1Sibling', 'Maze 1Non-sibling', 'Maze 2Sibling', 'Maze 2Non-sibling'], #'Open FieldSibling', 'Open FieldNon-sibling', 
    palette=['#003366', '#0099CC', '#66CCCC', '#99CCFF'], # , '#D4C9A8', '#8E9F85'
    ax = ax,
    errwidth=0.5,
    capsize=0.08,
    errcolor='black',
    width = 0.8
)
sns.stripplot(
    x='Variable',
    y='Mutual Information',
    data = Data,
    hue = 'hue',
    hue_order=['Maze 1Sibling', 'Maze 1Non-sibling', 'Maze 2Sibling', 'Maze 2Non-sibling'], #'Open FieldSibling', 'Open FieldNon-sibling', 
    ax = ax,
    palette='Blues',
    edgecolor='black',
    size=1,
    linewidth=0.05,
    jitter=0.1,
    dodge=True
)
ax.set_ylim(0, 0.05)
ax.set_yticks(np.linspace(0, 0.05, 6))
plt.savefig(join(loc, "Mutual Information.png"), dpi=600)
plt.savefig(join(loc, "Mutual Information.svg"), dpi=600)
plt.close()

print("Open Field ----------------------------------------------------------------")
idx_sib = np.where((Data['Variable'] == 'Stability')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Open Field'))[0]
idx_non = np.where((Data['Variable'] == 'Stability')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Open Field'))[0]
print("  [Stability] T Test for Mutual Informations: ", ttest_ind(Data['Mutual Information'][idx_sib], Data['Mutual Information'][idx_non]))
idx_sib = np.where((Data['Variable'] == 'Size')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Open Field'))[0]
idx_non = np.where((Data['Variable'] == 'Size')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Open Field'))[0]
print("  [Size] T Test for Mutual Informations: ", ttest_ind(Data['Mutual Information'][idx_sib], Data['Mutual Information'][idx_non]))
idx_sib = np.where((Data['Variable'] == 'Rate')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Open Field'))[0]
idx_non = np.where((Data['Variable'] == 'Rate')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Open Field'))[0]
print("  [Rate] T Test for Mutual Informations: ", ttest_ind(Data['Mutual Information'][idx_sib], Data['Mutual Information'][idx_non]), end='\n\n')

print("Maze 1 ----------------------------------------------------------------")
idx_sib = np.where((Data['Variable'] == 'Stability')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Maze 1'))[0]
idx_non = np.where((Data['Variable'] == 'Stability')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Maze 1'))[0]
print("  [Stability] T Test for Mutual Informations: ", ttest_ind(Data['Mutual Information'][idx_sib], Data['Mutual Information'][idx_non]))
idx_sib = np.where((Data['Variable'] == 'Size')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Maze 1'))[0]
idx_non = np.where((Data['Variable'] == 'Size')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Maze 1'))[0]
print("  [Size] T Test for Mutual Informations: ", ttest_ind(Data['Mutual Information'][idx_sib], Data['Mutual Information'][idx_non]))
idx_sib = np.where((Data['Variable'] == 'Rate')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Maze 1'))[0]
idx_non = np.where((Data['Variable'] == 'Rate')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Maze 1'))[0]
print("  [Rate] T Test for Mutual Informations: ", ttest_ind(Data['Mutual Information'][idx_sib], Data['Mutual Information'][idx_non]), end='\n\n')

print("Maze 2 ----------------------------------------------------------------")
idx_sib = np.where((Data['Variable'] == 'Stability')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Maze 2'))[0]
idx_non = np.where((Data['Variable'] == 'Stability')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Maze 2'))[0]
print("  [Stability] T Test for Mutual Informations: ", ttest_ind(Data['Mutual Information'][idx_sib], Data['Mutual Information'][idx_non]))
idx_sib = np.where((Data['Variable'] == 'Size')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Maze 2'))[0]
idx_non = np.where((Data['Variable'] == 'Size')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Maze 2'))[0]
print("  [Size] T Test for Mutual Informations: ", ttest_ind(Data['Mutual Information'][idx_sib], Data['Mutual Information'][idx_non]))
idx_sib = np.where((Data['Variable'] == 'Rate')&(Data['Field Pair Type'] == 'Sibling')&(Data['Maze Type'] == 'Maze 2'))[0]
idx_non = np.where((Data['Variable'] == 'Rate')&(Data['Field Pair Type'] == 'Non-sibling')&(Data['Maze Type'] == 'Maze 2'))[0]
print("  [Rate] T Test for Mutual Informations: ", ttest_ind(Data['Mutual Information'][idx_sib], Data['Mutual Information'][idx_non]), end='\n\n')

