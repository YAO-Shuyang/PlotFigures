from mylib.statistic_test import *

code_id = '0818 - Segmented Neural Trajectory'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Segments', 'Routes', 'Group', 'Mean Trajectory Distance', 'SubSpace Type'],
                              f=f2, 
                              function = SegmentedTrajectoryDistance_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

routes_order = np.array([0, 4, 1, 5, 2, 6, 3])

fig, axes = plt.subplots(nrows=6, ncols=1, figsize = (6, 6*2))
for j in range(1, 7):
    idx = np.where(
        (Data['SubSpace Type'] == 0) & (Data['Segments'] == j)
    )[0]
    SubData = SubDict(Data, Data.keys(), idx)
    
    ax = Clear_Axes(axes[j-1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    sns.barplot(
        x='Routes',
        y='Mean Trajectory Distance',
        hue='Group',
        data = SubData,
        ax = ax,
        palette=['#003366', '#0099CC', 'red'],
        width=0.8,
        capsize=0.1,
        errcolor='black',
        errwidth=0.5,
    )
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylim(0, 4)
    ax.set_xlim(-1,7)
plt.savefig(os.path.join(loc, 'Mean Trajectory Distance [Position Subspace].svg'))
plt.savefig(os.path.join(loc, 'Mean Trajectory Distance [Position Subspace].png'), dpi=600)
plt.close()

# Statistic Test
for j in range(1, 7):
    print(f"Segment {j} -------------------")
    for r in range(j+1):
        idx1 = np.where(
            (Data['SubSpace Type'] == 0) & 
            (Data['Segments'] == j) & 
            (Data['Routes'] == routes_order[r]) & 
            (Data['Group'] == "Within-Routes")
        )[0]
        idx2 = np.where(
            (Data['SubSpace Type'] == 0) & 
            (Data['Segments'] == j) & 
            (Data['Routes'] == routes_order[r]) & 
            (Data['Group'] == "Across-Routes")
        )[0]
        
        print(
            f"    Route {routes_order[r]}  ", 
            ttest_ind(
                Data['Mean Trajectory Distance'][idx1],
                Data['Mean Trajectory Distance'][idx2],
                alternative='less'
            )
        )
        print()

fig, axes = plt.subplots(nrows=6, ncols=1, figsize = (6, 6*2))
for j in range(1, 7):
    idx = np.where(
        (Data['SubSpace Type'] == 1) & (Data['Segments'] == j)
    )[0]
    SubData = SubDict(Data, Data.keys(), idx)
    
    ax = Clear_Axes(axes[j-1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    sns.barplot(
        x='Routes',
        y='Mean Trajectory Distance',
        hue='Group',
        data = SubData,
        ax = ax,
        palette=['#003366', '#0099CC', 'red'],
        width=0.8,
        capsize=0.1,
        errcolor='black',
        errwidth=0.5,
    )
    ax.set_xlim(-1,7)
plt.savefig(os.path.join(loc, 'Mean Trajectory Distance [Route Subspace].svg'))
plt.savefig(os.path.join(loc, 'Mean Trajectory Distance [Route Subspace].png'), dpi=600)
plt.close()

# Statistic Test
print("Route Subspace")
for j in range(1, 7):
    print(f"Segment {j} -------------------")
    for r in range(j+1):
        idx1 = np.where(
            (Data['SubSpace Type'] == 1) & 
            (Data['Segments'] == j) & 
            (Data['Routes'] == routes_order[r]) & 
            (Data['Group'] == "Within-Routes")
        )[0]
        idx2 = np.where(
            (Data['SubSpace Type'] == 1) & 
            (Data['Segments'] == j) & 
            (Data['Routes'] == routes_order[r]) & 
            (Data['Group'] == "Across-Routes")
        )[0]
        
        print(
            f"    Route {routes_order[r]}  ", 
            ttest_ind(
                Data['Mean Trajectory Distance'][idx1],
                Data['Mean Trajectory Distance'][idx2],
                alternative='less'
            )
        )
        if np.isnan(ttest_ind(
                Data['Mean Trajectory Distance'][idx1],
                Data['Mean Trajectory Distance'][idx2]
            )[0]):
            print(Data['Mean Trajectory Distance'][idx1], Data['Mean Trajectory Distance'][idx2])
print()