from mylib.statistic_test import *

code_id = "0319 - Test Results of Manually Checking CellReg"
loc = join(figpath, code_id)
mkdir(loc)

index_map_cellreg = ReadCellReg(r"E:\Data\Cross_maze\10224\Super Long-term Maze 1\Cell_reg\cellRegistered.mat")
index_map_cellreg = index_map_cellreg.astype(np.int64)

with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10224\neuromatch_res.pkl", 'rb') as handle:
    index_map_match = pickle.load(handle)
    index_map_match = index_map_match.astype(np.int64)
    
idx = np.where((f1['maze_type'] == 1)&(f1['MiceID'] == 10227))[0]


if os.path.exists(join(figdata, code_id+' 10224.pkl')) == False:
    diff_mat = np.zeros((idx.shape[0], idx.shape[0]), dtype=np.float64)

    for i in tqdm(range(idx.shape[0]-1)):
        for j in range(i+1, idx.shape[0]):
            with open(f1['Trace File'][idx[i]], 'rb') as handle:
                trace1 = pickle.load(handle)
            
            with open(f1['Trace File'][idx[j]], 'rb') as handle:
                trace2 = pickle.load(handle)
            
            cellpair1 = np.where((index_map_cellreg[i, :] > 0)&(index_map_cellreg[j, :] > 0))[0]
            cellpair2 = np.where((index_map_match[i, :] > 0)&(index_map_match[j, :] > 0))[0]
        
            corr1 = np.zeros(cellpair1.shape[0])
            corr2 = np.zeros(cellpair2.shape[0])
            for d in range(cellpair1.shape[0]):
                corr1[d], _ = pearsonr(
                    trace1['smooth_map_all'][index_map_cellreg[i, cellpair1[d]]-1, :],
                    trace2['smooth_map_all'][index_map_cellreg[j, cellpair1[d]]-1, :]
                )
        
            for d in range(cellpair2.shape[0]):
                corr2[d], _ = pearsonr(
                    trace1['smooth_map_all'][index_map_match[i, cellpair2[d]]-1, :],
                    trace2['smooth_map_all'][index_map_match[j, cellpair2[d]]-1, :]
                )
        
            diff_mat[i, j] = np.nanmean(corr1) - np.nanmean(corr2)
            diff_mat[j, i] = diff_mat[i, j]

    with open(join(figdata, code_id+' 10224.pkl'), 'wb') as f:
        pickle.dump(diff_mat, f)
else:
    with open(join(figdata, code_id+' 10224.pkl'), 'rb') as handle:
        diff_mat = pickle.load(handle)

"""
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes())
im = ax.imshow(diff_mat, cmap='coolwarm', vmin = -0.04, vmax = 0.04)
plt.colorbar(im, ax=ax)
ax.set_aspect("equal")
plt.savefig(join(loc, "diff of correlation 10224.svg"), dpi=600)
plt.savefig(join(loc, "diff of correlation 10224.png"), dpi=600)
plt.show()
"""
index_map = ReadCellReg(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Cell_reg\cellRegistered.mat")
with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10227\neuromatch_res.pkl", 'rb') as handle:
    index_map_check = pickle.load(handle)

print(index_map.shape, index_map_check.shape)
num = np.nansum(np.where(index_map > 0, 1, 0), axis=0)
num2 = np.nansum(np.where(index_map_check > 0, 1, 0), axis=0)
print(num, num2)
count = np.histogram(num, bins=26, range=(0.5, 26.5))
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(num, bins=26, range=(0.5, 26.5), rwidth=0.8, alpha=0.5, label = "cellreg")
ax.hist(num2, bins=26, range=(0.5, 26.5), rwidth=0.8, alpha=0.5, label="relabel")
ax.semilogy()
ax.legend()
plt.show()