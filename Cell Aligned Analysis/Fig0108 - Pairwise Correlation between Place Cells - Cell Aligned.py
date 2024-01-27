from mylib.statistic_test import *
from scipy.stats import linregress

# ================================================= Data Generation ==================================================================================
mice = 11095
maze_type = 1
row = 19

code_id = '0108 - Pairwise Correlation - Cell Aligned'
loc = os.path.join(figpath, 'Cell Aligned', code_id)
mkdir(loc)

def is_high_quilified_pair(trace1, trace2, i, j):
    return ((trace1['fir_sec_corr'][i] >= 0.6 or 
             trace1['odd_even_corr'][i] >= 0.6) and 
            (trace2['fir_sec_corr'][j] >= 0.6 or 
             trace2['odd_even_corr'][j] >= 0.6))

def pairwise_correlation(trace1, trace2, i, j):
    return pearsonr(trace1['smooth_rate_map'][i], trace2['smooth_rate_map'][j])[0]

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    idx = np.where((f1['MiceID'] == 11095)&(f1['date'] >= 20220820)&(f1['date'] <= 20220830)&(f1['maze_type'] == 1))[0]
    dateset = ['20220820','20220822','20220824','20220826','20220828','20220830']
    traceset = TraceFileSet(idx = idx, tp = r"E:\Data\Cross_maze")
    day = 6
    
    for trace in traceset:
        if 'laps' not in trace.keys():
            trace = CrossLapsCorrelation(trace, behavior_paradigm = trace['paradigm'])
            trace = OldMapSplit(trace)
        if trace['laps'] == 1:
            trace['fir_sec_corr'] = np.repeat(np.nan, trace['n_neuron'])
            trace['odd_even_corr'] = np.repeat(np.nan, trace['n_neuron'])
        if 'fir_sec_corr' not in trace.keys():
            trace = half_half_correlation(trace)
        if 'odd_even_corr' not in trace.keys():
            trace = odd_even_correlation(trace)
    
    # Maze 1
    index_map = Read_and_Sort_IndexMap(cellReg_95_maze1, occur_num=1)
    Mat = np.zeros((5, int(day*(day-1)/2)*index_map.shape[1]), dtype = np.float64)
    k = 0
    for n in range(index_map.shape[1]):
        for i in range(day-1):
            for j in range(i+1, day):
                
                if index_map[i, n] != 0 and index_map[j, n] != 0:
                    x, y = int(index_map[i, n]), int(index_map[j, n])
                    if traceset[i]['is_placecell'][x-1] == 1 and traceset[j]['is_placecell'][y-1] == 1:
                        Mat[0, k], _ = pearsonr(traceset[i]['smooth_map_all'][x-1],
                                                traceset[j]['smooth_map_all'][y-1])
                        Mat[2, k] = np.nanmean([traceset[i]['odd_even_corr'][x-1],
                                                traceset[j]['odd_even_corr'][y-1]])
                        Mat[3, k] = np.nanmean([traceset[i]['fir_sec_corr'][x-1],
                                                traceset[j]['fir_sec_corr'][y-1]])
                        Mat[4, k] = 1
                        if is_high_quilified_pair(traceset[i], traceset[j], x-1, y-1):
                            Mat[1, k] = 1
                        else:
                            Mat[1, k] = 0
                    else:
                        Mat[:, k] = np.nan    
                else:
                    Mat[:, k] = np.nan
                k += 1

    # Delete NAN
    a = np.sum(Mat, axis = 0)
    nan_idx = np.where(np.isnan(np.sum(Mat, axis = 0)))[0]
    Mat = np.delete(Mat, nan_idx, axis = 1)
    data = {'pair-wise correlation': cp.deepcopy(Mat[0, :]), 
            'Maze Type': cp.deepcopy(Mat[4, :]),
            'In-session stability': cp.deepcopy(Mat[1, :]),
            'In-session OEC': cp.deepcopy(Mat[2, :]),
            'In-session FSC': cp.deepcopy(Mat[3, :])}

    # Maze 2
    idx = np.where((f1['MiceID'] == 11095)&(f1['date'] >= 20220820)&(f1['date'] <= 20220830)&(f1['maze_type'] == 2))[0]
    dateset = ['20220820','20220822','20220824','20220826','20220828','20220830']
    traceset = TraceFileSet(idx = idx, tp = r"E:\Data\Cross_maze")
    day = 6
    
    for trace in traceset:
        if 'laps' not in trace.keys():
            trace = CrossLapsCorrelation(trace, behavior_paradigm = trace['paradigm'])
            trace = OldMapSplit(trace)
        if trace['laps'] == 1:
            trace['fir_sec_corr'] = np.repeat(np.nan, trace['n_neuron'])
            trace['odd_even_corr'] = np.repeat(np.nan, trace['n_neuron'])
        if 'fir_sec_corr' not in trace.keys():
            trace = half_half_correlation(trace)
        if 'odd_even_corr' not in trace.keys():
            trace = odd_even_correlation(trace)
    
    index_map = Read_and_Sort_IndexMap(cellReg_95_maze2, occur_num=1)
    Mat = np.zeros((5, int(day*(day-1)/2)*index_map.shape[1]), dtype = np.float64)
    k = 0
    for n in range(index_map.shape[1]):
        for i in range(day-1):
            for j in range(i+1, day):
                
                if index_map[i, n] != 0 and index_map[j, n] != 0:
                    x, y = int(index_map[i, n]), int(index_map[j, n])
                    if traceset[i]['is_placecell'][x-1] == 1 and traceset[j]['is_placecell'][y-1] == 1:
                        Mat[0, k], _ = pearsonr(traceset[i]['smooth_map_all'][x-1],
                                                traceset[j]['smooth_map_all'][y-1])
                        Mat[2, k] = np.nanmean([traceset[i]['odd_even_corr'][x-1],
                                                traceset[j]['odd_even_corr'][y-1]])
                        Mat[3, k] = np.nanmean([traceset[i]['fir_sec_corr'][x-1],
                                                traceset[j]['fir_sec_corr'][y-1]])
                        Mat[4, k] = 2
                        if is_high_quilified_pair(traceset[i], traceset[j], x-1, y-1):
                            Mat[1, k] = 1
                        else:
                            Mat[1, k] = 0
                    else:
                        Mat[:, k] = np.nan    
                else:
                    Mat[:, k] = np.nan
                k += 1
    nan_idx = np.where(np.isnan(np.sum(Mat, axis = 0)))[0]
    Mat = np.delete(Mat, nan_idx, axis = 1)
    
    data['pair-wise correlation'] = np.concatenate([data['pair-wise correlation'], Mat[0, :]])
    data['Maze Type'] = np.concatenate([data['Maze Type'], Mat[4, :]])
    data['In-session stability'] = np.concatenate([data['In-session stability'], Mat[1, :]])
    data['In-session OEC'] = np.concatenate([data['In-session OEC'], Mat[2, :]])
    data['In-session FSC'] = np.concatenate([data['In-session FSC'], Mat[3, :]])

    with open(os.path.join(figdata, code_id+'.pkl'), 'wb') as f:
        pickle.dump(data,f)
    
    Fs = pd.DataFrame(data)
    Fs.to_excel(os.path.join(figdata, code_id+'.xlsx'), index=False)

else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        data = pickle.load(handle)

def y(x, slope, intercepts):
    return slope*x + intercepts
 
fig = plt.figure(figsize=(2.5,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
colors = sns.color_palette("rocket", 2)
sns.barplot(x = 'In-session stability', 
            y = 'pair-wise correlation', 
            hue = 'Maze Type',
            data = data, 
            palette=colors,
            width = 0.6, capsize=0.05, errorbar=("ci",95), 
            errwidth=0.5, errcolor='black')
ax.legend(bbox_to_anchor = (1,1), loc = 'upper left', title='Maze Type', 
          title_fontsize = 8, fontsize = 8, ncol = 1, facecolor = 'white',
          edgecolor = 'white')

ax.set_ylabel("Pair-wise Correlation")
ax.set_yticks(np.linspace(0,1,6))
ax.set_xticks([0,1],['low', 'high'])
ax.axis([-0.5, 1.5, 0,1])
plt.tight_layout()
plt.savefig(os.path.join(loc, 'Cross-day Correlation (familiar).png'), dpi=2400)
plt.savefig(os.path.join(loc, 'Cross-day Correlation (familiar).svg'), dpi=2400)
plt.close()
print("Done Fig 1.")


idx_low = np.where((data['In-session stability'] == 0)&(data['Maze Type'] == 1))[0]
idx_hig = np.where((data['In-session stability'] == 1)&(data['Maze Type'] == 1))[0]
print("Fig 5D [maze 1]:", ttest_ind(data['pair-wise correlation'][idx_low], data['pair-wise correlation'][idx_hig]))
idx_low = np.where((data['In-session stability'] == 0)&(data['Maze Type'] == 2))[0]
idx_hig = np.where((data['In-session stability'] == 1)&(data['Maze Type'] == 2))[0]
print("Fig 5D [maze 2]:", ttest_ind(data['pair-wise correlation'][idx_low], data['pair-wise correlation'][idx_hig]))


x = np.linspace(np.nanmin(data['In-session OEC']),np.nanmax(data['In-session OEC']),1000)

fig = plt.figure(figsize=(5,2))
maze1_idx = np.where(data['Maze Type'] == 1)[0]
maze2_idx = np.where(data['Maze Type'] == 2)[0]
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(data['In-session OEC'][maze1_idx], data['pair-wise correlation'][maze1_idx])
print(linregress(data['In-session OEC'][maze1_idx], data['pair-wise correlation'][maze1_idx]))
slope2, intercept2, r_value2, p_value2, std_err2 = linregress(data['In-session OEC'][maze2_idx], data['pair-wise correlation'][maze2_idx])
print(linregress(data['In-session OEC'][maze2_idx], data['pair-wise correlation'][maze2_idx]))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.set_aspect('equal')
ax.axhline(0, ls=':', color = 'black', linewidth = 0.5)
colors = sns.color_palette("rocket", 2)
sns.scatterplot(x = 'In-session OEC', 
                y = 'pair-wise correlation', 
                hue = 'Maze Type',
                data = data, 
                size='In-session OEC', alpha = 0.5,
                palette=colors,
                sizes=(0.3,0.8)
                )
ax.plot(x, y(x, slope1, intercept1), color = colors[0], linewidth = 0.5,
        label = f'Maze 1\nr={round(r_value1,2)}\nk={round(slope1,2)}\nb={round(intercept1,2)}\np={p_value1}')
ax.plot(x, y(x, slope2, intercept2), color = colors[1], linewidth = 0.5,
        label = f'Maze 2\nr={round(r_value2,2)}\nk={round(slope2,2)}\nb={round(intercept2,2)}\np={p_value2}')
ax.legend(bbox_to_anchor = (1,1), loc = 'upper left', title='Maze Type', 
          title_fontsize = 8, fontsize = 8, ncol = 2, facecolor = 'white',
          edgecolor = 'white')

ax.set_ylabel("Pair-wise Correlation")
ax.set_xlabel("In-session stability (OEC)")
ax.set_yticks(np.linspace(-0.2,1,7))
ax.set_xticks(np.linspace(0,1,6))
ax.axis([0, 1, -0.2,1])
plt.tight_layout()
plt.savefig(os.path.join(loc, 'Cross-day Correlation vs. In-session OEC (familiar).png'), dpi=2400)
plt.savefig(os.path.join(loc, 'Cross-day Correlation vs. In-session OEC (familiar).svg'), dpi=2400)
plt.close()
print("Done Fig 2.")



x = np.linspace(np.nanmin(data['In-session FSC']),np.nanmax(data['In-session FSC']),1000)
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(data['In-session FSC'][maze1_idx], data['pair-wise correlation'][maze1_idx])
print(linregress(data['In-session FSC'][maze1_idx], data['pair-wise correlation'][maze1_idx]))
slope2, intercept2, r_value2, p_value2, std_err2 = linregress(data['In-session FSC'][maze2_idx], data['pair-wise correlation'][maze2_idx])
print(linregress(data['In-session FSC'][maze2_idx], data['pair-wise correlation'][maze2_idx]))

fig = plt.figure(figsize=(5,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.set_aspect('equal')
ax.axhline(0, ls=':', color = 'black', linewidth = 0.5)
colors = sns.color_palette("rocket", 2)
sns.scatterplot(x = 'In-session FSC', 
                y = 'pair-wise correlation', 
                hue = 'Maze Type',
                data = data, 
                size='In-session FSC', alpha = 0.5,
                palette=colors,
                sizes=(0.3,0.8)
                )
ax.plot(x, y(x, slope1, intercept1), color = colors[0], linewidth = 0.5,
        label = f'Maze 1\nr={round(r_value1,2)}\nk={round(slope1,2)}\nb={round(intercept1,2)}\np={p_value1}')
ax.plot(x, y(x, slope2, intercept2), color = colors[1], linewidth = 0.5,
        label = f'Maze 2\nr={round(r_value2,2)}\nk={round(slope2,2)}\nb={round(intercept2,2)}\np={p_value2}')
ax.legend(bbox_to_anchor = (1,1), loc = 'upper left', title='Maze Type', 
          title_fontsize = 8, fontsize = 8, ncol = 2, facecolor = 'white',
          edgecolor = 'white')
ax.set_ylabel("Pair-wise Correlation")
ax.set_xlabel("In-session stability (FSC)")
ax.set_yticks(np.linspace(-0.2,1,7))
ax.set_xticks(np.linspace(0,1,6))
ax.axis([0, 1, -0.2,1])

plt.tight_layout()
plt.savefig(os.path.join(loc, 'Cross-day Correlation vs. In-session FSC (familiar).png'), dpi=2400)
plt.savefig(os.path.join(loc, 'Cross-day Correlation vs. In-session FSC (familiar).svg'), dpi=2400)
plt.close()
print("Done Fig 3.")