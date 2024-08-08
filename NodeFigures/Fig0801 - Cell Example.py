from mylib.statistic_test import *
from mylib.calcium.axes.loc_time_curve import LocTimeCurveAxes

code_id = '0801 - Cell Example'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)


def plot_cell_examples(
    mouse: int,
    date: int,
    cell_list: list
):
    idx = np.where(
        (f2['MiceID'] == mouse) &
        (f2['date'] == date)
    )[0]
    
    if len(idx) == 0:
        raise ValueError(f"{mouse} {date} not found!")
    
    with open(f2['Trace File'][idx[0]], 'rb') as handle:
        trace = pickle.load(handle)
        
    for cell in cell_list:
        fig = plt.figure(figsize=(3, 4))
        ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        
        colors = [
            DSPPalette[0], DSPPalette[1], DSPPalette[2], DSPPalette[3], DSPPalette[0],
            DSPPalette[0], DSPPalette[4], DSPPalette[5], DSPPalette[6], DSPPalette[0]]
        
        for i in range(10):
            ax = LocTimeCurveAxes(
                ax,
                behav_time=trace[f'node {i}']['ms_time_behav'],
                behav_nodes=spike_nodes_transform(trace[f'node {i}']['spike_nodes'], 12),
                spikes=trace[f'node {i}']['Spikes'][cell, :],
                spike_time=trace[f'node {i}']['ms_time_behav'],
                maze_type=1,
                line_kwargs={'linewidth': 0.4, 'color': colors[i]},
                bar_kwargs={'markeredgewidth': 0.5, 'markersize': 2, 'color': 'k'},
                is_include_incorrect_paths=True
            )[0]
            
        t1 = trace['node 4']['ms_time_behav'][-1]/1000
        t2 = trace['node 5']['ms_time_behav'][0]/1000
        t3 = trace['node 9']['ms_time_behav'][-1]/1000
        ax.set_yticks([0, t1, t2, t3], [0, t1, 0, t3-t2])
        
        plt.savefig(os.path.join(loc, f'{mouse}_{date}_cell_{cell+1}.png'), dpi=600)
        plt.savefig(os.path.join(loc, f'{mouse}_{date}_cell_{cell+1}.svg'), dpi=600)
        plt.close()
        

plot_cell_examples(10224, 20231012, np.array([213]) - 1)
#plot_cell_examples(10212, 20230601, np.array([1, 9, 11, 40]) - 1)
#plot_cell_examples(10227, 20231015, np.array([1, 3, 5, 17, 50, 93]) - 1)
#plot_cell_examples(10224, 20231014, np.array([8, 14, 47, 53, 54, 78, 79]) - 1)
