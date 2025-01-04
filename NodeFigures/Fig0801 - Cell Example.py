from mylib.statistic_test import *
from mylib.calcium.axes.loc_time_curve import LocTimeCurveAxes

code_id = '0801 - Cell Example'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)


def plot_cell_examples(
    mouse: int,
    date: int,
    cell_list: list,
    save_loc: str = loc
):
    idx = np.where(
        (f2['MiceID'] == mouse) &
        (f2['date'] == date)
    )[0]
    
    if len(idx) == 0:
        raise ValueError(f"{mouse} {date} not found!")
    
    with open(f2['Trace File'][idx[0]], 'rb') as handle:
        trace = pickle.load(handle)
        
    mkdir(save_loc)   
            
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
        
        plt.savefig(os.path.join(save_loc, f'{mouse}_{date}_cell_{cell+1}.png'), dpi=600)
        plt.savefig(os.path.join(save_loc, f'{mouse}_{date}_cell_{cell+1}.svg'), dpi=600)
        plt.close()
        

def plot_cell_examples_ego(
    mouse: int,
    date: int,
    cell_list: list,
    save_loc: str = loc
):
    idx = np.where(
        (f2['MiceID'] == mouse) &
        (f2['date'] == date)
    )[0]
    
    if len(idx) == 0:
        raise ValueError(f"{mouse} {date} not found!")
    
    with open(f2['Trace File'][idx[0]], 'rb') as handle:
        trace = pickle.load(handle)
        
        if isinstance(cell_list, str):
            cell_list = np.where(trace['SC'] == 1)[0]
        
    mkdir(save_loc)   
    
    for cell in tqdm(cell_list):
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
                is_include_incorrect_paths=True,
                is_ego=True,
                NRG=DSP_NRG[trace[f'node {i}']['Route']]
            )[0]
            
        t1 = trace['node 4']['ms_time_behav'][-1]/1000
        t2 = trace['node 5']['ms_time_behav'][0]/1000
        t3 = trace['node 9']['ms_time_behav'][-1]/1000
        ax.set_yticks([0, t1, t2, t3], [0, t1, 0, t3-t2])
        
        ax.set_title(f"Cell {cell+1}  {trace['SC_P'][cell]}  {round(trace['SC_OI'][cell], 4)}\n{trace['SC_EncodePath'][cell]+1}")
        ax.axvline(trace['SC_FieldCenter'][cell] / 0.6612 / 8, color='black', linewidth=0.5, ls=':')
        plt.savefig(os.path.join(save_loc, f'{mouse}_{date}_cell_{cell+1}.png'), dpi=600)
        plt.savefig(os.path.join(save_loc, f'{mouse}_{date}_cell_{cell+1}.svg'), dpi=600)
        plt.close()


#plot_cell_examples_ego(10227, 20231015, np.array([3,  23,  27,  31,  32, 567, 571, 578, 579, 582]), save_loc=join(loc, "Ego", "10227-20231015"))
#plot_cell_examples_ego(10224, 20231015, np.array([8,  33,  37,  49,  54]), save_loc=join(loc, "Ego", "10224-20231015"))
#plot_cell_examples_ego(10209, 20230524, np.array([205, 105]) - 1, save_loc=join(loc, "Ego"))
#plot_cell_examples_ego(10209, 20230601, np.array([86]) - 1, save_loc=join(loc, "Ego"))
#plot_cell_examples_ego(10209, 20230529, np.array([89]) - 1, save_loc=join(loc, "Ego"))
#plot_cell_examples_ego(10212, 20230601, np.array([1, 9, 11, 40]) - 1, save_loc=join(loc, "Ego"))
#plot_cell_examples_ego(10212, 20230601, np.array([22, 25, 26, 71]) - 1, save_loc=join(loc, "Ego"))
#plot_cell_examples(10212, 20230601, np.array([1, 9, 11, 40]) - 1)
#plot_cell_examples_ego(10224, 20231015, np.array([15, 80]) - 1, save_loc=join(loc, "Ego"))
#plot_cell_examples_ego(10224, 20231010, np.array([289, 316]) - 1, save_loc=join(loc, "Ego"))
#plot_cell_examples_ego(10227, 20231012, np.array([162, 121, 314, 59, 82]) - 1, save_loc=join(loc, "Ego"))

plot_cell_examples_ego(10232, 20241025, 'all', save_loc=join(loc, "10232 - Ego"))

#plot_cell_examples(10209, 20230602, np.array([233])-1)
#plot_cell_examples(10212, 20230528, np.array([21])-1)

#plot_cell_examples(10224, 20231009, np.array([124, 155]) - 1)
#plot_cell_examples(10224, 20231010, np.array([1, 9, 11, 37, 39, 47]) - 1)
#plot_cell_examples(10224, 20231014, np.array([8, 14, 47, 53, 54, 78, 79]) - 1)
#plot_cell_examples(10224, 20231012, np.array([213]) - 1)
#plot_cell_examples(10227, 20231009, np.array([1, 6, 8, 15, 18, 52]) - 1)
#plot_cell_examples(10227, 20231010, np.array([3, 12, 54]) - 1)
#plot_cell_examples(10227, 20231011, np.array([6, 13]) - 1)
#plot_cell_examples(10227, 20231012, np.array([8]) - 1)
#plot_cell_examples(10227, 20231013, np.array([38, 57, 77]) - 1)
#plot_cell_examples(10227, 20231014, np.array([6, 7, 8]) - 1)
#plot_cell_examples(10227, 20231015, np.array([1, 3, 5, 17, 50, 93]) - 1)
#plot_cell_examples(10232, 20241025, np.array([6, 8, 10, 14, 16, 18, 24, 39])-1)
#plot_cell_examples(10232, 20241026, np.array([2, 10, 11, 14, 62])-1)
#plot_cell_examples(10232, 20241027, np.array([2, 4, 6, 10, 23, 24, 46])-1)
#plot_cell_examples(10232, 20241030, np.array([6, 18, 20])-1)
#plot_cell_examples(10232, 20241031, np.array([1, 3, 4, 5, 9, 11, 12, 15, 31])-1)

