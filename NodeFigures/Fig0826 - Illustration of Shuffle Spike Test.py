from mylib.statistic_test import *
from mylib.field.tracker import TrackerDsp

code_id = "0826 - Illustration of Shuffle Spike Test"
loc = join(figpath, "Dsp", code_id)
mkdir(loc)

def visualize_linear(trace, cells, save_loc: str):
    mkdir(save_loc)
    for i in cells:
        print(f"Plot Cell {i}...")
        
        fig = plt.figure(figsize = (4,1))
        ax = Clear_Axes(
            plt.axes(), 
            close_spines=['top', 'right'], 
            ifxticks=True,
            ifyticks=True
        )

        colors = [0, 1, 2, 3, 0, 0, 4, 5, 6, 0]
        for j in range(10):
            LinearizedRateMapAxes(
                ax, 
                trace[f'node {j}']['old_map_clear'][i, :],
                maze_type=1,
                smooth_window_length=20,
                sigma=1,
                color=DSPPalette[colors[j]],
                linewidth=0.5
            )
            
        peak = np.max([np.max(trace[f'node {j}']['old_map_clear'][i, :]) for j in range(10)])
        ax.set_ylim(-peak*0.1, round(peak, 2))
        ax.set_yticks([0, round(peak, 2)])
            
        plt.savefig(os.path.join(save_loc, f"Cell {i+1}.png"), dpi = 600)
        plt.savefig(os.path.join(save_loc, f"Cell {i+1}.svg"), dpi = 600)
        plt.close()
        
def visualize_summed(trace, cells, save_loc: str):
    mkdir(save_loc)
    for i in cells:
        print(f"Plot Cell {i}...")
        
        fig = plt.figure(figsize = (4,1))
        ax = Clear_Axes(
            plt.axes(), 
            close_spines=['top', 'right'], 
            ifxticks=True,
            ifyticks=True
        )

        colors = [0, 1, 2, 3, 0, 0, 4, 5, 6, 0]
        rate_map = np.vstack([trace[f'node {j}']['old_map_clear'][i, :] for j in range(10)])
        rate_map = np.sum(rate_map, axis = 0)
        LinearizedRateMapAxes(
                ax, 
                content = rate_map,
                maze_type=1,
                smooth_window_length=20,
                sigma=1,
                color='k',
                linewidth=0.5
        )
            
        peak = np.max(rate_map)
        ax.set_ylim(-peak*0.1, round(peak, 2))
        ax.set_yticks([0, round(peak, 2)])
            
        plt.savefig(os.path.join(save_loc, f"Cell {i+1} [summed].png"), dpi = 600)
        plt.savefig(os.path.join(save_loc, f"Cell {i+1} [summed].svg"), dpi = 600)
        plt.close()
        
def visualize_summed_with_field(trace, cells, save_loc: str):
    mkdir(save_loc)
    for i in cells:
        print(f"Plot Cell {i}...")
        
        fig = plt.figure(figsize = (4,1))
        ax = Clear_Axes(
            plt.axes(), 
            close_spines=['top', 'right'], 
            ifxticks=True,
            ifyticks=True
        )

        colors = [0, 1, 2, 3, 0, 0, 4, 5, 6, 0]
        rate_map = np.vstack([trace[f'node {j}']['old_map_clear'][i, :] for j in range(10)])
        rate_map = np.sum(rate_map, axis = 0)
        LinearizedRateMapAxes(
                ax, 
                content = rate_map,
                maze_type=1,
                smooth_window_length=20,
                sigma=1,
                color='k',
                linewidth=0.5
        )
            
        peak = np.max(rate_map)
        ax.set_ylim(-peak*0.1, round(peak, 2))
        ax.set_yticks([0, round(peak, 2)])
        
        shadow_colors = sns.color_palette("rainbow", len(trace['place_field_all'][i].keys()))
        print(trace['place_field_all'][i].keys())
        for j, k in enumerate(trace['place_field_all'][i].keys()):
            field_area = np.unique(spike_nodes_transform(trace['place_field_all'][i][k], 12))
            print(k, field_area)
            for l in field_area:
                ax.fill_betweenx(y=[0, peak], x1=NRG[1][l]-0.5, x2 = NRG[1][l]+0.5, alpha=0.3, edgecolor=None, linewidth=0, color = shadow_colors[j])
        
        plt.savefig(os.path.join(save_loc, f"Cell {i+1} [summed with field].png"), dpi = 600)
        plt.savefig(os.path.join(save_loc, f"Cell {i+1} [summed with field].svg"), dpi = 600)
        plt.close()

"""      
with open(r"E:\Data\Dsp_maze\10227\20231010\trace.pkl", 'rb') as f:
    trace = pickle.load(f)        
visualize_linear(trace=trace, cells=np.array([9, 10, 19])-1, save_loc=join(loc, "27-S2"))
"""
with open(r"E:\Data\Dsp_maze\10224\20231015\trace.pkl", 'rb') as f:
    trace = pickle.load(f)        
"""
visualize_linear(trace=trace, cells=np.array([49, 51])-1, save_loc=join(loc, "24-S7"))
visualize_summed(trace=trace, cells=np.array([49])-1, save_loc=join(loc, "24-S7"))

visualize_summed_with_field(trace=trace, cells=np.array([49])-1, save_loc=join(loc, "24-S7"))

TrackerDsp.visualize_single_field(
    trace=trace,
    cell=4-1,
    field_center=1880,
    n_shuffle=10000,
    save_loc=join(loc, "24-S7"),
    file_name="Cell 4 Field 1880 [with shuffle]"
)
"""

def RouteIndependentField(trace):
    field_info = trace['field_info']
    for i in tqdm(range(field_info.shape[1])):
        cell, field_center = int(field_info[0, i, 0])-1, int(field_info[0, i, 2])
        
        TrackerDsp.visualize_single_field(
            trace=trace,
            cell=cell,
            field_center=field_center,
            n_shuffle=10000,
            save_loc=join(trace['p'], "RouteIndependentField"),
            file_name=f"Cell {cell+1} Field {field_center}"
        )
        
#RouteIndependentField(trace)

# Example Fields
with open(f2['Trace File'][11], 'rb') as handle:
    trace = pickle.load(handle)
    

TrackerDsp.visualize_single_field(
    trace=trace,
    cell=9-1,
    field_center=1890,
    n_shuffle=10000,
    save_loc=join(loc, "Map-Swithing Fields"),
    file_name="12-S6-Cell 9-Field 1890"
)

with open(f2['Trace File'][24], 'rb') as handle:
    trace = pickle.load(handle)


TrackerDsp.visualize_single_field(
    trace=trace,
    cell=47-1,
    field_center=1884,
    n_shuffle=10000,
    save_loc=join(loc, "Map-Swithing Fields"),
    file_name="24-S6-Cell 47-Field 1884"
)

TrackerDsp.visualize_single_field(
    trace=trace,
    cell=47-1,
    field_center=1892,
    n_shuffle=10000,
    save_loc=join(loc, "Map-Swithing Fields"),
    file_name="24-S6-Cell 47-Field 1892"
)

TrackerDsp.visualize_single_field(
    trace=trace,
    cell=8-1,
    field_center=2104,
    n_shuffle=10000,
    save_loc=join(loc, "Map-Swithing Fields"),
    file_name="24-S6-Cell 8"
)

with open(f2['Trace File'][27], 'rb') as handle:
    trace = pickle.load(handle)
    
TrackerDsp.visualize_single_field(
    trace=trace,
    cell=5-1,
    field_center=2033,
    n_shuffle=10000,
    save_loc=join(loc, "Map-Swithing Fields"),
    file_name="27-S7-Cell 5"
)
