import numpy as np
from mylib.statistic_test import *

code_id = "0056 - Naive Place Cell"
loc = join(figpath, code_id)
mkdir(loc)

def preprocess(
    mouse: int,
    stage: str,
    session: int,
    occu_num: int = 3,
    naive_num: int = 3
) -> np.ndarray:
    index_map = GetMultidayIndexmap(
        mouse = mouse,
        stage = stage,
        session = session,
        occu_num=occu_num
    )

    iscell_map = np.where(index_map!=0, 1, 0)

    trace_set = TraceFileSet(
        idx=np.where((f1['MiceID'] == mouse) & (f1['Stage'] == stage) & (f1['session'] == session))[0][:naive_num],
        f=f1
    )
    
    cellregidx = np.where((f_CellReg_day['MiceID'] == mouse)&(f_CellReg_day['session'] == session)&(f_CellReg_day['Stage'] == stage))[0]
    
    if len(cellregidx) == 0:
        raise ValueError(f"Mouse {mouse} does not have {stage} session {session} data.")
    
    cellregpath = f_CellReg_day['cellreg_folder'][cellregidx[0]]
    sfps = GetSFPSet(cellregpath, f1, np.where((f1['MiceID'] == mouse) & (f1['Stage'] == stage) & (f1['session'] == session))[0][:naive_num])

    idx = np.where(np.nansum(iscell_map[:naive_num, :], axis=0) == naive_num)[0]
    index_map = index_map[:naive_num, idx].astype(np.int64)
    
    return index_map, trace_set, sfps

import cv2
def _select_roi(sfp: np.ndarray):
    boundaries = []
    for i in range(sfp.shape[2]):
        # Find contours in the binary image
        image = np.uint8(sfp[:, :, i]*255)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the longest contour (assuming it is the boundary of the closed shape)
        longest_contour = max(contours, key=len)

        longest_contour = np.concatenate([longest_contour, [[[longest_contour[0, 0, 0], longest_contour[0, 0, 1]]]] ], axis=0)
        # Convert the contour points to numpy array
        traced_boundary = np.array(longest_contour)
        boundaries.append(traced_boundary)
        
    return boundaries

def _add_footprint(
    ax: Axes,
    sfp: np.ndarray,
    cell_id: int,
    edge: float = 20
) -> Axes:
    boundaries = _select_roi(sfp)
    n = sfp.shape[2]

    ax.set_aspect('equal')
    #im = ax.imshow(footprint, cmap = 'gray')
    #cbar = plt.colorbar(im, ax = ax)
    for i in range(n):
        if i == cell_id:
            continue
        color = (169/255, 169/255, 169/255)
        ax.plot(boundaries[i][:, 0, 0], boundaries[i][:, 0, 1], color = color, alpha=0.8)
    
    ax.plot(boundaries[cell_id][:, 0, 0], boundaries[cell_id][:, 0, 1], color = 'cornflowerblue', alpha=0.8)
    
    center = [(max(boundaries[cell_id][:, 0, 0]) + min(boundaries[cell_id][:, 0, 0]))/2, (max(boundaries[cell_id][:, 0, 1]) + min(boundaries[cell_id][:, 0, 1]))/2]
    
    ax.axis([center[0] - edge, center[0] + edge, center[1] - edge, center[1] + edge])
    
    return ax

from mylib import RateMapAxes, TraceMapAxes
def crossday_ratemap(
    index_map: np.ndarray,
    trace_set: list,
    sfps: list,
    save_loc: str,
    **imshow_kw
):  
    loc = join(save_loc, 'RateMap')
    mkdir(loc)
    n_neuron = index_map.shape[1]
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
    ax1, ax2, ax3 = Clear_Axes(axes[0, 0]), Clear_Axes(axes[0, 1]), Clear_Axes(axes[0, 2])
    ax4, ax5, ax6 = Clear_Axes(axes[1, 0]), Clear_Axes(axes[1, 1]), Clear_Axes(axes[1, 2])
    DrawMazeProfile(axes=ax1, maze_type=trace_set[0]['maze_type'], linewidth=1)
    DrawMazeProfile(axes=ax2, maze_type=trace_set[0]['maze_type'], linewidth=1)
    DrawMazeProfile(axes=ax3, maze_type=trace_set[0]['maze_type'], linewidth=1)
    
    colors = ['k', 'r']
    
    for i in tqdm(range(n_neuron)):
        ax1, im1, cbar1 = RateMapAxes(
            ax=ax1,
            content=trace_set[0]['LA']['smooth_map_all'][index_map[0, i]-1, :],
            maze_type=trace_set[0]['maze_type'],
            title=f"{round(trace_set[0]['LA']['SI_all'][index_map[0, i]-1], 2)}, {len(trace_set[0]['LA']['place_field_all'][index_map[0, i]-1])}",
            title_color=colors[trace_set[0]['LA']['is_placecell'][index_map[0, i]-1]],
            is_plot_maze_walls=False,
        )
        
        ax2, im2, cbar2 = RateMapAxes(
            ax=ax2,
            content=trace_set[1]['LA']['smooth_map_all'][index_map[1, i]-1, :],
            maze_type=trace_set[1]['maze_type'],
            title=f"{round(trace_set[1]['LA']['SI_all'][index_map[1, i]-1], 2)}, {len(trace_set[1]['LA']['place_field_all'][index_map[1, i]-1])}",
            title_color=colors[trace_set[1]['LA']['is_placecell'][index_map[1, i]-1]],
            is_plot_maze_walls=False
        )
        
        ax3, im3, cbar3 = RateMapAxes(
            ax=ax3,
            content=trace_set[2]['LA']['smooth_map_all'][index_map[2, i]-1, :],
            maze_type=trace_set[2]['maze_type'],
            title=f"{round(trace_set[2]['LA']['SI_all'][index_map[2, i]-1], 2)}, {len(trace_set[2]['LA']['place_field_all'][index_map[2, i]-1])}",
            title_color=colors[trace_set[2]['LA']['is_placecell'][index_map[2, i]-1]],
            is_plot_maze_walls=False
        )
        
        ax4 = _add_footprint(
            ax=ax4,
            sfp=sfps[0],
            cell_id=index_map[0, i]-1
        )
        
        ax5 = _add_footprint(
            ax=ax5,
            sfp=sfps[1],
            cell_id=index_map[1, i]-1
        )
        
        ax6 = _add_footprint(
            ax=ax6,
            sfp=sfps[2],
            cell_id=index_map[2, i]-1
        )
        
        ax1.axis([-0.6, 47.6, 47.6, -0.6])
        ax2.axis([-0.6, 47.6, 47.6, -0.6])
        ax3.axis([-0.6, 47.6, 47.6, -0.6])
        
        plt.savefig(join(loc, f"{index_map[0,i]}-{index_map[1,i]}-{index_map[2,i]}.png"), dpi = 600)
        plt.savefig(join(loc, f"{index_map[0,i]}-{index_map[1,i]}-{index_map[2,i]}.svg"), dpi = 600)

        ax4.clear(), ax5.clear(), ax6.clear()
        
        cbar1.remove()
        cbar2.remove()
        cbar3.remove()
        im1.remove()
        im2.remove()
        im3.remove()


def crossday_tracemap(
    index_map: np.ndarray,
    trace_set: list,
    sfps: list,
    save_loc: str,
    **imshow_kw
):  
    loc = join(save_loc, 'TraceMap')
    mkdir(loc)
    n_neuron = index_map.shape[1]
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
    ax1, ax2, ax3 = Clear_Axes(axes[0, 0]), Clear_Axes(axes[0, 1]), Clear_Axes(axes[0, 2])
    ax4, ax5, ax6 = Clear_Axes(axes[1, 0]), Clear_Axes(axes[1, 1]), Clear_Axes(axes[1, 2])
    
    colors = ['k', 'r']
    
    for i in tqdm(range(n_neuron)):
        ax1, a1, b1 = TraceMapAxes(
            ax=ax1,
            trajectory=cp.deepcopy(trace_set[0]['correct_pos']),
            behav_time=cp.deepcopy(trace_set[0]['correct_time']),
            spikes=cp.deepcopy(trace_set[0]['LA']['Spikes'][index_map[0, i]-1, :]),
            spike_time=cp.deepcopy(trace_set[0]['LA']['ms_time_behav']),
            maze_type=trace_set[0]['maze_type'],
            title=f"{round(trace_set[0]['LA']['SI_all'][index_map[0, i]-1], 2)}, {len(trace_set[0]['LA']['place_field_all'][index_map[0, i]-1])}",
            title_color=colors[trace_set[0]['LA']['is_placecell'][index_map[0, i]-1]],
        )
        
        ax2, a2, b2 = TraceMapAxes(
            ax=ax2,
            trajectory=cp.deepcopy(trace_set[1]['correct_pos']),
            behav_time=cp.deepcopy(trace_set[1]['correct_time']),
            spikes=cp.deepcopy(trace_set[1]['LA']['Spikes'][index_map[1, i]-1, :]),
            spike_time=cp.deepcopy(trace_set[1]['LA']['ms_time_behav']),
            maze_type=trace_set[1]['maze_type'],
            title=f"{round(trace_set[1]['LA']['SI_all'][index_map[1, i]-1], 2)}, {len(trace_set[1]['LA']['place_field_all'][index_map[1, i]-1])}",
            title_color=colors[trace_set[1]['LA']['is_placecell'][index_map[1, i]-1]],
        )
        
        ax3, a3, b3 = TraceMapAxes(
            ax=ax3,
            trajectory=cp.deepcopy(trace_set[2]['correct_pos']),
            behav_time=cp.deepcopy(trace_set[2]['correct_time']),
            spikes=cp.deepcopy(trace_set[2]['LA']['Spikes'][index_map[2, i]-1, :]),
            spike_time=cp.deepcopy(trace_set[2]['LA']['ms_time_behav']),
            maze_type=trace_set[2]['maze_type'],
            title=f"{round(trace_set[2]['LA']['SI_all'][index_map[2, i]-1], 2)}, {len(trace_set[2]['LA']['place_field_all'][index_map[2, i]-1])}",
            title_color=colors[trace_set[2]['LA']['is_placecell'][index_map[2, i]-1]],
        )
        
        ax4 = _add_footprint(
            ax=ax4,
            sfp=sfps[0],
            cell_id=index_map[0, i]-1
        )
        
        ax5 = _add_footprint(
            ax=ax5,
            sfp=sfps[1],
            cell_id=index_map[1, i]-1
        )
        
        ax6 = _add_footprint(
            ax=ax6,
            sfp=sfps[2],
            cell_id=index_map[2, i]-1
        )
        ax1.axis([-0.6, 47.6, 47.6, -0.6])
        ax2.axis([-0.6, 47.6, 47.6, -0.6])
        ax3.axis([-0.6, 47.6, 47.6, -0.6])
        plt.savefig(join(loc, f"{index_map[0,i]}-{index_map[1,i]}-{index_map[2,i]}.png"), dpi = 600)
        plt.savefig(join(loc, f"{index_map[0,i]}-{index_map[1,i]}-{index_map[2,i]}.svg"), dpi = 600)
        
        ax1.clear(), ax2.clear(), ax3.clear()
        ax4.clear(), ax5.clear(), ax6.clear()


def field_number(
    index_map: np.ndarray,
    trace_set: list,
    save_loc: str,
    **imshow_kw
):
    n_neuron = index_map.shape[1]
    
    field_num = []
    days = []
    
    for i in range(n_neuron):
        n = [len(trace_set[k]['LA']['place_field_all'][index_map[k, i]-1]) for k in range(3)]
        d = [f'Day {k}' for k in range(1,4)]
        
        field_num = field_num + n
        days = days + d
    
    Data = {'Field Number': field_num, 'Training Day': days}
    
    fig = plt.figure(figsize=(2,3))
    ax = Clear_Axes(plt.axes(), close_spines=['right', 'top'], ifxticks=True, ifyticks=True)
    
    sns.barplot(
        data=Data,
        x='Training Day',
        y='Field Number',
        ax=ax,
        palette='Blues',
        capsize=0.3,
        linewidth=0.8,
        width=0.8
    )
    plt.show()

index_map, trace_set, sfps = preprocess(
    mouse = 10212,
    stage = "Stage 2",
    session = 3,
    occu_num = 3
)

save_loc = join(loc, f"{10212}_{'Stage 2'}")
mkdir(save_loc)
field_number(index_map=index_map, trace_set=trace_set, save_loc=save_loc)
    
#crossday_tracemap(index_map=index_map, trace_set=trace_set, sfps=sfps, save_loc=save_loc)
#crossday_ratemap(index_map=index_map, trace_set=trace_set, sfps=sfps, save_loc=save_loc)

index_map, trace_set, sfps = preprocess(
    mouse = 10209,
    stage = "Stage 2",
    session = 3,
    occu_num = 3
)

save_loc = join(loc, f"{10209}_{'Stage 2'}")
mkdir(save_loc)
field_number(index_map=index_map, trace_set=trace_set, save_loc=save_loc)
    
#crossday_tracemap(index_map=index_map, trace_set=trace_set, sfps=sfps, save_loc=save_loc)
#crossday_ratemap(index_map=index_map, trace_set=trace_set, sfps=sfps, save_loc=save_loc)

index_map, trace_set, sfps = preprocess(
    mouse = 10212,
    stage = "Stage 1",
    session = 2,
    occu_num = 3
)

save_loc = join(loc, f"{10212}_{'Stage 1'}")
mkdir(save_loc)
field_number(index_map=index_map, trace_set=trace_set, save_loc=save_loc)
    
#crossday_tracemap(index_map=index_map, trace_set=trace_set, sfps=sfps, save_loc=save_loc)
#crossday_ratemap(index_map=index_map, trace_set=trace_set, sfps=sfps, save_loc=save_loc)

index_map, trace_set, sfps = preprocess(
    mouse = 10209,
    stage = "Stage 1",
    session = 2,
    occu_num = 3
)

save_loc = join(loc, f"{10209}_{'Stage 1'}")
mkdir(save_loc)
field_number(index_map=index_map, trace_set=trace_set, save_loc=save_loc)
    
#crossday_tracemap(index_map=index_map, trace_set=trace_set, sfps=sfps, save_loc=save_loc)
#crossday_ratemap(index_map=index_map, trace_set=trace_set, sfps=sfps, save_loc=save_loc)
"""
for i in tqdm(range(len(f1))):
    if f1['maze_type'][i] in [1, 2] and f1['training_day'][i] in ['Day 1', 'Day 2', 'Day 3']:
        with open(f1['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
            
        if 'LA' not in trace.keys():
            print(f"{i}, {f1['MiceID'][i]}, {f1['maze_type'][i]}, {f1['training_day'][i]}, {f1['date'][i]}")

"""