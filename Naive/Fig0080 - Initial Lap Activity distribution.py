from mylib.statistic_test import *
from matplotlib.animation import FuncAnimation
code_id = '0080 - Initial Lap Activity distribution'
loc = os.path.join(figpath, code_id)
mkdir(loc)

def update(frame, t, dF, x, y, ax1, ax2, thre):
    if frame <= 1:
        return
    
    idxlef = max(0, frame-30)
    idxrig = max(frame, 30)
    ax2.plot(t[frame-2:frame]/1000, dF[frame-2:frame], color='black')
    ax2.set_xlim([t[idxlef]/1000, t[idxrig]/1000])
    ax2.set_ylim([0, 4])

    ax1.plot(x[frame-2:frame], y[frame-2:frame], color='gray')
    ax1.set_aspect('equal')
    
    if dF[frame] > thre:
        ax1.plot([x[frame]], [y[frame]], 'o', color='red', markeredgewidth = 0, markersize = 4)
        
def animation(t, dF, x, y, maze_type=1):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,3), gridspec_kw={"width_ratios": [1, 3]})
    ax1, ax2 = Clear_Axes(axes[0]), Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax1.set_aspect('equal')
    ax2.set_xlabel("Time / s")
    ax2.set_ylabel("dF/F")
    DrawMazeProfile(maze_type=1, axes=ax1, nx=48, color='black', linewidth=1)
    ax1.axis([-0.7, 47.7, 47.7, -0.7])
    thre = np.std(dF)*3 + np.mean(dF)
    anim = FuncAnimation(fig, update, frames=len(t), fargs=(t, dF, x, y, ax1, ax2, thre), interval=1)

    plt.show()
    


    
if __name__ == '__main__':
    with open(r"E:\Data\Cross_maze\10227\20230806\session 2\trace.pkl", 'rb') as handle:
        trace = pickle.load(handle)

    beg, end = trace['lap beg time'][0], trace['lap end time'][0]
    idx = np.where((trace['ms_time'] > beg) & (trace['ms_time'] < end))[0]
    
    dF = trace['RawTraces'][:, idx]
    t = trace['ms_time'][idx]
    x, y = idx_to_loc(trace['spike_nodes_original'][idx], nx=48, ny=48)
    
    diff = np.where(dF - np.mean(dF, axis=1) - np.std(dF, axis=1)*3 > 0, 1, 0)
    