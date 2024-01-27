from mylib.statistic_test import *
from mazepy.behav.graph import Graph

code_id = "0054 - Theoretical Shortest Trajectory in Maze"
loc = os.path.join(figpath, code_id)
mkdir(loc)

def add_color_path(
    ax: Axes,
    bins: list,
    d = .5,
    line_kw: dict = {'color':'black', 'linewidth':.4, 'ls':':'},
    line_band: bool = False,
    is_plot_line: bool = True,
    is_plot_decision_points: bool = False,
    color: str = 'black',
    band_width: float=0.1,
    band_color: str = 'black',
    band_kw: dict = {},
    **kwargs
):  
    xp, yp = None, None
    for i, b in enumerate(bins):
        x, y = (b-1)%12, (b-1)//12
        
        if is_plot_decision_points:
            ax.fill_betweenx([y+d, y-d], x-d, x+d, color=color, **kwargs)
            continue
            
        if xp is not None and d == 0.5 and is_plot_line:
            ax.plot([xp, x], [yp, y], **line_kw)

        if xp is not None and line_band:
            if type(band_color) is str or type(band_color) is tuple:
                band_colors = [band_color for i in range(len(bins)-1)]
            else:
                band_colors = band_color
            
            if yp == y:
                ax.fill_betweenx([y-band_width, y+band_width], min(x, xp)-band_width, max(x, xp)+band_width, color = band_colors[i-1], **band_kw)
            elif xp == x:
                ax.fill_between([x-band_width, x+band_width], min(y, yp)-band_width, max(y, yp)+band_width, color = band_colors[i-1], **band_kw)
        
        xp, yp = x, y
    
    return ax
"""
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
ax1, ax2 = Clear_Axes(axes[0]), Clear_Axes(axes[1])

G1 = Graph(12, 12, cp.deepcopy(maze_graphs[(1, 12)]))
G1.plot_shortest_path((0.125, 0.5), (11.875, 11.5), ax=ax1, color='red', linewidth=3, is_show_nodes=False)
print(G1.shortest_distance((0.125, 0.5), (11.875, 11.5))*8)

G2 = Graph(12, 12, cp.deepcopy(maze_graphs[(2, 12)]))
G2.plot_shortest_path((0.125, 0.5), (11.875, 11.5), ax=ax2, color='red', linewidth=3, is_show_nodes=False)
print(G2.shortest_distance((0.125, 0.5), (11.875, 11.5))*8)

DrawMazeProfile(maze_type=1, axes=ax1, nx=12, color='black')
DrawMazeProfile(maze_type=2, axes=ax2, nx=12, color='black')

ax1.axis([-0.6, 11.6, 11.6, -0.6])
ax2.axis([-0.6, 11.6, 11.6, -0.6])
ax1.set_aspect('equal')
ax2.set_aspect('equal')
plt.savefig(join(loc, "Shortest Path.png"), dpi=600)
plt.savefig(join(loc, "Shortest Path.svg"), dpi=600)
plt.close()
"""
"""
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
ax1, ax2 = Clear_Axes(axes[0]), Clear_Axes(axes[1])
DrawMazeProfile(maze_type=1, axes=ax1, nx=12, color='black', linewidth=.6)
DrawMazeProfile(maze_type=2, axes=ax2, nx=12, color='black', linewidth=.6)

CP1 = correct_paths[1]
CP2 = correct_paths[2]
IP1 = incorrect_paths[1]
IP2 = incorrect_paths[2]

DP1 = DPs[1]
DP2 = DPs[2]

add_color_path(ax1, CP1, edgecolor=None, line_band=True, band_color = sns.color_palette('crest', len(CP1)-1))
add_color_path(ax2, CP2, edgecolor=None, line_band=True, band_color = sns.color_palette('flare', len(CP1)-1))

add_color_path(ax1, DP1, d=.1, edgecolor=None, is_plot_decision_points=True)
add_color_path(ax2, DP2, d=.1, edgecolor=None, is_plot_decision_points=True)
ax1.axis([-0.6, 11.6, 11.6, -0.6])
ax2.axis([-0.6, 11.6, 11.6, -0.6])
ax1.set_aspect('equal')
ax2.set_aspect('equal')
plt.savefig(join(loc, "maze illustrator.png"), dpi=600)
plt.savefig(join(loc, "maze illustrator.svg"), dpi=600)
plt.close()

fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(8,2))
ax1, ax2 = Clear_Axes(axes[0]), Clear_Axes(axes[1])

crest = sns.color_palette('crest', len(CP1))
flare = sns.color_palette('flare', len(CP2))


for i in range(len(CP1)):
    ax1.fill_betweenx([0,5], i, i+1, color=crest[i])
    
for i in range(len(CP1)):
    if CP1[i] in DP1:
        ax1.fill_betweenx([1.5,3.5], i+0.4, i+0.6, color='black', edgecolor=None)

ax1.plot([0.5, len(CP1)], [2.5, 2.5], color='black', ls=':', linewidth=0.4)
    
for i in range(len(CP2)):
    ax2.fill_betweenx([0,5], i, i+1, color=flare[i])
    
for i in range(len(CP2)):
    if CP2[i] in DP2:
        ax2.fill_betweenx([1.5,3.5], i+0.4, i+0.6, color='black', edgecolor=None)
        
ax2.plot([0.5, len(CP2)], [2.5, 2.5], color='black', ls=':', linewidth=0.4)

ax1.set_xlim([0, len(CP1)])
ax2.set_xlim([0, len(CP2)])
plt.savefig(join(loc, "maze illustrator2.png"), dpi=600)
plt.savefig(join(loc, "maze illustrator2.svg"), dpi=600)
plt.close()

a = sns.color_palette("rocket", 4)
l = []
for item in a:
    r,g,b = item
    l.append((int(r*255), int(g*255), int(b*255)))

print(l)

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
ax1, ax2 = Clear_Axes(axes[0]), Clear_Axes(axes[1])
DrawMazeProfile(maze_type=3, axes=ax1, nx=12, color='black', linewidth=.6)
DrawMazeProfile(maze_type=3, axes=ax2, nx=12, color='black', linewidth=.6)

CP3 = correct_paths[3]
print(CP3)

blue = sns.color_palette('Blues', 9)[1]
yellow = sns.color_palette('YlOrRd', 9)[1]

add_color_path(ax1, CP3, edgecolor=None, line_band=True, band_color=blue)
add_color_path(ax2, CP3, edgecolor=None, line_band=True, band_color=yellow)

ax1.axis([-0.6, 11.6, 11.6, -0.6])
ax2.axis([-0.6, 11.6, 11.6, -0.6])
ax1.set_aspect('equal')
ax2.set_aspect('equal')
plt.savefig(join(loc, "Hairpin maze illustrator.png"), dpi=600)
plt.savefig(join(loc, "Hairpin maze illustrator.svg"), dpi=600)
plt.close()


fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
ax1, ax2 = Clear_Axes(axes[0]), Clear_Axes(axes[1])
DrawMazeProfile(maze_type=1, axes=ax1, nx=12, color='black', linewidth=.6)
DrawMazeProfile(maze_type=1, axes=ax2, nx=12, color='black', linewidth=.6)

CP1 = correct_paths[1]
IP1 = incorrect_paths[1]

DP1 = DPs[1]

blue = sns.color_palette('Blues', 9)[1]
yellow = sns.color_palette('YlOrRd', 9)[1]

add_color_path(ax1, CP1, edgecolor=None, line_band=True, band_color = blue)
add_color_path(ax2, CP1, edgecolor=None, line_band=True, band_color = yellow)

add_color_path(ax1, DP1, d=.1, edgecolor=None)
add_color_path(ax2, DP1, d=.1, edgecolor=None)
ax1.axis([-0.6, 11.6, 11.6, -0.6])
ax2.axis([-0.6, 11.6, 11.6, -0.6])
ax1.set_aspect('equal')
ax2.set_aspect('equal')
plt.savefig(join(loc, "reverse maze paradigm.png"), dpi=600)
plt.savefig(join(loc, "reverse maze paradigm.svg"), dpi=600)
plt.close()


DSPCorrectTrackPalette = [sns.color_palette('Blues', 9)[1], sns.color_palette('YlOrRd', 9)[1], sns.color_palette("crest", 9)[0], sns.color_palette("flare", 9)[0]]
DSPIncorrectTrackPalette = [sns.color_palette('Blues', 9)[1], sns.color_palette('YlOrRd', 9)[3], sns.color_palette("crest", 9)[2], sns.color_palette("flare", 9)[2]]

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,8))
ax1, ax2, ax3, ax4 = Clear_Axes(axes[0, 0]), Clear_Axes(axes[0, 1]), Clear_Axes(axes[1, 0]), Clear_Axes(axes[1, 1])
DrawMazeProfile(maze_type=1, axes=ax1, nx=12, color='black', linewidth=.6)
DrawMazeProfile(maze_type=1, axes=ax2, nx=12, color='black', linewidth=.6)
DrawMazeProfile(maze_type=1, axes=ax3, nx=12, color='black', linewidth=.6)
DrawMazeProfile(maze_type=1, axes=ax4, nx=12, color='black', linewidth=.6)

ax1.axis([-0.6, 11.6, 11.6, -0.6])
ax2.axis([-0.6, 11.6, 11.6, -0.6])
ax3.axis([-0.6, 11.6, 11.6, -0.6])
ax4.axis([-0.6, 11.6, 11.6, -0.6])

ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')
ax4.set_aspect('equal')

correct_routes = {
    0: np.array([1,13,14,26,27,15,3,4,5,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
    1: np.array([23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
    2: np.array([66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
    3: np.array([99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
    4: np.array([1,13,14,26,27,15,3,4,5,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
}
incorrect_routes = {
    0: np.array([1,13,14,26,27,15,3,4,5,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
    1: np.array([8,7,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
    2: np.array([93,105,106,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
    3: np.array([135,134,133,121,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
    4: np.array([1,13,14,26,27,15,3,4,5,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
}

print(np.where(incorrect_routes[0] == 94)[0], np.where(incorrect_routes[0] == 109)[0])
crest = sns.color_palette('crest', len(correct_routes[0]))

add_color_path(ax1, correct_routes[0], edgecolor = None, line_band=True, band_color = DSPCorrectTrackPalette[0])
add_color_path(ax2, correct_routes[0][:23], edgecolor = None, line_band=True, is_plot_line=False, band_color = sns.color_palette("Greys", 9)[1])
add_color_path(ax2, correct_routes[0][23:], edgecolor = None, line_band=True, band_color = DSPCorrectTrackPalette[1])
add_color_path(ax4, correct_routes[0][:60], edgecolor = None, line_band=True, is_plot_line=False, band_color = sns.color_palette("Greys", 9)[1])
add_color_path(ax4, correct_routes[0][60:], edgecolor = None, line_band=True, band_color = DSPCorrectTrackPalette[2])
add_color_path(ax3, correct_routes[0][:84], edgecolor = None, line_band=True, is_plot_line=False, band_color = sns.color_palette("Greys", 9)[1])
add_color_path(ax3, correct_routes[0][84:], edgecolor = None, line_band=True, band_color = DSPCorrectTrackPalette[3])
plt.savefig(os.path.join(loc,'dsp paradigm [correct].png'), dpi=600)
plt.savefig(os.path.join(loc,'dsp paradigm [correct].svg'), dpi=600)
plt.close()


# Plot dsp paradigm for incorrect path
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,8))
ax1, ax2, ax3, ax4 = Clear_Axes(axes[0, 0]), Clear_Axes(axes[0, 1]), Clear_Axes(axes[1, 0]), Clear_Axes(axes[1, 1])
DrawMazeProfile(maze_type=1, axes=ax1, nx=12, color='black', linewidth=.6)
DrawMazeProfile(maze_type=1, axes=ax2, nx=12, color='black', linewidth=.6)
DrawMazeProfile(maze_type=1, axes=ax3, nx=12, color='black', linewidth=.6)
DrawMazeProfile(maze_type=1, axes=ax4, nx=12, color='black', linewidth=.6)

ax1.axis([-0.6, 11.6, 11.6, -0.6])
ax2.axis([-0.6, 11.6, 11.6, -0.6])
ax3.axis([-0.6, 11.6, 11.6, -0.6])
ax4.axis([-0.6, 11.6, 11.6, -0.6])

ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')
ax4.set_aspect('equal')

crest1 = sns.color_palette('crest', len(incorrect_routes[0]))
crest2 = sns.color_palette('crest', len(incorrect_routes[1]))
crest3 = sns.color_palette('crest', len(incorrect_routes[2]))
crest4 = sns.color_palette('crest', len(incorrect_routes[3]))

add_color_path(ax1, correct_routes[0], edgecolor = None, line_band=True, band_color = DSPIncorrectTrackPalette[0])
add_color_path(ax2, correct_routes[0][:9], edgecolor = None, line_band=True, is_plot_line=False, band_color = sns.color_palette("Greys", 9)[1])
add_color_path(ax2, incorrect_routes[1], edgecolor = None, line_band=True, band_color = DSPIncorrectTrackPalette[1])
add_color_path(ax4, correct_routes[0][:46], edgecolor = None, line_band=True, is_plot_line=False, band_color = sns.color_palette("Greys", 9)[1])
add_color_path(ax4, incorrect_routes[2], edgecolor = None, line_band=True, band_color = DSPIncorrectTrackPalette[2])
add_color_path(ax3, correct_routes[0][:77], edgecolor = None, line_band=True, is_plot_line=False, band_color = sns.color_palette("Greys", 9)[1])
add_color_path(ax3, incorrect_routes[3], edgecolor = None, line_band=True, band_color = DSPIncorrectTrackPalette[3])
plt.savefig(os.path.join(loc,'dsp paradigm [incorrect].png'), dpi=600)
plt.savefig(os.path.join(loc,'dsp paradigm [incorrect].svg'), dpi=600)
plt.close()
"""


# Zoom out
fig = plt.figure(figsize=(4,4))
ax1 = Clear_Axes(plt.axes())
DrawMazeProfile(maze_type=1, axes=ax1, nx=12, color='black', linewidth=.6)

CP1 = correct_paths[1]
IP1 = incorrect_paths[1]

DP1 = DPs[1]

add_color_path(ax1, CP1, edgecolor=None, line_band=True, band_color = sns.color_palette('crest', len(CP1)-1))

add_color_path(ax1, DP1, d=.1, edgecolor=None, is_plot_decision_points=True)
ax1.axis([8.4, 11.6, 2.6, -0.6])
ax1.set_aspect('equal')
plt.savefig(join(loc, "Fig 3f Zoom out 1.png"), dpi=600)
plt.savefig(join(loc, "Fig 3f Zoom out 1.svg"), dpi=600)
plt.close()


# Zoom out
fig = plt.figure(figsize=(4,4))
ax1 = Clear_Axes(plt.axes())
DrawMazeProfile(maze_type=2, axes=ax1, nx=12, color='black', linewidth=.6)

CP2 = correct_paths[2]
IP2 = incorrect_paths[2]

DP2 = DPs[2]

add_color_path(ax1, CP2, edgecolor=None, line_band=True, band_color = sns.color_palette('flare', len(CP2)-1))

add_color_path(ax1, DP2, d=.1, edgecolor=None, is_plot_decision_points=True)
ax1.axis([-0.6, 5.6, 11.6, 8.4])
ax1.set_aspect('equal')
plt.savefig(join(loc, "Fig 3f Zoom out 1.png"), dpi=600)
plt.savefig(join(loc, "Fig 3f Zoom out 1.svg"), dpi=600)
plt.close()