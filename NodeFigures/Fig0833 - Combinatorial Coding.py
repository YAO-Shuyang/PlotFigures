from mylib.statistic_test import *

code_id = '0833 - Combinatorial Coding'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ["Segment", "Route Num", "Proportion"],
                              f=f2, 
                              function = CombinatorialCoding_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

print_estimator(Data['Proportion'][np.where(((Data['Route Num'] - Data['Segment'] == 0) | (Data['Route Num'] == 5)))[0]])

for seg in  range(2, 8):
    print(seg)
    print_estimator(Data['Proportion'][np.where((Data['Segment'] == seg) & ((Data['Route Num'] - Data['Segment'] == 0) | (Data['Route Num'] == 5)))[0]])

colors = sns.color_palette("rainbow", 7)[::-1]
fig = plt.figure(figsize=(4, 2.5))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
        x="Route Num",
        y="Proportion",
        data=Data,
        hue="Segment",
        palette=colors[1:],
        ax=ax,
        linewidth=0.5,
        err_style="bars",
        markers='o',
        markersize=4,
        err_kws={"linewidth": 0.5, "capthick": 0.3},
        zorder=2
)
ax.set_ylim(0, 1.03)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(os.path.join(loc, f"All.svg"), dpi=600)
plt.savefig(os.path.join(loc, f"All.png"), dpi=600)
plt.close()

# Illustration
fig = plt.figure(figsize=(3, 3))
ax = Clear_Axes(plt.axes())
DrawMazeProfile(maze_type=1, axes=ax, nx=12, linewidth=0.5, color = 'black')


segs = [seg1, seg2, seg3, seg4, seg5, seg6, seg7]
for i in range(7):
    for n in range(len(segs[i])):
        x, y = (segs[i][n]-1) % 12, (segs[i][n]-1) // 12
        ax.fill_between([x-0.5, x+0.5], [y-0.5, y-0.5], [y+0.5, y+0.5], color=colors[i], edgecolor=None, zorder=1, alpha=0.8)

ax.axis([-0.7, 11.7, 11.7, -0.7])
plt.savefig(os.path.join(loc, 'Illustration.svg'), dpi=600)
plt.savefig(os.path.join(loc, 'Illustration.png'), dpi=600)
plt.close()
