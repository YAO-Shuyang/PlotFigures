from mylib.statistic_test import *

code_id = "0314 - Schematics for long-term evolution"
loc = join(figpath, code_id)
mkdir(loc)

old_map = np.zeros(144)
old_map[67-1] = 10
old_map[55-1] = 7
old_map[54-1] = 3
old_map[66-1] = 1
old_map[79-1] = 7
old_map[78-1] = 3
old_map[90-1] = 1

fig = plt.figure(figsize=(2,2))
ax = Clear_Axes(plt.axes())
MTOP = gaussian_smooth_matrix1d(1000, window = 20, sigma=3, folder=0.1)

ax, _ = LinearizedRateMapAxes(
    ax=ax,
    content=old_map,
    maze_type=1,
    M=MTOP,
    linewidth=0.5
)
ax.set_xlim(45, 72)
plt.savefig(os.path.join(loc, 'Field Illustrator.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Field Illustrator.svg'), dpi=600)
plt.close()


old_map = np.zeros(144)
fields = {
    0: np.array([27, 30,     90, 112     ]),
    1: np.array([27,     60, 90, 112,    ]),
    2: np.array([27,     60,     112,    ]),
    3: np.array([27, 30, 60,     112, 141]),
    4: np.array([    30, 60,     112, 141, 73]),
    5: np.array([        60,     112, 141, 73]),
    6: np.array([        60,     112, 141, 73]),
    7: np.array([        60,     112, 141]),
    8: np.array([        60,     112]),
    9: np.array([        60,     112, 141])
}
fig, axes = plt.subplots(ncols=1, nrows=10, figsize=(4, 6))
for i in range(10):
    ax = Clear_Axes(axes[i])
    old_map = np.zeros(144)
    old_map[fields[i]-1] = np.random.rand(len(fields[i]))*0.5+3
    ax, _ = LinearizedRateMapAxes(
        ax=ax,
        content=old_map,
        maze_type=1,
        M=MTOP,
        linewidth=0.5
    )
    ax.set_yticks([])
plt.savefig(os.path.join(loc, 'Multifield Evolution.png'), dpi=600)
plt.savefig(os.path.join(loc, 'Multifield Evolution.svg'), dpi=600)
plt.close()