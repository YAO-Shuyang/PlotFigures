from mylib.statistic_test import *
from mylib.field.sfer import func_curved_surface, get_data, get_surface, fit_curved_surface

code_id = "0335 - Fitted 2d Curved Plane"
loc = join(figpath, code_id)
mkdir(loc)

I, A, P = get_data(1, 'CrossMaze')

P_REAL = get_surface(1, 'CrossMaze')
I_FIT, A_FIT = np.meshgrid(np.arange(P_REAL.shape[0]), np.arange(1, P_REAL.shape[1]+1))

params = fit_curved_surface(I, A, P)
print(params)
P_FIT = func_curved_surface((I_FIT, A_FIT), *params)
P_FIT[np.where(I_FIT+A_FIT >= 26)] = np.nan

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 2, 1, projection='3d')
ax1.plot_surface(I_FIT, A_FIT, P_FIT, edgecolor='none', alpha=0.8, cmap='viridis')
ax1.plot(np.repeat(0, P_FIT.shape[0]), np.arange(1, P_FIT.shape[0] + 1), P_FIT[:, 0], '#D6ABA1', linewidth=1)
ax1.plot(np.arange(0, P_FIT.shape[1]), np.repeat(1, P_FIT.shape[1]), P_FIT[0, :],'#F0E5A7', linewidth=1)
ax1.set_zlim(0, 1)
ax1.view_init(azim=21, elev=11)

ax2 = Clear_Axes(plt.subplot(1, 2, 2))
im = ax2.imshow(P_FIT)
plt.colorbar(im, ax=ax2)
ax2.invert_yaxis()
ax2.set_xlabel('I')
ax2.set_ylabel('A')
ax2.set_aspect('equal')

plt.savefig(join(loc, 'Fitted Curve.png'), dpi=600)
plt.savefig(join(loc, 'Fitted Curve.svg'), dpi=600)
plt.show()
