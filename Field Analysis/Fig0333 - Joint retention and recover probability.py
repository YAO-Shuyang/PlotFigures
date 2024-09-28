from mylib.statistic_test import *
from mylib.field.tracker_v2 import Tracker2d
from mylib.field.sfer import get_surface, get_data, fit_kww, fit_reci

code_id = "0333 - Joint retention and recover probability"
loc = join(figpath, code_id)
mkdir(loc)

def plot(data_info: tuple[int, str, str], file_name: str):
    P = get_surface(*data_info)
    IS, AS, PS = get_data(*data_info)

    # plot 3d surface
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(2, 2, 1,projection='3d')
    ax2 = Clear_Axes(fig.add_subplot(2, 2, 3))
    ax3 = Clear_Axes(fig.add_subplot(2, 2, 2), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax4 = Clear_Axes(fig.add_subplot(2, 2, 4), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

    I, A = np.meshgrid(np.arange(P.shape[1]), np.arange(1, P.shape[0]+1))
    ax1.plot_surface(I, A, P, edgecolor='none', alpha=0.8, cmap='viridis')
    ax1.plot(np.repeat(0, P.shape[0]), np.arange(1, P.shape[0] + 1), P[:, 0], '#D6ABA1', linewidth=1)
    ax1.plot(np.arange(0, P.shape[1]), np.repeat(1, P.shape[1]), P[0, :],'#F0E5A7', linewidth=1)
    ax1.set_xlabel('I')
    ax1.set_ylabel('A')
    ax1.view_init(azim=-139, elev=15)
    ax1.set_zlim(0, 1)

    """
    ax2.plot_surface(I, A, P, edgecolor='none', alpha=0.8, cmap='viridis')
    ax2.plot(np.repeat(0, P.shape[0]), np.arange(1, P.shape[0] + 1), P[:, 0], '#D6ABA1', linewidth=1)
    ax2.plot(np.arange(0, P.shape[1]), np.repeat(1, P.shape[1]), P[0, :],'#F0E5A7', linewidth=1)
    ax2.set_xlabel('I')
    ax2.set_ylabel('A')
    ax2.view_init(azim=-139, elev=15)
    ax2.set_zlim(0, 1)
    ax2.view_init(azim=21, elev=11)
    """
    im = ax2.imshow(P)
    plt.colorbar(im, ax=ax2)
    ax2.invert_yaxis()
    ax2.set_xlabel('I')
    ax2.set_ylabel('A')
    ax2.set_aspect('equal')
    
    colors = sns.color_palette('rainbow', len(P) - 2)
    for act in range(len(P) - 2):
        fit_kww(IS, AS, PS, act=act, ax=ax3, color=colors[act], linewidth=0.5)
    idx = np.where(AS <= len(P) - 3)[0]
    sns.stripplot(x=IS[idx], y=PS[idx], hue=AS[idx], jitter=0.2, edgecolor='k', linewidth=0.15, size=3, alpha=0.8, ax=ax3, palette='rainbow')
    ax3.set_xlabel('Action')
    ax3.set_ylim(0, 1.03)
    ax3.set_yticks(np.linspace(0, 1, 6))
    ax3.legend()
    
    for inact in range(len(P) - 2):
        fit_reci(IS, AS, PS, inact=inact, ax=ax4, color=colors[inact], linewidth=0.5)
    idx = np.where(IS <= len(P) - 3)[0]
    sns.stripplot(x=AS[idx], y=PS[idx], hue=IS[idx], jitter=0.2, edgecolor='k', linewidth=0.15, size=3, alpha=0.8, ax=ax4, palette='rainbow')
    ax4.legend()
    ax4.set_ylim(0, 1.03)
    ax4.set_yticks(np.linspace(0, 1, 6))
    plt.savefig(join(loc, f'{file_name}.svg'))
    plt.savefig(join(loc, f'{file_name}.png'), dpi = 1200)
    plt.show()

plot((1, 'CrossMaze'), 'MA')
plot((2, 'CrossMaze'), 'MB')
plot((1, 'ReverseMaze', 'cis'), 'MAf')
plot((1, 'ReverseMaze', 'trs'), 'MAb')
plot((3, 'HairpinMaze', 'cis'), 'HPf')
plot((3, 'HairpinMaze', 'trs'), 'HPb')