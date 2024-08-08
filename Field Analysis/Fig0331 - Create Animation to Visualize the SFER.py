from mylib.statistic_test import *
import matplotlib.patches as patches

file_name = r"E:\Data\FigData\PermenantFieldAnalysis\mouse_1000_converged_10000fields_50days.pkl"
code_id = '0331 - Create Animation to Visualize the SFER'
loc = os.path.join(figpath, code_id)
mkdir(loc)

fig_temp = join(loc, 'temp')
mkdir(fig_temp)

with open(file_name, 'rb') as handle:
    Data = pickle.load(handle)

print(Data['field_reg'].shape)

def plot_figure(field_reg: np.ndarray):
    fig = plt.figure(figsize=(10, 4))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    
    for i in tqdm(range(field_reg.shape[1])):
        for j in range(field_reg.shape[0]):
            idx = np.where(field_reg[j, :] == 1)[0]
            
            if idx.shape[0] == 0:
                continue
            
            for k in range(idx[0], i+1):
                
                color = '#CFDFEE' if field_reg[j, k] == 1 else '#F7DC86'
                round_square = patches.FancyBboxPatch(
                    (k+1.2, j+0.2),  # 左下角坐标
                    0.6, 0.6,    # 宽和高
                    boxstyle=patches.BoxStyle("Round", pad=0.1, rounding_size=0.1),  # 圆角风格
                    linewidth=0,
                    edgecolor=color,
                    facecolor=color
                )
                #ax.fill_betweenx(y=[j-0.4, j+0.4], x1=k-0.4, x2 = k+0.4, edgecolor=None, linewidth=0, color = color, zorder = 0)
                ax.add_patch(round_square)
        
        ax.axis([0, field_reg.shape[1]+1, -1, field_reg.shape[0]])
        ax.set_ylabel("Place Fields")
        ax.set_xlabel("Session No.")
        ax.set_aspect("equal")
        ax.set_xticks(np.append([1], np.linspace(5, 100, 20)))
        plt.savefig(join(fig_temp, f'{i}.png'), dpi=600)
        ax.clear()


#plot_figure(Data['field_reg'])     

from moviepy.editor import ImageSequenceClip
image_files = [join(fig_temp, f'{i}.png') for i in range(100)]

clip = ImageSequenceClip(image_files, fps=10)
clip.write_videofile(join(loc, f'convergent_fields.mp4'), codec='libx264')