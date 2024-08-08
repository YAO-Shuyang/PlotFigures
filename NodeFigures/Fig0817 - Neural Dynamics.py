from mylib.statistic_test import *

code_id = '0817 - Neural Dynamics'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

from mylib.dsp.neural_traj import visualize_neurotraj

with open(r"E:\Data\Dsp_maze\10224\20231012\trace.pkl", 'rb') as handle:
    trace = pickle.load(handle)
    
"""
visualize_neurotraj(
    trace, 
    n_components=20, 
    component_i=0,
    component_j=1,
    palette='default',
    save_dir=loc,
    method="UMAP",
    n_neighbors = 20, # 27: 12; 12: 12,
    min_dist = 0.1
)
"""

# Create color bar
colors = sns.color_palette("rainbow", 144)
a = np.arange(111)
im = plt.imshow(a[np.newaxis, :], cmap="rainbow", vmax = 143, vmin = 0)
plt.colorbar(im)
plt.savefig(os.path.join(loc, 'colorbar.svg'), dpi = 600)
plt.savefig(os.path.join(loc, 'colorbar.png'), dpi = 600)
plt.close()

