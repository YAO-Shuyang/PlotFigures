# Shuffle: if a matrix follows normal distribution

from mylib.betti_curves import betti_curves
from mylib.statistic_test import *

code_id = '0032 - Clique Topology'
p = os.path.join(figpath, code_id, 'pure-shuffle')
mkdir(p)

datap = os.path.join(figdata, code_id, 'pure-shuffle')
mkdir(datap)

def pure_random_shuffle(file_name:str = 'default', n_neuron:int = 50, **kwargs):
    # Test if a random array will perform normal distribution
    a = np.random.randn(n_neuron,2304)
    betti = betti_curves(a, corr_type = 'pearson', intervals = 0.001)
    cor_mat = cp.deepcopy(betti.ResMat)
    cor_mat = cor_mat.flatten()

    fig = plt.figure(figsize=(4,3))
    ax = Clear_Axes(plt.axes(), close_spines = ['top', 'right'], ifxticks=True, ifyticks=True)
    a, _, _ = ax.hist(cor_mat, bins = 50, **kwargs)
    MAX = np.nanmax(a)
    ax.set_yticks(ColorBarsTicks(peak_rate = MAX, is_auto = True))
    ax.set_xlabel("Correlation Value")
    ax.set_ylabel("Count")
    plt.tight_layout()

    plt.savefig(os.path.join(p, file_name+'.png'), dpi = 600)
    plt.savefig(os.path.join(p, file_name+'.svg'), dpi = 600)
    plt.close()

    betti.plot_betti_curve(save_loc = os.path.join(p, file_name))

    with open(os.path.join(datap, file_name+'.pkl'), 'wb') as f:
        pickle.dump(betti,f)

    return betti.edge_density, betti.betti

def plot_shuffle_betti_curve(Data:dict, save_loc:str = None):
    KeyWordErrorCheck(Data, __file__, keys = ['Edge Density', 'Betti Number', 'Circle Dimension'])
    fig = plt.figure(figsize=(4,3))
    ax = Clear_Axes(plt.axes(), close_spines = ['top', 'right'], xticks = np.linspace(0,1,6), ifyticks = True)
    sns.lineplot(x = 'Edge Density', y = 'Betti Number', data = Data, hue = 'Circle Dimension')
    MAX = int(np.nanmax(Data['Betti Number'])*1.2)
    ax.axis([0,1,0,MAX])
    ax.legend(facecolor = 'white', edgecolor = 'white', ncol = 3, loc = 'upper center', fontsize = 8, title_fontsize = 8)
    if save_loc is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(save_loc+'.png', dpi = 600)
        plt.savefig(save_loc+'.svg', dpi = 600)
        plt.close()


if os.path.exists(os.path.join(datap, 'Total.pkl')):
    with open(os.path.join(datap, 'Total.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    # Pure random shuffle, 100 times
    Data = {'Betti Number':np.array([], np.float64),'Circle Dimension':np.array([]), 'Edge Density':np.array([], np.float64)}
    for i in range(20):
        print(i, ' ----------------------------------------------------------')
        edge_density, betti = pure_random_shuffle(str(i+1), color = 'gray', range = (0,1),)
        l = edge_density.shape[0]
        Data['Betti Number'] = np.concatenate([Data['Betti Number'], betti[1,:], betti[2,:], betti[3,:]])
        Data['Circle Dimension'] = np.concatenate([Data['Circle Dimension'], np.repeat('1D', l), np.repeat('2D', l), np.repeat('3D', l)])
        Data['Edge Density'] = np.concatenate([Data['Edge Density'], edge_density, edge_density, edge_density])
        print('Done.', end = '\n\n\n\n')

    with open(os.path.join(datap, 'Total.pkl'), 'wb') as f:
        pickle.dump(Data, f)

plot_shuffle_betti_curve(Data, save_loc = os.path.join(figpath, code_id, 'pure-shuffle'))


def shuffle_true_data(trace:dict, shuffle_time:int = 100, size:int = 50, **kwargs):
    n = trace['n_neuron']
    assert n >= size
    spike_num = np.nansum(trace['Spikes'], axis = 1)
    delete_idx = np.delete(np.arange(n), trace['SilentNeuron'])
    remain_idx = np.where((spike_num[delete_idx] >= 30)&(trace['is_placecell'][delete_idx] == 1))[0]
    idx = delete_idx[remain_idx]
    if len(idx) < size:
        size = len(idx)

    loc = os.path.join(figpath, code_id, 'Cross Maze Data', str(int(trace['MiceID'])) + '-' + str(int(trace['date']))+'-session'+str(int(trace['session'])))
    dataloc = os.path.join(figdata, code_id, 'Cross Maze Data', str(int(trace['MiceID'])) + '-' + str(int(trace['date']))+'-session'+str(int(trace['session'])))
    mkdir(loc)
    mkdir(dataloc)

    Data = {'Betti Number':np.array([], np.float64),'Circle Dimension':np.array([]), 'Edge Density':np.array([], np.float64)}
    for i in range(shuffle_time):
        print(loc,i,'---------------------------------------')
        rand_idx = np.random.choice(a = idx, size = size, replace = False)
        betti = betti_curves(input_mat1 = trace['smooth_map_all'][rand_idx], corr_type = 'pearson')
        cor_mat = cp.deepcopy(betti.ResMat)
        cor_mat = cor_mat.flatten()

        fig = plt.figure(figsize=(4,3))
        ax = Clear_Axes(plt.axes(), close_spines = ['top', 'right'], xticks=np.linspace(-1,1,11), ifyticks=True)
        a, _, _ = ax.hist(cor_mat, bins = 50, **kwargs)
        MAX = np.nanmax(a)
        ax.axis([-1,1,0, MAX])
        ax.set_yticks(ColorBarsTicks(peak_rate = MAX, is_auto = True))
        ax.set_xlabel("Correlation Value")
        ax.set_ylabel("Count")
        plt.tight_layout()

        plt.savefig(os.path.join(loc, str(i+1)+'.png'), dpi = 600)
        plt.savefig(os.path.join(loc, str(i+1)+'.svg'), dpi = 600)
        plt.close()

        l = betti.edge_density.shape[0]
        Data['Betti Number'] = np.concatenate([Data['Betti Number'], betti.betti[1,:], betti.betti[2,:], betti.betti[3,:]])
        Data['Circle Dimension'] = np.concatenate([Data['Circle Dimension'], np.repeat('1D', l), np.repeat('2D', l), np.repeat('3D', l)])
        Data['Edge Density'] = np.concatenate([Data['Edge Density'], betti.edge_density, betti.edge_density, betti.edge_density])

        betti.plot_betti_curve(save_loc = os.path.join(loc, str(i+1)))

        with open(os.path.join(dataloc, str(i+1)+'.pkl'), 'wb') as f:
            pickle.dump(betti, f)

    with open(os.path.join(dataloc, 'Total.pkl'), 'wb') as f:
        pickle.dump(Data, f)

    plot_shuffle_betti_curve(Data, save_loc = os.path.join(loc, 'Final'))

    return Data

# Experimental data
p = os.path.join(figpath, code_id, 'Cross Maze Data')
mkdir(p)
datap = os.path.join(figdata, code_id, 'Cross Maze Data')
mkdir(datap)
if os.path.exists(os.path.join(datap, 'Total.pkl')):
    with open(os.path.join(datap, 'Total.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    for i in range(73):

        print(i, '--------------------------------------------------------------------------------------')
        if f1['MiceID'][i] == 11094:
            continue
        if f1['maze_type'][i] == 1 and f1['date'][i] == 20220817 or f1['date'][i] == 20220814:
            continue
    
        with open(f1['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)

        datasession = shuffle_true_data(trace = trace)
        print(end = '\n\n\n\n')
        
