from mylib.statistic_test import *
from mazepy.datastruc.neuact import NeuralTrajectory, SpikeTrain
from mazepy.datastruc.variables import VariableBin
from mylib.calcium.dsp_ms import classify_lap
from mazepy.basic._time_sync import _coordinate_recording_time

from umap.umap_ import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, MDS
from scipy.optimize import curve_fit

code_id = "0860 - Decoding Retrieval With GNB"
loc = os.path.join(figpath, "Dsp", code_id)
mkdir(loc)

saved_dir = join(loc, "Decoded Results")
dir_name = join(figpath, "Dsp", "Reforming Neural Trajectory Analysis") 
mkdir(saved_dir)
pass

saved_dir_lossmap = join(loc, "Loss Maps")
mkdir(saved_dir_lossmap)

pretest_gnb_res_dir = join(loc, "Pretest GNB")
mkdir(pretest_gnb_res_dir)
if exists(join(saved_dir_lossmap, f" {10212} [Loss Map].pkl")) == False:
    for mouse in [10212, 10224, 10227, 10232]:
        with open(join(dir_name, f"{mouse}.pkl"), 'rb') as f:
            res = pickle.load(f)
        
        nodes_traj = res['raw_nodes_traj']
        father_nodes_traj = S2F[nodes_traj - 1]
        behav_params_templ_traj = res['behav_params_templ_traj']
        behav_to_raw_traj = res['behav_to_raw_traj']
        behav_params_traj = res['behav_params_traj']
        raw_params_traj = behav_params_traj[:, behav_to_raw_traj]
        print(behav_to_raw_traj.shape, res['behav_params_traj'].shape, res['neural_traj'].shape)
        raw_params_templ_traj = behav_params_templ_traj[:, behav_to_raw_traj]

        behav_res_angles = raw_params_traj[4, :] - raw_params_templ_traj[2, :]
        behav_res_angles[behav_res_angles > np.pi] = 2*np.pi - behav_res_angles[behav_res_angles > np.pi]
        behav_res_angles[behav_res_angles < -np.pi] = -2*np.pi - behav_res_angles[behav_res_angles < -np.pi]

        idx = np.where((np.isin(father_nodes_traj, CP_DSPs[1][0])) & (np.abs(behav_res_angles) <= np.pi/4))[0]

        session_traj = res['raw_session_traj'][idx]
        lap_traj = res['raw_lap_traj'][idx]
        route_traj = res['raw_route_traj'][idx]

        time_traj = res['raw_time_traj'][idx]
        smoothed_loss_traj = res['raw_smoothed_loss_traj'][idx]
        speed_traj = res['raw_speed_traj'][idx]
        raw_traj = res['raw_traj'][:, idx]
        nodes_traj = res['raw_nodes_traj'][idx]
        father_nodes_traj = S2F[nodes_traj - 1]

        beg = np.concatenate(([0], np.where(np.diff(lap_traj) != 0)[0] + 1))
        end = np.concatenate((np.where(np.diff(lap_traj) != 0)[0]+1, [len(lap_traj)]))
        print(f"{mouse}:")
        loss_map = []
        routes = []
        for i in tqdm(range(len(beg))):
            rt = route_traj[beg[i]]
            if rt in [0]:
                continue
            
            lmap = np.zeros(144, np.float64) * np.nan
            for j in range(beg[i], end[i]):
                if speed_traj[j] < 2.5:
                    continue
                lmap[father_nodes_traj[j]-1] = smoothed_loss_traj[j]
            loss_map.append(lmap)
            routes.append(rt)
        
        loss_map = np.vstack(loss_map)[:, CP_DSPs[1][0]-1]
        routes = np.array(routes, dtype=np.int64)
        
        with open(join(saved_dir_lossmap, f"{mouse} [Loss Map].pkl"), 'wb') as f:
            pickle.dump([loss_map, routes], f)
            
        with open(join(pretest_gnb_res_dir, f"{mouse}.pkl"), 'rb') as f:
            pretest_res = pickle.load(f)
            
        loss_traj = pretest_res['loss_traj']
        smoothed_loss_traj = pretest_res['smoothed_loss_traj']
        session_traj = pretest_res['session_traj']
        nodes_traj = pretest_res['nodes_traj']
        father_nodes_traj = S2F[nodes_traj - 1]
        lap_traj = pretest_res['lap_traj']
        time_traj = pretest_res['time_traj']
        route_traj = np.repeat(0, len(lap_traj))
        
        beg = np.concatenate(([0], np.where(np.diff(lap_traj) != 0)[0] + 1))
        end = np.concatenate((np.where(np.diff(lap_traj) != 0)[0]+1, [len(lap_traj)]))
        print(f"{mouse} (Pretest):")
        loss_map = []
        routes = []
        for i in tqdm(range(len(beg))):
            rt = route_traj[beg[i]]
            if session_traj[beg[i]] <9:
                continue
            if rt in [0]:
                lmap = np.zeros(144, np.float64) * np.nan
                for j in range(beg[i], end[i]):
                    lmap[father_nodes_traj[j]-1] = smoothed_loss_traj[j]
                loss_map.append(lmap)
                routes.append(rt)
        loss_map = np.vstack(loss_map)[:, CP_DSPs[1][0]-1]
        routes = np.array(routes, dtype=np.int64)
        with open(join(saved_dir_lossmap, f"{mouse} [Loss Map Pretest].pkl"), 'wb') as f:
            pickle.dump([loss_map, routes], f)

from scipy.optimize import minimize_scalar

def gmm_pdf_scalar(x, weights, means, sigmas):
    # x is scalar
    val = 0.0
    for w, m, s in zip(weights, means, sigmas):
        val += w * (1.0 / (np.sqrt(2 * np.pi) * s)) * np.exp(-0.5 * ((x - m) / s)**2)
    return val
    
def find_gmm_saddle(weights, means, sigmas):
    """
    Find the local minimum of the 2-component GMM pdf between the two means.
    """
    # ensure sorted means
    order = np.argsort(means)
    means = means[order]
    sigmas = sigmas[order]
    weights = weights[order]

    # Minimize p(x) between the two component centers
    res = minimize_scalar(
        lambda z: gmm_pdf_scalar(z, weights, means, sigmas),
        bounds=(means[0], means[1]),
        method='bounded'
    )

    x_saddle = res.x
    y_saddle = gmm_pdf_scalar(x_saddle, weights, means, sigmas)
    return x_saddle, y_saddle

def fit_two_component_gmm(data):
    """
    data: 1D array-like of samples
    returns: fitted GaussianMixture instance
    """
    data = np.asarray(data).reshape(-1, 1)  # GMM expects 2D array

    gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',   # fine for 1D
        random_state=0
    )
    gmm.fit(data)
    return gmm

for mouse in [10212, 10224, 10227, 10232]:
    with open(join(saved_dir_lossmap, f"{mouse} [Loss Map].pkl"), 'rb') as f:
        loss_map, routes = pickle.load(f)

    with open(join(saved_dir_lossmap, f"{mouse} [Loss Map Pretest].pkl"), 'rb') as f:
        loss_map_pretest, routes_pretest = pickle.load(f)
    
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8, 6))
    ax0 = Clear_Axes(axes[0])
    im = ax0.imshow(np.log10(loss_map), aspect='auto', cmap=RetrievCmap, vmin=0, vmax=2)
    cbar = plt.colorbar(im, ax=ax0)
    cbar.set_ticks(np.log10(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])), 
                   labels=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
    ax1 = Clear_Axes(axes[1])
    im = ax1.imshow(np.log10(loss_map_pretest), aspect='auto', cmap=RetrievCmap, vmin=0, vmax=2)
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_ticks(np.log10(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])), 
                   labels=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
    plt.suptitle(f"Loss Map [{mouse}]")
    plt.savefig(join(saved_dir, f"Loss Map [{mouse}].png"), dpi=600)
    plt.savefig(join(saved_dir, f"Loss Map [{mouse}].svg"), dpi=600)
    plt.show()
    
    X, Y = np.meshgrid(np.arange(loss_map.shape[1]), np.arange(loss_map.shape[0]))
    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(
        X,
        Y,
        (np.clip(np.log10(loss_map), 0, 2))*20,
        cmap=RetrievCmap
    )
    ax.set_aspect("equal")
    plt.savefig(join(saved_dir, f"Loss Map 3D [{mouse}].png"), dpi=600)
    plt.savefig(join(saved_dir, f"Loss Map 3D [{mouse}].svg"), dpi=600)
    plt.show()
    """
    x = np.tile(np.arange(loss_map.shape[1]), (loss_map.shape[0], 1))
    fig = plt.figure(figsize=(4, 2))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    sns.lineplot(
        x=x.flatten(),
        y=np.log10(loss_map).flatten(),
        
    )
    plt.show()
    """
    diff_map = np.diff(np.log10(loss_map), axis=1)
    x = np.tile(np.arange(loss_map.shape[1]-1), (loss_map.shape[0], 1))
    rts = np.tile(routes[:, np.newaxis], (1, loss_map.shape[1]-1))

    data = np.vstack([np.log10(loss_map)[:, :-1].flatten(), diff_map.flatten()]).T
    x = x.flatten()
    rts = rts.flatten()
    
    idx = np.where(
        (np.isnan(data).sum(axis=1) == 0) &
        (np.isneginf(data).sum(axis=1) == 0)
    )[0]
    data = data[idx, :]
    x = x[idx]
    rts = rts[idx]
    from sklearn.mixture import GaussianMixture
    
    gmm = fit_two_component_gmm(data[:, 0])
    means = gmm.means_.flatten()        # shape (2,)
    covs = gmm.covariances_.flatten()   # shape (2,) for 1D
    weights = gmm.weights_.flatten()    # shape (2,)
    sigmas = np.sqrt(covs)

    # Sort components by mean so we know "left" and "right"
    order = np.argsort(means)
    means = means[order]
    sigmas = sigmas[order]
    weights = weights[order]
    
    x_saddle, y_saddle = find_gmm_saddle(weights, means, sigmas)
    print(f"Saddle point at x = {x_saddle}, p(x) = {y_saddle}")
    thre = 0.25
    
    data_pretest = np.vstack([
        np.log10(loss_map_pretest)[:, :-1].flatten(),
        np.diff(np.log10(loss_map_pretest), axis=1).flatten()
    ])
    x_pretest = np.tile(
        np.arange(loss_map_pretest.shape[1]-1),
        (loss_map_pretest.shape[0], 1)
    ).flatten()
    """
    plt.figure(figsize=(4, 2))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.hist(data[:, 0], bins=250, density=True, alpha=0.5, color='gray', range=(0, 2.5))
    ax.axvline(x_saddle, color='red', lw=0.5, ls='--')
    ax.set_title(10**x_saddle)
    plt.show()
    
    plt.figure(figsize=(4, 4))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.scatter(
        data[:, 0],
        data[:, 1],
        color='gray',
        s=1,
        alpha=0.5
    )
    
    ax.plot([0, 2.5], [x_saddle, x_saddle-2.5], color='k', lw=0.5, ls='--')
    ax.axvline(x_saddle, color='k', lw=0.5, ls='--')
    ax.plot([x_saddle-thre, x_saddle], [thre, thre], color='k', lw=0.5, ls='--')
    ax.plot([x_saddle, x_saddle+thre], [-thre, -thre], color='k', lw=0.5, ls='--')
    plt.show()
    """
    
    retrieval_points = np.where(
        (data[:, 0] > x_saddle) &
        #(data[:, 1] < -thre) &
        (data[:, 1] < x_saddle - data[:, 0])
    )[0]
    
    retrieval_points_pretest = np.where(
        (data_pretest[0, :] > x_saddle) &
        #(data_pretest[1, :] < -thre) &
        (data_pretest[1, :] < x_saddle - data_pretest[0, :])
    )[0]
    
    gkernel = np.exp(-0.5 * ((np.arange(-2, 3) / 1) ** 2))
    gkernel = gkernel / np.sum(gkernel)
    counts = np.zeros((6, 110), dtype=np.float64)
    n_laps = np.zeros(6, dtype=np.int64)
    for rt in range(1,7):
        rt_idx = np.where(rts[retrieval_points] == rt)[0]
        counts[rt-1, :] = np.histogram(x[retrieval_points][rt_idx], bins=110, range=(-0.5, 109.5))[0]
        counts[rt-1, :] = counts[rt-1, :] / np.where(routes == rt)[0].shape[0]
        n_laps[rt-1] = np.where(routes == rt)[0].shape[0]

        counts[rt-1, :] = np.convolve(counts[rt-1, :], gkernel, mode='same')
    
    n_shuf_selected = int(np.mean(n_laps))
    counts_shuf = np.zeros((10000, 110), dtype=np.float64)
    for shuf_i in range(10000):
        rand_idx = np.random.choice(loss_map_pretest.shape[0], n_shuf_selected, replace=False)
        data_pretest = np.vstack([
            np.log10(loss_map_pretest)[rand_idx, :-1].flatten(),
            np.diff(np.log10(loss_map_pretest)[rand_idx, :], axis=1).flatten()
        ])
        retrieval_points_shuf = np.where(
            (data_pretest[0, :] > x_saddle) &
            #(data_pretest[1, :] < -thre) &
            (data_pretest[1, :] < x_saddle - data_pretest[0, :])
        )[0]
        counts_shuf[shuf_i, :] = np.histogram(x_pretest[retrieval_points_shuf], bins=110, range=(-0.5, 109.5))[0]
        counts_shuf[shuf_i, :] = counts_shuf[shuf_i, :] / n_shuf_selected
        counts_shuf[shuf_i, :] = np.convolve(counts_shuf[shuf_i, :], gkernel, mode='same')
        
    counts_shuf_upper = np.percentile(counts_shuf, 95, axis=0)
    p_values = np.zeros((6, 110), dtype=np.float64)
    for rt in range(6):
        for pos in range(110):
            p_values[rt, pos] = np.sum(counts_shuf[:, pos] >= counts[rt, pos]) / 10000.0
    
    rts_all = np.tile(np.arange(1, 7)[:, np.newaxis], (1, 110))
    
    res = {
        "Routes": rts_all.flatten(),
        "Positions": np.tile(np.arange(110)[np.newaxis, :], (6, 1)).flatten(),
        "Counts": counts.flatten()
    }
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(4, 4))
    ax = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    sns.lineplot(
        x="Positions",
        y="Counts",
        hue="Routes",
        palette=DSPPalette[1:],
        data=res,
        ax=ax
    )
    ax.fill_between(
        x=np.arange(110),
        y1=counts_shuf_upper,
        y2=0,
        color='gray',
        alpha=0.3,
        edgecolor=None
    )
    ax.semilogy()
    ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    p_values[p_values < 1e-3] = 0
    p_values[p_values >= 1e-3] = 1
    for rt in range(6):
        ax2.scatter(
            np.arange(110),
            p_values[rt, :]+ 0.02 * rt,
            color=DSPPalette[rt+1],
            lw=0,
            s=5, 
            label=f"R{rt+1}"
        )
    plt.show()
    