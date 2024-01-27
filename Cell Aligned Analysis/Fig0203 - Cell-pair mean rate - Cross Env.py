from mylib.cross_env_utils import *

code_id = '0203 - Cross Env mean Rate'
data_loc = join(figdata, code_id)
mkdir(data_loc)
p = join(figpath, 'Cell Aligned', code_id)
mkdir(p)

def get_mean_rate(i: int, f: pd.DataFrame, env1: str = 'op', env2: str = 'm1', **kwargs):
    trace1, trace2 = get_trace(i, f, env1, env2, **kwargs)
    if trace1 is None or trace2 is None:
        return np.array([[],[]], dtype=np.float64)

    index_map = get_placecell_pair(i, f, env1, env2, **kwargs)
    mean_rate1 = np.nansum(trace1['Spikes'], axis = 1)/trace1['t_total']
    mean_rate2 = np.nansum(trace2['Spikes'], axis = 1)/trace2['t_total']

    MR = np.zeros_like(index_map, dtype=np.float64)
    MR[0, :] = mean_rate1[index_map[0, :]-1]
    MR[1, :] = mean_rate2[index_map[1, :]-1]
    return MR

mice = [11095, 11092, 12009, 12012]
fs = [f_CellReg_opm1, f_CellReg_opm2, f_CellReg_m1m2, f_CellReg_opop]
env1s = ['op', 'op', 'm1', 'op1']
env2s = ['m1', 'm2', 'm2', 'op2']
row = 4
col = 4
fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(col*3, row*3))
for r in range(row):
    for l in tqdm(range(col)):
        ax = Clear_Axes(axes[r, l], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        ax.set_aspect('equal')

        if exists(join(data_loc, str(mice[r])+'-'+env1s[l]+'-'+env2s[l]+'.pkl')) == False:
            idx = np.where(fs[l]['MiceID'] == mice[r])[0]
            if len(idx) == 0:
                ax = Clear_Axes(ax)
                continue
            else:
                MR = np.array([[], []], dtype = np.float64)
                for i in idx:
                    MR = np.concatenate([MR, get_mean_rate(i, fs[l], env1=env1s[l], env2=env2s[l])], axis = 1)
            with open(join(data_loc, str(mice[r])+'-'+env1s[l]+'-'+env2s[l]+'.pkl'), 'wb') as df:
                pickle.dump(MR, df)
        else:
            with open(join(data_loc, str(mice[r])+'-'+env1s[l]+'-'+env2s[l]+'.pkl'), 'rb') as handle:
                MR = pickle.load(handle)            

        xy_max = int(np.nanmax(MR))+1
        ax.plot(MR[0, :], MR[1, :], 'ok', markeredgewidth=0, markersize = 2)
        ax.set_xticks(ColorBarsTicks(xy_max, is_auto=True, tick_number=4))
        ax.set_yticks(ColorBarsTicks(xy_max, is_auto=True, tick_number=4))
        ax.set_xlabel(env1s[l])
        ax.set_ylabel(env2s[l])
        ax.set_title(str(mice[r]))

        res = linregress(MR[0, :], MR[1, :])
        k, b = res[0], res[1]
        ax.plot([0, xy_max], [b, k*xy_max+b], color = 'orange')
        ax.axis([0, xy_max, 0, xy_max])
        print(res)

plt.tight_layout()
plt.savefig(join(p,'mean rate.svg'), dpi = 1200)
plt.savefig(join(p,'mean rate.png'), dpi = 1200)
plt.close()


    