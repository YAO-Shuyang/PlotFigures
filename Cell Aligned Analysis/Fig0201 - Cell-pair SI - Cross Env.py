from mylib.cross_env_utils import *

code_id = '0201 - Cross Env SI'
data_loc = join(figdata, code_id)
mkdir(data_loc)
p = join(figpath, 'Cell Aligned', code_id)
mkdir(p)

def get_SI(i: int, f: pd.DataFrame, env1: str = 'op', env2: str = 'm1', **kwargs):
    trace1, trace2 = get_trace(i, f, env1, env2, **kwargs)
    if trace1 is None or trace2 is None:
        return np.array([[],[]], dtype=np.float64)

    index_map = get_placecell_pair(i, f, env1, env2, **kwargs)
    SI = np.zeros_like(index_map, dtype=np.float64)
    SI[0, :] = trace1['SI_all'][index_map[0, :]-1]
    SI[1, :] = trace2['SI_all'][index_map[1, :]-1]
    return SI

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
                SI = np.array([[], []], dtype = np.float64)
                for i in idx:
                    SI = np.concatenate([SI, get_SI(i, fs[l], env1=env1s[l], env2=env2s[l])], axis = 1)
            with open(join(data_loc, str(mice[r])+'-'+env1s[l]+'-'+env2s[l]+'.pkl'), 'wb') as df:
                pickle.dump(SI, df)
        else:
            with open(join(data_loc, str(mice[r])+'-'+env1s[l]+'-'+env2s[l]+'.pkl'), 'rb') as handle:
                SI = pickle.load(handle)            

        xy_max = np.nanmax(SI)
        ax.plot(SI[0, :], SI[1, :], 'ok', markeredgewidth=0, markersize = 2)
        ax.set_xticks([0,1,2,3,4,5])
        ax.set_yticks([0,1,2,3,4,5])
        ax.set_xlabel(env1s[l])
        ax.set_ylabel(env2s[l])
        ax.set_title(str(mice[r]))
        res = linregress(SI[0, :], SI[1, :], alternative='greater')
        slope, intercpt = res[0], res[1]

        ax.plot([0, xy_max], [intercpt, slope*xy_max+intercpt], color = 'orange')
        ax.axis([0, 5, 0, 5])
        print(r, l)
        print(res, end='\n\n')

plt.tight_layout()
plt.savefig(join(p,'SI.svg'), dpi = 1200)
plt.savefig(join(p,'SI.png'), dpi = 1200)
plt.close()


    