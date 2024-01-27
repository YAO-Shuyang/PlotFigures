# We can record tens of, if not hundreds of, cell spontaneously within a session using in vivo calcium imaging methods, and many of these cells recorded show a robust
# capacity of spatial coding. These cells are place cells which are identified by our criteria.

# Many of place cells exhibit a multifield spatial code, and the field number of each cell obeys certain distribution.
# This code is to draw the distribution figure for each session and save them.

from mylib.statistic_test import *
import scipy.stats
from mylib.stats.gamma_poisson import gamma_poisson_pdf

code_id = '0028 - Place Field Number Distribution Statistics'
loc = os.path.join(figpath, code_id)
mkdir(loc)
        
# Test data against negative binomial distribution
if os.path.exists(os.path.join(figdata,code_id+' [Monte Carlo].pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['lam', 'Poisson KS Statistics', 'Poisson KS P-Value', 
                                                      'r', 'p', 'nbinom KS Statistics', 'nbinom KS P-Value',
                                                      'mean', 'sigma', 'Normal KS Statistics', 'Normal KS P-Value'], f = f1,
                              function = FieldDistributionStatistics_TestAll_Interface, 
                              file_name = code_id+' [Monte Carlo]', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata,code_id+' [Monte Carlo].pkl'), 'rb') as handle:
        Data = pickle.load(handle)

op_num = np.where((Data['Maze Type'] == 'Open Field'))[0].shape[0]
m1_num = np.where(Data['Maze Type'] == 'Maze 1')[0].shape[0]
m2_num = np.where(Data['Maze Type'] == 'Maze 2')[0].shape[0]

poisson_op = np.where((Data['Poisson KS P-Value'] >= 0.05)&(Data['Maze Type'] == 'Open Field'))[0].shape[0]                                                              
poisson_m1 = np.where((Data['Poisson KS P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 1'))[0].shape[0]
poisson_m2 = np.where((Data['Poisson KS P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 2'))[0].shape[0]
print("Poisson No reject:", poisson_op, op_num, poisson_m1, m1_num, poisson_m2, m2_num)
#normal_op = np.where((Data['KS P-Value'] >= 0.05)&(Data['Maze Type'] == 'Open Field'))[0].shape[0]/op_num*100
#normal_m1 = np.where((Data['KS P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 1'))[0].shape[0]/m1_num*100
##normal_m2 = np.where((Data['KS P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 2'))[0].shape[0]/m2_num*100
#print("Normal No reject:", normal_op, op_num, normal_m1, m1_num, normal_m2, m2_num)
nb_op = np.where((Data['nbinom KS P-Value'] >= 0.05)&(Data['Maze Type'] == 'Open Field'))[0].shape[0]
nb_m1 = np.where((Data['nbinom KS P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 1'))[0].shape[0]
nb_m2 = np.where((Data['nbinom KS P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 2'))[0].shape[0]
print("Negative Binomial No reject:", nb_op, op_num, nb_m1, m1_num, nb_m2, m2_num)


ovm1idx = np.where((Data['nbinom KS P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 1'))[0]
ovm2idx = np.where((Data['nbinom KS P-Value'] >= 0.05)&(Data['Maze Type'] == 'Maze 2'))[0]
print("Overlaped No reject:", ovm1idx.shape[0], m1_num, ovm2idx.shape[0], m2_num)
data = Data['r'][np.where((Data['nbinom KS P-Value'] >= 0.05))[0]]
print_estimator(data)

r, p = Data['r'][ovm1idx], Data['p'][ovm1idx]
ranges = np.zeros(r.shape[0])
for i in range(r.shape[0]):
    ranges[i] = gamma_poisson_pdf(r[i], p[i]/(1-p[i]))

fig = plt.figure(figsize=(4,4))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(r, 1-p,'o', color='k', markersize=2, markeredgewidth=0)
ax.plot([0.57], [1-0.14], 'o', color='red', markersize=2, markeredgewidth=0)
ax.set_aspect("auto")
ax.set_xlim([0.1, 100000])
ax.set_ylim(0.0001,1)
ax.semilogx()
ax.semilogy()
plt.savefig(join(loc, "r,p - Maze 1.png"), dpi = 600)
plt.savefig(join(loc, "r,p - Maze 1.svg"), dpi = 600)
plt.close()

fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
counts = ax.hist(ranges, bins=24, range=(0,12), color='k', label='Maze 1')[0]
print(np.cumsum(counts))
ax.set_xlim(0, np.max(int(np.max(ranges))+1))
ax.set_ylim(0, 40)
ax.set_yticks(np.linspace(0, 40, 5))
plt.savefig(join(loc, "λ range [Maze 1].png"), dpi=600)
plt.savefig(join(loc, "λ range [Maze 1].svg"), dpi=600)
plt.close()

r, p = Data['r'][ovm2idx], Data['p'][ovm2idx]

ranges2 = np.zeros(r.shape[0])
for i in range(r.shape[0]):
    ranges2[i] = gamma_poisson_pdf(r[i], p[i]/(1-p[i]))

maxr, minr = np.argmax(r), np.argmin(r)

x, y = gamma_poisson_pdf(r[maxr], p[maxr]/(1-p[maxr]), output="pdf")
lb, ub = gamma_poisson_pdf(r[maxr], p[maxr]/(1-p[maxr]), output="bound")
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x, y, color='k', linewidth=0.5)
ax.axvline(x=lb, color='red', linewidth=0.5)
ax.axvline(x=ub, color='red', linewidth=0.5)
ax.set_xlim(0, 12)
ax.set_yticks(ColorBarsTicks(peak_rate=np.nanmax(y), is_auto=True, tick_number=5))
ax.set_title(f"r = {r[maxr]:.2f}, p = {p[maxr]:.2f}, idx = {maxr}\nlower bound = {lb:.2f}, upper bound = {ub:.2f}")
plt.tight_layout()
plt.savefig(join(loc, "r,p - max sample.png"), dpi = 600)
plt.savefig(join(loc, "r,p - max sample.svg"), dpi = 600)
plt.close()

x, y = gamma_poisson_pdf(r[minr], p[minr]/(1-p[minr]), output="pdf")
lb, ub = gamma_poisson_pdf(r[minr], p[minr]/(1-p[minr]), output="bound")
fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(x, y, color='k', linewidth=0.5)
ax.axvline(x=lb, color='red', linewidth=0.5)
ax.axvline(x=ub, color='red', linewidth=0.5)
ax.set_xlim(0, 12)
ax.set_title(f"r = {r[minr]:.2f}, p = {p[minr]:.2f}, idx = {minr}\nlower bound = {lb:.2f}, upper bound = {ub:.2f}")
plt.tight_layout()
plt.savefig(join(loc, "r,p - min sample.png"), dpi = 600)
plt.savefig(join(loc, "r,p - min sample.svg"), dpi = 600)
plt.close()

fig = plt.figure(figsize=(4,4))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.plot(r, 1-p,'o', color='k', markersize=2, markeredgewidth=0)
ax.plot([0.57], [1-0.14], 'o', color='red', markersize=2, markeredgewidth=0)
ax.plot(r[maxr], 1-p[maxr], 'o', markersize=2, markeredgewidth=0, label='max')
ax.plot(r[minr], 1-p[minr], 'o', markersize=2, markeredgewidth=0, label='min')
ax.legend()
ax.set_aspect("auto")
ax.set_xlim([0.1, 100000])
ax.set_ylim(0.0001,1)
ax.semilogx()
ax.semilogy()
plt.savefig(join(loc, "r,p - Maze 2.png"), dpi = 600)
plt.savefig(join(loc, "r,p - Maze 2.svg"), dpi = 600)
plt.close()

fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
counts = ax.hist(ranges2, bins=20, range=(0,10), color='k', label='Maze 2')[0]
print(np.cumsum(counts))
ax.set_xlim(0, np.max(int(np.max(ranges2))+1))
ax.set_ylim(0, 15)
ax.set_yticks(np.linspace(0, 15, 6))
plt.savefig(join(loc, "λ range [Maze 2].png"), dpi=600)
plt.savefig(join(loc, "λ range [Maze 2].svg"), dpi=600)
plt.close()