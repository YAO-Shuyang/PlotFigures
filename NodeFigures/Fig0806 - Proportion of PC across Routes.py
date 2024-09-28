from mylib.statistic_test import *

code_id = "0806 - Proportion of PC across Routes"
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Route', 'Proportion'],
                              f=f2, 
                              function = ProportionOfPCAcrossRoutes_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:    
        Data = pickle.load(handle)
        
Data['Route'] = Data['Route'].astype(np.int64)
        
fig = plt.figure(figsize=(5, 2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
order = [0, 4, 1, 5, 2, 6, 3]
sns.barplot(
    x='Training Day',
    y='Proportion',
    hue='Route',
    data=Data,
    ax=ax,
    palette=[DSPPalette[i] for i in order],
    capsize=0.2,
    err_kws={"linewidth": 0.5, 'color':'k'},
    edgecolor = 'k',
    linewidth = 0.5,
    hue_order=[1, 5, 2, 6, 3, 7, 4],
    zorder=2
)
ax.set_ylim(0, 1.03)
ax.set_yticks(np.linspace(0, 1, 6))
plt.savefig(os.path.join(loc, 'PC proportion.png'), dpi=600)
plt.savefig(os.path.join(loc, 'PC proportion.svg'), dpi=600)
plt.close()

# Perform One-way ANOVA
from scipy.stats import f_oneway
for d in np.unique(Data['Training Day']):
    
    grouplists = []
    for route in np.unique(Data['Route']):
        idx = np.where((Data['Training Day'] == d) & (Data['Route'] == route))[0]
        grouplists.append(Data['Proportion'][idx])
    
    F, p = f_oneway(*grouplists)
    print(f"{d} - F = {F}, p = {p}")

routes_convertor = np.array([len(CP_DSP[i]) for i in range(7)])
x = routes_convertor[Data['Route']-1]
print(f" Pearson Correlation: {pearsonr(Data['Proportion'], x)}")

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x.reshape(-1, 1), Data['Proportion'])

slope, intercept = reg.coef_, reg.intercept_
print(f" slope: {slope}, intercept: {intercept}")
x_fit = np.linspace(np.min(x), np.max(x), 2)
y_fit = intercept + slope * x_fit

fig = plt.figure(figsize=(3, 2.5))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
x_uniq = np.unique(x)
for i in range(7):
    idx = np.where(x == x_uniq[i])[0]
    ax.bar(
        x_uniq[i], 
        Data['Proportion'][idx], 
        capsize = 0.5, 
        width=0.8, 
        color = DSPPalette[6-i], 
        ecolor='k', 
        error_kw={"linewidth": 0.5}
    )
sns.scatterplot(
    x = x,
    y = Data['Proportion'],
    hue = x,
    ax = ax,
    palette = [DSPPalette[i] for i in order][::-1],
    s = 20,
    alpha = 0.8,
    edgecolor = 'k',
    linewidth = 0.2
)
ax.plot(x_fit, y_fit, color = 'k', linewidth = 0.5)
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_xlim(20, 120)
ax.set_xticks(np.linspace(20, 120, 6))
ax.invert_xaxis()
plt.savefig(os.path.join(loc, 'PC proportion vs route.png'), dpi=600)
plt.savefig(os.path.join(loc, 'PC proportion vs route.svg'), dpi=600)
plt.close()