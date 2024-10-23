from mylib.statistic_test import *
from mylib.dsp.starting_cell import *

code_id = '0835 - Allocentric Proportion with Spatial Distance'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ["Position", "Field Type"],
                              f=f2, 
                              function = AllocentricProportionWithSpatialDistance_DSP_Interface, 
                              file_name = code_id, behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
    
if os.path.exists(os.path.join(figdata, code_id+' [differentiate].pkl')) == False:
    SData = {
        "Relative Distance": [],
        "Rate": [],
        "Route": [],
        "Field Type": []
    }
    
    x = np.arange(1, 112)
    y = np.histogram(Data['Position'][np.where((Data['Field Type'] == 0))[0]], bins = 111, 
                         range=(0.5, 111.5))[0]
    y_average = np.histogram(Data['Position'], bins = 111,
                         range=(0.5, 111.5))[0]
    dy = y / y_average
        
    route_id = np.array([0, 4, 1, 5, 2])
    for n, seg in enumerate([seg2, seg3, seg4, seg5]):
        SData['Relative Distance'].append(x[segs[n]:segs[n+1]-1] - x[segs[n]])
        SData['Rate'].append(dy[segs[n]:segs[n+1]-1])
        SData['Route'].append(np.repeat(route_id[n+1], segs[n+1]-1-segs[n]))
        SData['Field Type'].append(np.repeat(0, segs[n+1]-1-segs[n]))
        
    x = np.arange(1, 112)
    y = np.histogram(Data['Position'][np.where((Data['Field Type'] == 1))[0]], bins = 111, 
                         range=(0.5, 111.5))[0]
    dy = y / y_average
        
    route_id = np.array([0, 4, 1, 5, 2, 6, 3])
    for n, seg in enumerate([seg2, seg3, seg4, seg5, seg6, seg7]):
        SData['Relative Distance'].append(x[segs[n]:segs[n+1]-1] - x[segs[n]])
        SData['Rate'].append(dy[segs[n]:segs[n+1]-1])
        SData['Route'].append(np.repeat(route_id[n+1], segs[n+1]-1-segs[n]))
        SData['Field Type'].append(np.repeat(1, segs[n+1]-1-segs[n]))
    
    for k in SData.keys():
        SData[k] = np.concatenate(SData[k])
        
    with open(os.path.join(figdata, code_id+' [differentiate].pkl'), 'wb') as handle:
        pickle.dump(SData, handle)
    
    D = pd.DataFrame(SData)
    D.to_excel(os.path.join(figdata, code_id+' [differentiate].xlsx'), index = False)
else:
    with open(os.path.join(figdata, code_id+' [differentiate].pkl'), 'rb') as handle:
        SData = pickle.load(handle)

fig = plt.figure(figsize = (2, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = "Relative Distance",
    y = "Rate",
    data = SData,
    hue="Field Type",
    ax = ax,
    linewidth=0.5,
    err_kws={"linewidth": 0.5, 'edgecolor':None},
    palette=["#ABD0D1", DSPPalette[2]],
    zorder=2
)
ax.set_ylim(0, 1)
ax.set_xlim(0, 16)
ax.set_xticks(np.linspace(0, 16, 5))
plt.savefig(os.path.join(loc, f'line.svg'), dpi=600)
plt.savefig(os.path.join(loc, f'line.png'), dpi=600)
plt.show()

fig = plt.figure(figsize = (2, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.ecdfplot(
    x='Position',
    hue='Field Type',
    data = Data,
    ax = ax,
    palette=["#ABD0D1", DSPPalette[2]],
    linewidth=0.5
)
ax.set_xlim(0.5, 111.5)
ax.set_xticks(np.append(np.linspace(0, 100, 9)+0.5, 111.5))
plt.savefig(os.path.join(loc, f'CDF.svg'), dpi=600)
plt.savefig(os.path.join(loc, f'CDF.png'), dpi=600)
plt.show()