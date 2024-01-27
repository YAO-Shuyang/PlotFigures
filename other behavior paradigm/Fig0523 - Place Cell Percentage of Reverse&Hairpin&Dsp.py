from mylib.statistic_test import *

code_id = r"0523 - Place Cell Percentage of Reverse&Hairpin&Dsp"

loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(os.path.join(figdata, code_id+' [Reverse].pkl')) == False:
    ReverseData = DataFrameEstablish(variable_names = ['Percentage', 'Direction', 'Cell Number', 'Place Cell Number'], 
                              f = f3, function = PlaceCellPercentage_ReverseInterface, 
                              file_name = code_id+' [Reverse]', behavior_paradigm = 'ReverseMaze')
else:
    with open(os.path.join(figdata, code_id+' [Reverse].pkl'), 'rb') as handle:
        ReverseData = pickle.load(handle)
        
if os.path.exists(os.path.join(figdata, code_id+' [Hairpin].pkl')) == False:
    HairpinData = DataFrameEstablish(variable_names = ['Percentage', 'Direction', 'Cell Number', 'Place Cell Number'], 
                              f = f4, function = PlaceCellPercentage_ReverseInterface, 
                              file_name = code_id+' [Hairpin]', behavior_paradigm = 'HairpinMaze')
else:
    with open(os.path.join(figdata, code_id+' [Hairpin].pkl'), 'rb') as handle:
        HairpinData = pickle.load(handle)

Data = {}
nlen = len(ReverseData['Percentage'])
totlen = nlen + len(HairpinData['Percentage'])
for k in ReverseData.keys():
    Data[k] = np.concatenate([ReverseData[k], HairpinData[k]])

markercolors = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[0]]

Data['Paradigm'] = np.concatenate([np.repeat('Reverse', nlen), np.repeat('Hairpin', totlen-nlen)])
Data['hue'] = np.array([Data['Paradigm'][i]+'-'+Data['Direction'][i] for i in range(totlen)])
Data['Percentage'] = Data['Percentage'] * 100
ReverseData['Percentage'] = ReverseData['Percentage'] * 100
HairpinData['Percentage'] = HairpinData['Percentage'] * 100
    
plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Training Day',
    y = 'Percentage',
    data = ReverseData,
    hue = 'Direction',
    #style=Data['MiceID'][idx],
    palette=BidirectionalPalette,
    err_style='bars',
    ax=ax, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    #err_kws={'elinewidth':1, 'capthick':1, 'capsize':2},
    linewidth=0.5,
    estimator=np.median,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
)
ax.set_ylim(0, 100)
plt.savefig(join(loc, 'Reverse Place Cell Percentage.png'), dpi=600)
plt.savefig(join(loc, 'Reverse Place Cell Percentage.svg'), dpi=600)
plt.close()


plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x = 'Training Day',
    y = 'Percentage',
    data = HairpinData,
    hue = 'Direction',
    #style=Data['MiceID'][idx],
    palette=BidirectionalPalette,
    err_style='bars',
    ax=ax, 
    marker='o',
    markeredgecolor=None,
    markersize=2,
    legend=False,
    #err_kws={'elinewidth':1, 'capthick':1, 'capsize':2},
    linewidth=0.5,
    estimator=np.median,
    err_kws={'elinewidth':0.5, 'capthick':0.5, 'capsize':3},
)
ax.set_ylim(0, 100)
plt.savefig(join(loc, 'Hairpin Place Cell Percentage.png'), dpi=600)
plt.savefig(join(loc, 'Hairpin Place Cell Percentage.svg'), dpi=600)
plt.close()