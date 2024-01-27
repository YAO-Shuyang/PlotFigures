from mylib.statistic_test import *

code_id = "0301 - Percentage of enhanced, weakened and robust cells"
loc = join(figpath, code_id)
mkdir(loc)

with open (r'E:\Data\Cross_maze\10209\20230728\session 2\trace.pkl', 'rb') as handle:
    trace = pickle.load(handle)
    
count = {
    'weakened': 0,
    'enhanced': 0,
    'retained': 0
}

info = cp.deepcopy(trace['place_fields_info'])
for key in tqdm(info.keys()):
    if trace['is_placecell'][key[0]] == 1:
        count[info[key]['ctype']] += 1

def pieplot(count: dict):
    print(count)
    Data = {
        'Number': [count[k] for k in count.keys()],
    }
    tot_num = np.sum(Data['Number']) 
       
    Data['CType'] = [k+" "+str(int(count[k]/tot_num*100))+"%" for k in count.keys()]
    
    
    colors = sns.color_palette("rocket", 3)
    fig = plt.figure(figsize=(4, 4))
    ax = Clear_Axes(plt.axes())
    ax.pie(Data['Number'], labels=Data['CType'], colors=colors)
    plt.show()
    
pieplot(count=count)

laps_info = {
    'emerge lap': [],
    'disappear lap': [],
    'active lap percent': []
}

T = trace['correct_time'][-1]
laps = trace['laps']

for key in tqdm(info.keys()):
    if trace['is_placecell'][key[0]] == 1:
        laps_info['emerge lap'].append(info[key]['cal_events_time'][int(info[key]['emerge lap']), -1]/T*laps+1),
        laps_info['disappear lap'].append(info[key]['cal_events_time'][int(info[key]['disappear lap'])-1, -1]/T*laps+1),
        laps_info['active lap percent'].append(info[key]['active lap percent'])

from scipy.optimize import curve_fit
def exponential(x, a, tao):
    return a*np.exp(-x/tao)


fig = plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
count = ax.hist(laps_info['emerge lap'], bins=laps, range=(0.5,laps+0.5), rwidth=0.8, color = 'gray', alpha=0.5)[0]

x = np.linspace(1, laps, laps)
x_pred = np.linspace(0, laps, 1000)
popt, pcov = curve_fit(exponential, x, count, p0=[count[0], 1])
y_pred = exponential(x_pred, popt[0], popt[1])
ax.plot(x_pred, y_pred, linewidth=0.8, color = 'red')
ax.set_xlim([0.5, laps+0.5])
ax.set_xticks(ColorBarsTicks(peak_rate=laps, is_auto=True, tick_number=5))
ax.set_xlabel("field emerge lap")
ax.set_ylabel("field count")
plt.tight_layout()
plt.savefig(join(loc, 'emerge lap.png'), dpi=600)
plt.savefig(join(loc, 'emerge lap.svg'), dpi=600)
plt.close()

fig = plt.figure(figsize=(4, 3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(laps_info['disappear lap'], bins=laps, range=(0.5,laps+0.5), rwidth=0.8, color = 'gray', alpha=0.5)
ax.set_xlim([0.5, laps+0.5])
ax.set_xlabel("field disappear lap")
ax.set_ylabel("field count")
plt.show()
