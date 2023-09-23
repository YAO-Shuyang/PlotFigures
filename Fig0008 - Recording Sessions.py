from mylib.statistic_test import *

code_id = '0008 - Recording Sessions'
p = os.path.join(figpath, code_id)
dataloc = os.path.join(figdata, code_id)
mkdir(p)
mkdir(dataloc)

colors = sns.color_palette("rocket", 3)[1::]


if os.path.exists(os.path.join(dataloc, '11095.pkl')):
    with open(os.path.join(dataloc, '11095.pkl'), 'rb') as handle:
        ses = pickle.load(handle)
else:
    f = recording_sessions_95
    ses = np.zeros((8, 48), dtype=np.float64)
    tits = [i for i in range(1,44)]
    for i, d in enumerate(tits):
        ses[:, i] = f[d][0:8]
        
    with open(os.path.join(dataloc, '11095.pkl'), 'wb') as f:
        pickle.dump(ses, f)

def plot_grid(ax: Axes):
    for i in range(0,9):
        ax.plot([-0.5, 42.5], [i-0.5, i-0.5], color = 'black', linewidth = 1)
        
    for i in range(0,43):
        ax.plot([i-0.5, i-0.5], [-0.5, 7.5], color = 'black', linewidth = 1)
        
    return ax
    
fig = plt.figure(figsize=(8,3))
ax = Clear_Axes(plt.axes())
plot_grid(ax)
ax.set_aspect("equal")
ax.axis([-0.6, 42.6, -0.6, 7.6])
plt.imshow(ses)
ax.invert_yaxis()
plt.savefig(os.path.join(p, 'Recording Sessions - 11095.png'), dpi =2400)
plt.savefig(os.path.join(p, 'Recording Sessions - 11095.svg'), dpi =2400)
plt.close()











if os.path.exists(os.path.join(dataloc, '11092.pkl')):
    with open(os.path.join(dataloc, '11092.pkl'), 'rb') as handle:
        ses = pickle.load(handle)
else:
    f = recording_sessions_92
    ses = np.zeros((8, 48), dtype=np.float64)
    tits = [i for i in range(1,44)]
    for i, d in enumerate(tits):
        ses[:, i] = f[d][0:8]
        
    with open(os.path.join(dataloc, '11092.pkl'), 'wb') as f:
        pickle.dump(ses, f)

def plot_grid(ax: Axes):
    for i in range(0,9):
        ax.plot([-0.5, 42.5], [i-0.5, i-0.5], color = 'black', linewidth = 1)
        
    for i in range(0,43):
        ax.plot([i-0.5, i-0.5], [-0.5, 7.5], color = 'black', linewidth = 1)
        
    return ax
    
fig = plt.figure(figsize=(8,3))
ax = Clear_Axes(plt.axes())
plot_grid(ax)
ax.set_aspect("equal")
ax.axis([-0.6, 42.6, -0.6, 7.6])
plt.imshow(ses)
ax.invert_yaxis()
plt.savefig(os.path.join(p, 'Recording Sessions - 11092.png'), dpi =2400)
plt.savefig(os.path.join(p, 'Recording Sessions - 11092.svg'), dpi =2400)
plt.close()










if os.path.exists(os.path.join(dataloc, '10209.pkl')):
    with open(os.path.join(dataloc, '10209.pkl'), 'rb') as handle:
        ses = pickle.load(handle)
else:
    f = recording_sessions_09
    ses = np.zeros((10, 42), dtype=np.float64)
    tits = [i for i in range(1,43)]
    for i, d in enumerate(tits):
        ses[:, i] = f[d][0:10]
        
    with open(os.path.join(dataloc, '10209.pkl'), 'wb') as f:
        pickle.dump(ses, f)

def plot_grid2(ax: Axes):
    for i in range(0,11):
        ax.plot([-0.5, 41.5], [i-0.5, i-0.5], color = 'black', linewidth = 1)
        
    for i in range(0,43):
        ax.plot([i-0.5, i-0.5], [-0.5, 9.5], color = 'black', linewidth = 1)
        
    return ax
    
fig = plt.figure(figsize=(8,3))
ax = Clear_Axes(plt.axes())
plot_grid2(ax)
ax.set_aspect("equal")
ax.axis([-0.6, 41.6, -0.6, 9.6])
plt.imshow(ses, vmin=0, vmax=2)
ax.invert_yaxis()
plt.savefig(os.path.join(p, 'Recording Sessions - 10209.png'), dpi =2400)
plt.savefig(os.path.join(p, 'Recording Sessions - 10209.svg'), dpi =2400)
plt.close()





if os.path.exists(os.path.join(dataloc, '10212.pkl')):
    with open(os.path.join(dataloc, '10212.pkl'), 'rb') as handle:
        ses = pickle.load(handle)
else:
    f = recording_sessions_09
    ses = np.zeros((10, 42), dtype=np.float64)
    tits = [i for i in range(1,43)]
    for i, d in enumerate(tits):
        ses[:, i] = f[d][0:10]
        
    with open(os.path.join(dataloc, '10212.pkl'), 'wb') as f:
        pickle.dump(ses, f)
    
fig = plt.figure(figsize=(8,3))
ax = Clear_Axes(plt.axes())
plot_grid2(ax)
ax.set_aspect("equal")
ax.axis([-0.6, 41.6, -0.6, 9.6])
plt.imshow(ses, vmin=0, vmax=2)
ax.invert_yaxis()
plt.savefig(os.path.join(p, 'Recording Sessions - 10212.png'), dpi =2400)
plt.savefig(os.path.join(p, 'Recording Sessions - 10212.svg'), dpi =2400)
plt.close()



if os.path.exists(os.path.join(dataloc, '10209-2.pkl')):
    with open(os.path.join(dataloc, '10209-2.pkl'), 'rb') as handle:
        ses = pickle.load(handle)
else:
    f = recording_sessions_12_2
    ses = np.zeros((8, 26), dtype=np.float64)
    tits = [i for i in range(1,27)]
    for i, d in enumerate(tits):
        ses[:, i] = f[d][0:8]
        
    with open(os.path.join(dataloc, '10209-2.pkl'), 'wb') as f:
        pickle.dump(ses, f)

def plot_grid2(ax: Axes):
    for i in range(0,9):
        ax.plot([-0.5, 25.5], [i-0.5, i-0.5], color = 'black', linewidth = 1)
        
    for i in range(0,27):
        ax.plot([i-0.5, i-0.5], [-0.5, 7.5], color = 'black', linewidth = 1)
        
    return ax
    
fig = plt.figure(figsize=(8,3))
ax = Clear_Axes(plt.axes())
plot_grid2(ax)
ax.set_aspect("equal")
ax.axis([-0.6, 25.6, -0.6, 7.6])
plt.imshow(ses, vmin=0, vmax=2)
ax.invert_yaxis()
plt.savefig(os.path.join(p, 'Recording Sessions - 10209-2.png'), dpi =2400)
plt.savefig(os.path.join(p, 'Recording Sessions - 10209-2.svg'), dpi =2400)
plt.close()








if os.path.exists(os.path.join(dataloc, '10212-2.pkl')):
    with open(os.path.join(dataloc, '10212-2.pkl'), 'rb') as handle:
        ses = pickle.load(handle)
else:
    f = recording_sessions_12_2
    ses = np.zeros((8, 26), dtype=np.float64)
    tits = [i for i in range(1,27)]
    for i, d in enumerate(tits):
        ses[:, i] = f[d][0:8]
        
    with open(os.path.join(dataloc, '10212-2.pkl'), 'wb') as f:
        pickle.dump(ses, f)
    
fig = plt.figure(figsize=(8,3))
ax = Clear_Axes(plt.axes())
plot_grid2(ax)
ax.set_aspect("equal")
ax.axis([-0.6, 25.6, -0.6, 7.6])
plt.imshow(ses, vmin=0, vmax=2)
ax.invert_yaxis()
plt.savefig(os.path.join(p, 'Recording Sessions - 10212-2.png'), dpi =2400)
plt.savefig(os.path.join(p, 'Recording Sessions - 10212-2.svg'), dpi =2400)
plt.close()