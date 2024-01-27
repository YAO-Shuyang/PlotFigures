from mylib.statistic_test import *

code_id = '0006 - Footprint'
p = os.path.join(figpath, code_id)
mkdir(p)
'''
for i in range(len(f1)):
    print(i,"----------------------------------------")
    SFPpath = os.path.join(f1['recording_folder'][i], 'SFP.mat')

    if os.path.exists(SFPpath):
        with h5py.File(SFPpath, 'r') as f:
            sfp = np.array(f['SFP'])
            footprint = np.nanmax(sfp,axis = 2)

    loc = os.path.join(p, str(int(f1['MiceID'][i])), 'session '+str(int(f1['session'][i])))
    mkdir(loc)
    file_name = str(int(f1['date'][i]))
    
    ax = Clear_Axes(plt.axes())
    im = ax.imshow(footprint, cmap = 'gray')
    cbar = plt.colorbar(im, ax = ax)
    ax.set_title('Mouse #'+str(int(f1['MiceID'][i]))+' '+str(int(f1['date'][i]))+' session '+str(int(f1['session'][i])))
    plt.savefig(os.path.join(loc, file_name+'.png'), dpi = 600)
    plt.savefig(os.path.join(loc, file_name+'.svg'), dpi = 600)
    plt.close()
    print("Done.",end='\n\n\n')
'''
import cv2
def select_roi(sfp: np.ndarray):
    boundaries = []
    for i in range(sfp.shape[2]):
        # Find contours in the binary image
        image = np.uint8(sfp[:, :, i]*255)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the longest contour (assuming it is the boundary of the closed shape)
        longest_contour = max(contours, key=len)

        longest_contour = np.concatenate([longest_contour, [[[longest_contour[0, 0, 0], longest_contour[0, 0, 1]]]] ], axis=0)
        # Convert the contour points to numpy array
        traced_boundary = np.array(longest_contour)
        boundaries.append(traced_boundary)
        
    
    return boundaries

def plot_footprint(cell_idx:list|np.ndarray, SFP_loc:str = None, save_loc:str = None, is_plot_boundary: bool = True):
    if os.path.exists(SFP_loc):
        with h5py.File(SFP_loc, 'r') as f:
            sfp = np.array(f['SFP'])
            footprint = np.nanmax(sfp,axis = 2)
    else:
        return

    boundaries = select_roi(sfp)
    n = sfp.shape[2]

    plt.figure(figsize=(4.5,7.5))
    ax = Clear_Axes(plt.axes())
    
    if is_plot_boundary:
        #im = ax.imshow(footprint, cmap = 'gray')
        #cbar = plt.colorbar(im, ax = ax)
        for i in tqdm(range(n)):
            if i in cell_idx:
                continue
            color = (169/255, 169/255, 169/255)
            ax.plot(boundaries[i][:, 0, 0], boundaries[i][:, 0, 1], color = color, linewidth = 0.5, alpha=0.8)
    
        for i in cell_idx:
            ax.plot(boundaries[i][:, 0, 0], boundaries[i][:, 0, 1], color = 'cornflowerblue', linewidth = 0.5, alpha=0.8)

            peak = np.nanmax(sfp[:,:,i])
            peak_idx = np.where(sfp[:,:,i] == peak)
            #ax.plot(peak_idx[1][0], peak_idx[0][0], '.', markersize = 3, color ='yellow')
            ax.text(x = peak_idx[1][0], y = peak_idx[0][0], s = str(i+1), color = 'red')
    
        ax.set_aspect("equal")

        if save_loc is None:
            plt.show()
        else:
            plt.savefig(save_loc+'.png', dpi = 2400)
            plt.savefig(save_loc+'.svg', dpi = 2400)
            

#plot_footprint(np.array([4, 6, 9, 27, 36, 43, 44, 72, 85, 86])-1, SFP_loc = r'E:\Data\FinalResults\0006 - Footprint\10227\20230928\session 2\SFP.mat', 
#               save_loc = r"E:\Data\FinalResults\0006 - Footprint\10227\20230928\session 2\footprint")

def plot_overdays(
    dir_name: str,
    save_loc: str,
    find_chars: str = "SFP"
):
    mkdir(save_loc)
    files = os.listdir(dir_name)
    
    for file in tqdm(files):
        if find_chars not in file:
            continue
        
        with h5py.File(os.path.join(dir_name, file), 'r') as f:
            sfp = np.array(f['SFP'])
            for i in range(sfp.shape[2]):
                sfp[:, :, i] = sfp[:, :, i] / np.nanmax(sfp[:, :, i])

            footprint = np.nanmax(sfp, axis = 2)
        
        plt.figure(figsize=(7.5,4.5))
        ax = Clear_Axes(plt.axes())
        ax.imshow(footprint.T, cmap='gray')
        plt.savefig(os.path.join(save_loc, file.split('.')[0]+'.png'), dpi=600)
        plt.savefig(os.path.join(save_loc, file.split('.')[0]+'.svg'), dpi=600)
        plt.close()
plot_overdays(r"E:\Data\Cross_maze\11095\Maze1-footprint", save_loc=os.path.join(p, "11095-Stage2-Maze 1"))
plot_overdays(r"E:\Data\Cross_maze\11095\Open Field-2-footprint", save_loc=os.path.join(p, "11095-Stage2-Open Field2"))
plot_overdays(r"E:\Data\Cross_maze\11095\Maze2-footprint", save_loc=os.path.join(p, "11095-Stage2-Maze 2"))
plot_overdays(r"E:\Data\Cross_maze\11095\Open Field-footprint", save_loc=os.path.join(p, "11095-Stage2-Open Field1"))
plot_overdays(r"E:\Data\Cross_maze\11092\Maze1-footprint", save_loc=os.path.join(p, "11092-Stage2-Maze 1"))
plot_overdays(r"E:\Data\Cross_maze\11092\Open Field-2-footprint", save_loc=os.path.join(p, "11092-Stage2-Open Field2"))
plot_overdays(r"E:\Data\Cross_maze\11092\Maze2-footprint", save_loc=os.path.join(p, "11092-Stage2-Maze 2"))
plot_overdays(r"E:\Data\Cross_maze\11092\Open Field-footprint", save_loc=os.path.join(p, "11092-Stage2-Open Field1"))
"""
plot_overdays(r"E:\Data\Cross_maze\10227\Maze1-footprint", save_loc=os.path.join(p, "10227-Stage1-Maze 1"))
plot_overdays(r"E:\Data\Cross_maze\10227\Maze1-2-footprint", save_loc=os.path.join(p, "10227-Stage2-Maze 1"))
plot_overdays(r"E:\Data\Cross_maze\10227\Maze2-footprint", save_loc=os.path.join(p, "10227-Stage2-Maze 2"))
plot_overdays(r"E:\Data\Cross_maze\10227\Open Field-footprint", save_loc=os.path.join(p, "10227-Stage1-Open Field1"))
plot_overdays(r"E:\Data\Cross_maze\10227\Open Field-2-footprint", save_loc=os.path.join(p, "10227-Stage1-Open Field2"))
plot_overdays(r"E:\Data\Cross_maze\10227\Open Field-3-footprint", save_loc=os.path.join(p, "10227-Stage2-Open Field3"))
plot_overdays(r"E:\Data\Cross_maze\10227\Open Field-4-footprint", save_loc=os.path.join(p, "10227-Stage2-Open Field4"))

plot_overdays(r"E:\Data\Cross_maze\10224\Maze1-footprint", save_loc=os.path.join(p, "10224-Stage1-Maze 1"))
plot_overdays(r"E:\Data\Cross_maze\10224\Maze1-2-footprint", save_loc=os.path.join(p, "10224-Stage2-Maze 1"))
plot_overdays(r"E:\Data\Cross_maze\10224\Maze2-footprint", save_loc=os.path.join(p, "10224-Stage2-Maze 2"))
plot_overdays(r"E:\Data\Cross_maze\10224\Open Field-footprint", save_loc=os.path.join(p, "10224-Stage1-Open Field1"))
plot_overdays(r"E:\Data\Cross_maze\10224\Open Field-2-footprint", save_loc=os.path.join(p, "10224-Stage1-Open Field2"))

plot_overdays(r"E:\Data\Cross_maze\10224\Open Field-3-footprint", save_loc=os.path.join(p, "10224-Stage2-Open Field3"))
plot_overdays(r"E:\Data\Cross_maze\10224\Open Field-4-footprint", save_loc=os.path.join(p, "10224-Stage2-Open Field4"))
"""
plot_overdays(r"E:\Data\Cross_maze\10212\Maze1-footprint", save_loc=os.path.join(p, "10212-Stage1-Maze 1"))
plot_overdays(r"E:\Data\Cross_maze\10212\Maze1-2-footprint", save_loc=os.path.join(p, "10212-Stage2-Maze 1"))
plot_overdays(r"E:\Data\Cross_maze\10212\Maze2-footprint", save_loc=os.path.join(p, "10212-Stage2-Maze 2"))
plot_overdays(r"E:\Data\Cross_maze\10212\Open Field-footprint", save_loc=os.path.join(p, "10212-Stage1-Open Field1"))
plot_overdays(r"E:\Data\Cross_maze\10212\Open Field-2-footprint", save_loc=os.path.join(p, "10212-Stage1-Open Field2"))
plot_overdays(r"E:\Data\Cross_maze\10212\Open Field-3-footprint", save_loc=os.path.join(p, "10212-Stage2-Open Field3"))
plot_overdays(r"E:\Data\Cross_maze\10212\Open Field-4-footprint", save_loc=os.path.join(p, "10212-Stage2-Open Field4"))

plot_overdays(r"E:\Data\Cross_maze\10209\Maze1-footprint", save_loc=os.path.join(p, "10209-Stage1-Maze 1"))
plot_overdays(r"E:\Data\Cross_maze\10209\Maze1-2-footprint", save_loc=os.path.join(p, "10209-Stage2-Maze 1"))
plot_overdays(r"E:\Data\Cross_maze\10209\Maze2-footprint", save_loc=os.path.join(p, "10209-Stage2-Maze 2"))
plot_overdays(r"E:\Data\Cross_maze\10209\Open Field-footprint", save_loc=os.path.join(p, "10209-Stage1-Open Field1"))
plot_overdays(r"E:\Data\Cross_maze\10209\Open Field-2-footprint", save_loc=os.path.join(p, "10209-Stage1-Open Field2"))
plot_overdays(r"E:\Data\Cross_maze\10209\Open Field-3-footprint", save_loc=os.path.join(p, "10209-Stage2-Open Field3"))
plot_overdays(r"E:\Data\Cross_maze\10209\Open Field-4-footprint", save_loc=os.path.join(p, "10209-Stage2-Open Field4"))