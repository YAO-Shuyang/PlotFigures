from mylib.statistic_test import *

code_id = "0404 - Poisson Distribution Sample"
loc = os.path.join(figpath, 'Independent Field', code_id)
mkdir(loc)


if os.path.exists(os.path.join(figdata, 'Field Counts Per Session.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Path Type', 'Threshold', 'Field Count', 'Field Number'], f = f1,
                              function = FieldCountPerSession_Interface,
                              file_name='Field Counts Per Session', behavior_paradigm = 'CrossMaze')
else:
    with open(os.path.join(figdata, 'Field Counts Per Session.pkl'), 'rb') as handle:
        Data = pickle.load(handle)

def plot_cumulative_distribution(Data, mouse: int, maze_type: str, save_loc: str):
    familiaridx = np.where((Data['Training Day'] != 'Day 1')&(Data['Training Day'] != 'Day 2'))[0]
    SubData = SubDict(Data, Data.keys(), familiaridx)
    
    nopreidx = np.where(SubData['Stage'] != 'PRE')[0]
    SubData = SubDict(SubData, SubData.keys(), nopreidx)
    
    idx = np.where((SubData['MiceID'] == mouse)&(SubData['Maze Type'] == maze_type))[0]
    SubData = SubDict(SubData, SubData.keys(), idx)

    if maze_type == 'Open Field':
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3))
        ax1, ax2 = Clear_Axes(axes[0], close_spines=['top','right'], ifxticks=True, ifyticks=True), Clear_Axes(axes[1], close_spines=['top','right'], ifxticks=True, ifyticks=True)
        
        # Axes 1, thre = 10, path = OP
        idx = np.where((SubData['Path Type'] == 'OP')&(SubData['Threshold'] == 10))[0]
        nday = int(SubData['Field Count'][idx].shape[0] / 50)
        Count = cp.deepcopy(SubData['Field Count'][idx])
        Count = np.reshape(Count, [nday, 50])
        mean_count = np.mean(Count, axis = 0)[:25]
    
        # Fit Poisson Distribution
        lam = EqualPoissonFit(np.arange(1, 26), mean_count/np.sum(mean_count))
        y = EqualPoisson(np.arange(1, 26), l = lam)
        y = y / np.sum(y) * np.sum(mean_count)  
        print(f"{mouse} {maze_type} OP, thre=10:  ", chisquare(f_obs=mean_count, f_exp=y))
        ax1.bar(
            np.arange(1, 26),
            mean_count,
            color='gray',
            width=0.8
        )
        ax1.plot(np.arange(1, 26), y, color = 'red', label = 'Equal Poisson\n'+'lambda:'+str(round(lam,3)))
        ax1.legend(title='Fit Type', facecolor='white', edgecolor='white', loc='upper right')
        sta, p = chisquare(f_obs=mean_count, f_exp=y)
        ax1.set_title(f"{maze_type} OP, thre=10, p={p}")
        try:
            ax1.set_xlim(0.5, np.where(mean_count == 0)[0][0]+0.5)
            ax1.set_xticks([1, np.where(mean_count == 0)[0][0]])
            
        except:
            ax1.set_xlim(0.5, 50.5)
        ax1.set_xlabel('Average Count')
        ax1.set_ylabel('Field Number per Cell')
        
        
        # Axes 1, thre = 5, path = CP
        idx = np.where((SubData['Path Type'] == 'OP')&(SubData['Threshold'] == 5))[0]
        nday = int(SubData['Field Count'][idx].shape[0] / 50)
        Count = cp.deepcopy(SubData['Field Count'][idx])
        Count = np.reshape(Count, [nday, 50])
        mean_count = np.mean(Count, axis = 0)[:25]
    
        # Fit Poisson Distribution
        lam = EqualPoissonFit(np.arange(1, 26), mean_count/np.sum(mean_count))
        y = EqualPoisson(np.arange(1, 26), l = lam) * np.sum(mean_count)    
        y = y / np.sum(y) * np.sum(mean_count)  
        print(f"{mouse} {maze_type} OP, thre=5:  ", chisquare(f_obs=mean_count, f_exp=y), end='\n\n')
        ax2.bar(
            np.arange(1, 26),
            mean_count,
            color='gray',
            width=0.8
        )
        ax2.plot(np.arange(1, 26), y, color = 'red', label = 'Equal Poisson\n'+'lambda:'+str(round(lam,3)))
        ax2.legend(title='Fit Type', facecolor='white', edgecolor='white', loc='upper right')
        sta, p = chisquare(f_obs=mean_count, f_exp=y)
        ax2.set_title(f"{maze_type} OP, thre=5, p={p}")
        try:
            ax2.set_xlim(0.5, np.where(mean_count == 0)[0][0]+0.5)
            ax2.set_xticks([1, np.where(mean_count == 0)[0][0]])
        except:
            ax2.set_xlim(0.5, 50.5)
        ax2.set_xlabel('Average Count')
        ax2.set_ylabel('Field Number per Cell')
    else:   
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,6))
        ax1, ax2, ax3, ax4 = Clear_Axes(axes[0, 0], close_spines=['top','right'], ifxticks=True, ifyticks=True), Clear_Axes(axes[0, 1], close_spines=['top','right'], ifxticks=True, ifyticks=True), Clear_Axes(axes[1, 0], close_spines=['top','right'], ifxticks=True, ifyticks=True), Clear_Axes(axes[1, 1], close_spines=['top','right'], ifxticks=True, ifyticks=True)

        
        # Axes 1, thre = 10, path = OP
        idx = np.where((SubData['Path Type'] == 'CP')&(SubData['Threshold'] == 10))[0]
        nday = int(SubData['Field Count'][idx].shape[0] / 50)
        Count = cp.deepcopy(SubData['Field Count'][idx])
        Count = np.reshape(Count, [nday, 50])
        mean_count = np.mean(Count, axis = 0)[:25]
    
        # Fit Poisson Distribution
        lam = EqualPoissonFit(np.arange(1, 26), mean_count/np.sum(mean_count))
        y = EqualPoisson(np.arange(1, 26), l = lam) * np.sum(mean_count)    
        y = y / np.sum(y) * np.sum(mean_count)  
        print(f"{mouse} {maze_type} CP, thre=10:  ", chisquare(f_obs=mean_count, f_exp=y))
        ax1.bar(
            np.arange(1, 26),
            mean_count,
            color='gray',
            width=0.8
        )
        ax1.plot(np.arange(1, 26), y, color = 'red', label = 'Equal Poisson\n'+'lambda:'+str(round(lam,3)))
        ax1.legend(title='Fit Type', facecolor='white', edgecolor='white', loc='upper right')
        sta, p = chisquare(f_obs=mean_count, f_exp=y)
        ax1.set_title(f"{maze_type} CP, thre=10, p={p}")
        try:
            ax1.set_xlim(0.5, np.where(mean_count == 0)[0][0]+0.5)
            ax1.set_xticks([1, np.where(mean_count == 0)[0][0]])
            
        except:
            ax1.set_xlim(0.5, 50.5)
        ax1.set_xlabel('Average Count')
        ax1.set_ylabel('Field Number per Cell')
        
        
        # Axes 1, thre = 5, path = CP
        idx = np.where((SubData['Path Type'] == 'CP')&(SubData['Threshold'] == 5))[0]
        nday = int(SubData['Field Count'][idx].shape[0] / 50)
        Count = cp.deepcopy(SubData['Field Count'][idx])
        Count = np.reshape(Count, [nday, 50])
        mean_count = np.mean(Count, axis = 0)[:25]
    
        # Fit Poisson Distribution
        lam = EqualPoissonFit(np.arange(1, 26), mean_count/np.sum(mean_count))
        y = EqualPoisson(np.arange(1, 26), l = lam) * np.sum(mean_count)    
        y = y / np.sum(y) * np.sum(mean_count)  
        print(f"{mouse} {maze_type} CP, thre=5:  ", chisquare(f_obs=mean_count, f_exp=y))
        ax2.bar(
            np.arange(1, 26),
            mean_count,
            color='gray',
            width=0.8
        )
        ax2.plot(np.arange(1, 26), y, color = 'red', label = 'Equal Poisson\n'+'lambda:'+str(round(lam,3)))
        ax2.legend(title='Fit Type', facecolor='white', edgecolor='white', loc='upper right')
        sta, p = chisquare(f_obs=mean_count, f_exp=y)
        ax2.set_title(f"{maze_type} CP, thre=5, p={p}")
        try:
            ax2.set_xlim(0.5, np.where(mean_count == 0)[0][0]+0.5)
            ax2.set_xticks([1, np.where(mean_count == 0)[0][0]])
            
        except:
            ax2.set_xlim(0.5, 50.5)
        ax2.set_xlabel('Average Count')
        ax2.set_ylabel('Field Number per Cell')
        
        # Axes 1, thre = 5, path = CP
        idx = np.where((SubData['Path Type'] == 'All')&(SubData['Threshold'] == 10))[0]
        nday = int(SubData['Field Count'][idx].shape[0] / 50)
        Count = cp.deepcopy(SubData['Field Count'][idx])
        Count = np.reshape(Count, [nday, 50])
        mean_count = np.mean(Count, axis = 0)[:25]
    
        # Fit Poisson Distribution
        lam = EqualPoissonFit(np.arange(1, 26), mean_count/np.sum(mean_count))
        y = EqualPoisson(np.arange(1, 26), l = lam) * np.sum(mean_count)  
        y = y / np.sum(y) * np.sum(mean_count)  
        print(f"{mouse} {maze_type} All, thre=10:  ", chisquare(f_obs=mean_count, f_exp=y))  
        ax3.bar(
            np.arange(1, 26),
            mean_count,
            color='gray',
            width=0.8
        )
        ax3.plot(np.arange(1, 26), y, color = 'red', label = 'Equal Poisson\n'+'lambda:'+str(round(lam,3)))
        ax3.legend(title='Fit Type', facecolor='white', edgecolor='white', loc='upper right')
        sta, p = chisquare(f_obs=mean_count, f_exp=y)
        ax3.set_title(f"{maze_type} All, thre=10, chi2p={p}")
        try:
            ax3.set_xlim(0.5, np.where(mean_count == 0)[0][0]+0.5)
            ax3.set_xticks([1, np.where(mean_count == 0)[0][0]])
        except:
            ax3.set_xlim(0.5, 50.5)
        ax3.set_xlabel('Average Count')
        ax3.set_ylabel('Field Number per Cell')
        
        
        # Axes 1, thre = 5, path = CP
        idx = np.where((SubData['Path Type'] == 'All')&(SubData['Threshold'] == 5))[0]
        nday = int(SubData['Field Count'][idx].shape[0] / 50)
        Count = cp.deepcopy(SubData['Field Count'][idx])
        Count = np.reshape(Count, [nday, 50])
        mean_count = np.mean(Count, axis = 0)[:25]
    
        # Fit Poisson Distribution
        lam = EqualPoissonFit(np.arange(1, 26), mean_count/np.sum(mean_count))
        y = EqualPoisson(np.arange(1, 26), l = lam) * np.sum(mean_count)  
        y = y / np.sum(y) * np.sum(mean_count)  
        print(f"{mouse} {maze_type} All, thre=5:  ", chisquare(f_obs=mean_count, f_exp=y), end='\n\n')  
        ax4.bar(
            np.arange(1, 26),
            mean_count,
            color='gray',
            width=0.8
        )
        ax4.plot(np.arange(1, 26), y, color = 'red', label = 'Equal Poisson\n'+'lambda:'+str(round(lam,3)))
        ax4.legend(title='Fit Type', facecolor='white', edgecolor='white', loc='upper right')
        sta, p = chisquare(f_obs=mean_count, f_exp=y)
        ax4.set_title(f"{maze_type} All, thre=5, chi2p={p}")
        try:
            ax4.set_xlim(0.5, np.where(mean_count == 0)[0][0]+0.5)
            ax4.set_xticks([1, np.where(mean_count == 0)[0][0]])
            ax4.set_xticks([1, np.where(mean_count == 0)[0][0]])
        except:
            ax4.set_xlim(0.5, 50.5)
        ax4.set_xlabel('Average Count')
        ax4.set_ylabel('Field Number per Cell')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, f"Session Average Field Count [{mouse}, {maze_type}].png"), dpi=600)
    plt.savefig(os.path.join(save_loc, f"Session Average Field Count [{mouse}, {maze_type}].svg"), dpi=600)
    plt.close()
    

plot_cumulative_distribution(Data, 10209, 'Open Field', loc)
plot_cumulative_distribution(Data, 10209, 'Maze 1', loc)
plot_cumulative_distribution(Data, 10209, 'Maze 2', loc)

plot_cumulative_distribution(Data, 10212, 'Open Field', loc)
plot_cumulative_distribution(Data, 10212, 'Maze 1', loc)
plot_cumulative_distribution(Data, 10212, 'Maze 2', loc)

plot_cumulative_distribution(Data, 10224, 'Open Field', loc)
plot_cumulative_distribution(Data, 10224, 'Maze 1', loc)
plot_cumulative_distribution(Data, 10224, 'Maze 2', loc)

plot_cumulative_distribution(Data, 10227, 'Open Field', loc)
plot_cumulative_distribution(Data, 10227, 'Maze 1', loc)
plot_cumulative_distribution(Data, 10227, 'Maze 2', loc)

plot_cumulative_distribution(Data, 11095, 'Open Field', loc)
plot_cumulative_distribution(Data, 11095, 'Maze 1', loc)
plot_cumulative_distribution(Data, 11095, 'Maze 2', loc)

plot_cumulative_distribution(Data, 11092, 'Open Field', loc)
plot_cumulative_distribution(Data, 11092, 'Maze 1', loc)
plot_cumulative_distribution(Data, 11092, 'Maze 2', loc)