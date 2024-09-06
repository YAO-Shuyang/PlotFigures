from mylib.statistic_test import *

code_id = '0819 - Latent Space Orthogonality'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)


if os.path.exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['Segment', 
                                                'Smallest Singlar Value', 
                                                'Conditional Number', 
                                                'Subspace Comparison Type',
                                                'Shinkage'
                                                ], f = f2,
                              function = SubspacesOrthogonality_DSP_Interface, 
                              file_name = code_id+'', behavior_paradigm = 'DSPMaze')
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)




for shrinkage in np.unique(Data['Shinkage']):
    fig = plt.figure(figsize = (4, 3))
    ax = Clear_Axes(plt.axes(), close_spines = ['top', 'right'], ifxticks = True, ifyticks = True)
    idx = np.where(Data['Shinkage'] == shrinkage)[0]
    SubData = SubDict(Data, Data.keys(), idx = idx)
    sns.barplot(
        x = 'Segment',
        y = 'Smallest Singlar Value',
        hue = 'Subspace Comparison Type',
        data = SubData,
        palette = sns.color_palette("rainbow", 7)[1:],
        ax = ax,
        width=0.8,
        capsize=0.1,
        errcolor='black',
        errwidth=0.5,
    )
    plt.savefig(os.path.join(loc, 'Smallest Singlar Value [Shinkage = '+str(shrinkage)+'].svg'))
    plt.savefig(os.path.join(loc, 'Smallest Singlar Value [Shinkage = '+str(shrinkage)+'].png'), dpi=600)
    plt.close()
    
    fig = plt.figure(figsize = (4, 3))
    ax = Clear_Axes(plt.axes(), close_spines = ['top', 'right'], ifxticks = True, ifyticks = True)
    sns.barplot(
        x = 'Segment',
        y = 'Conditional Number',
        hue = 'Subspace Comparison Type',
        data = SubData,
        palette = sns.color_palette("rainbow", 7)[1:],
        ax = ax,
        width=0.8,
        capsize=0.1,
        errcolor='black',
        errwidth=0.5,
    )
    plt.savefig(os.path.join(loc, 'Conditional Number [Shinkage = '+str(shrinkage)+'].svg'))
    plt.savefig(os.path.join(loc, 'Conditional Number [Shinkage = '+str(shrinkage)+'].png'), dpi=600)
    plt.close()