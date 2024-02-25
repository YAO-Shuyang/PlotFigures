from mylib.statistic_test import *

code_id = '0321 - Fraction of Superstable Fields changes over time'
loc = join(figpath, code_id)
mkdir(loc)

f = pd.read_excel(join(figdata, "0300 - Kinetic Model Simulation.xlsx"))

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = [
                        'Duration', 'Superstable Frac.', 'Threshold', 'Drift Model'], 
                        f = f, 
                        function = Superstable_Fraction_Interface, 
                        file_name = code_id, 
                        behavior_paradigm = 'CrossMaze')

idx = np.where(Data['Drift Model'] == 'converged')[0]
SubData = SubDict(Data, Data.keys(), idx)
sns.lineplot(
    x='Duration', 
    y='Superstable Frac.',
    data=SubData,
    hue='Threshold',
)
plt.show()

