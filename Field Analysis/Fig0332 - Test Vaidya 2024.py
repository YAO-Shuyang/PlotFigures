from mylib.statistic_test import *

code_id = '0832 - Test Vaidya 2024'
loc = os.path.join(figpath, 'Dsp', code_id)
mkdir(loc)

with open(r"E:\Data\FigData\PermenantFieldAnalysis\mouse_1_converged_10000fields_50days.pkl", 'rb') as handle:
    Data = pickle.load(handle)

field_reg = Data['field_reg'].T

sum = np.sum(field_reg[:13, :], axis=0)

plt.hist(sum, range=(-0.5, 13.5), bins=14, rwidth=0.8, color='grey', density=True, alpha=0.5)


with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10224\Maze1-footprint\trace_mdays_conc.pkl", 'rb') as handle:
    trace = pickle.load(handle)

field_reg = trace['field_reg']

idx = np.where(np.isnan(np.sum(field_reg[:13, :], axis=0)) == False)[0]
sum = np.sum(field_reg[:13, idx], axis=0)

with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10227\Maze1-footprint\trace_mdays_conc.pkl", 'rb') as handle:
    trace = pickle.load(handle)

field_reg = trace['field_reg']

idx = np.where(np.isnan(np.sum(field_reg[:13, :], axis=0)) == False)[0]
sum2 = np.sum(field_reg[:13, idx], axis=0)

sum = np.concatenate([sum, sum2])

with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10209\Maze1-2-footprint\trace_mdays_conc.pkl", 'rb') as handle:
    trace = pickle.load(handle)

field_reg = trace['field_reg']

idx = np.where(np.isnan(np.sum(field_reg[:13, :], axis=0)) == False)[0]
sum2 = np.sum(field_reg[:13, idx], axis=0)

sum = np.concatenate([sum, sum2])

with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10212\Maze1-2-footprint\trace_mdays_conc.pkl", 'rb') as handle:
    trace = pickle.load(handle)

field_reg = trace['field_reg']

idx = np.where(np.isnan(np.sum(field_reg[:13, :], axis=0)) == False)[0]
sum2 = np.sum(field_reg[:13, idx], axis=0)

sum = np.concatenate([sum, sum2])

plt.hist(sum, range=(-0.5, 13.5), bins=14, rwidth=0.8, density=True, alpha=0.5)
plt.show()