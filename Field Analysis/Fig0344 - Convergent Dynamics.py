from mylib.statistic_test import *

code_id = '0344 - Convergent Dynamics'
loc = os.path.join(figpath, code_id)
mkdir(loc)

with open(r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\test\demo_seq.pkl", 'rb') as handle:
    template_seq = pickle.load(handle)

from mylib.model import ProbabilityRNN, ContinuousHiddenStateModel
from mylib.model import FeatureKnockOut

M53 = ContinuousHiddenStateModel('poly2')
M53.fit(template_seq)

def func(n_step=200, n_fields=10000, is_noise=True):
    simu_reg, simu_identity = M53.simulate_across_day(n_step=n_step, n_fields=n_fields, is_noise=is_noise)

    predicted_prob = np.zeros((simu_reg.shape[0], simu_reg.shape[1]-1)) * np.nan

    for i in range(simu_reg.shape[0]):
        idx = np.where(simu_reg[i, :] == 1)[0][0]
        predicted_prob[i, idx:] = M53.get_predicted_prob([simu_reg[i, idx:]], is_noise=True)[0]

    predicted_prob *= simu_identity[:, 1:]
    
    return predicted_prob, simu_reg, simu_identity

with open(join(figdata, code_id, "SFER General [1000 Sessions].pkl"), 'rb') as handle:
    predicted_prob, simu_reg, simu_identity = pickle.load(handle)

mat = np.zeros((40, 998), np.float64)
for i in range(998):
    mat[:, i] = np.histogram(
        predicted_prob[np.where(np.isnan(predicted_prob[:, i]) == False)[0], i],
        range=(0, 1),
        bins=40,
        density=True
    )[0]

mat_n = mat / np.max(mat, axis = 0)
fig = plt.figure(figsize = (8, 2))
ax = Clear_Axes(plt.axes(), ifxticks=True, ifyticks=True)
im = ax.imshow(mat_n[:, :200], vmin = 0, vmax = 1, aspect='auto', interpolation=None)
plt.colorbar(im, ax = ax)
plt.savefig(join(loc, "Pt Distribution Heatmap.svg"), dpi = 2400)
plt.savefig(join(loc, "Pt Distribution Heatmap.png"), dpi = 2400)
plt.show()

# Proportion of New Field
# Proportion of Permanent Silent Field
# Total Field Number

num = np.nancumsum(simu_reg[:, :20], axis=1)
plt.hist(num[:, -1], range=(0.5, 20.5), bins=20)
plt.show()

fig = plt.figure(figsize=(4,3))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)

NewField = np.zeros(predicted_prob.shape[1])
for i in range(predicted_prob.shape[1]):
    idx = np.where((simu_identity[:, i+1] == 1) & (np.isnan(simu_identity[:, i]) == True))[0]
    NewField[i] = idx.shape[0]

PermanentSilent = np.zeros(predicted_prob.shape[1])
for i in range(predicted_prob.shape[1]):
    idx = np.where((np.nansum(simu_identity[:, :i+1], axis=1) >= 1) & (np.isnan(simu_identity[:, i+1]) == True))[0]
    PermanentSilent[i] = idx.shape[0]

TotalFieldNumber = np.zeros(predicted_prob.shape[1])
for i in range(predicted_prob.shape[1]):
    idx = np.where(np.isnan(np.nansum(simu_reg[:, :i+1], axis=1)) >1)[0]
    TotalFieldNumber[i] = idx.shape[0]
ax.plot(NewField, linewidth=0.5)
ax.plot(PermanentSilent, linewidth=0.5)
ax.plot(TotalFieldNumber, linewidth=0.5)
plt.show()