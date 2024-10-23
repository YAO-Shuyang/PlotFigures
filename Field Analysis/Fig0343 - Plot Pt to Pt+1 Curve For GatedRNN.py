from mylib.statistic_test import *

from mylib.model import ProbabilityRNN

code_id = '0343 - Plot Pt to Pt+1 Curve For GatedRNN'
loc = os.path.join(figpath, code_id)
mkdir(loc)

with open(r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\test\demo_seq.pkl", 'rb') as handle:
    template_seq = pickle.load(handle)

from mylib.model import ProbabilityRNN, ContinuousHiddenStateModel
from mylib.model import FeatureKnockOut

model = ProbabilityRNN.process_fit(
    template_seq,
    hidden_size=16,
    lr=0.001,
    epochs=1000, 
    batch_size=2048
)

predicted_prob = model.get_predicted_prob(template_seq)
p_one_x, p_one_y = [], []
p_zero_x, p_zero_y = [], []

for i in range(len(predicted_prob)):
    one_idx = np.where(template_seq[i][1:-1] == 1)[0]
    zero_idx = np.where(template_seq[i][1:-1] == 0)[0]
    
    p_one_x.append(predicted_prob[i][one_idx])
    p_one_y.append(predicted_prob[i][one_idx+1])
    
    p_zero_x.append(predicted_prob[i][zero_idx])
    p_zero_y.append(predicted_prob[i][zero_idx+1])

p_one_x = np.concatenate(p_one_x)
p_one_y = np.concatenate(p_one_y)

p_zero_x = np.concatenate(p_zero_x)
p_zero_y = np.concatenate(p_zero_y)

one_idx = np.random.choice(len(p_one_x), 1000, replace = False)
zero_idx = np.random.choice(len(p_zero_x), 1000, replace = False)

fig = plt.figure(figsize = (4, 4))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks = True, ifyticks = True)
ax.plot(p_one_x[one_idx], p_one_y[one_idx], 'o', color="#CFDFEE", markersize = 4, markeredgewidth = 0, alpha=0.8)
ax.plot(p_zero_x[zero_idx], p_zero_y[zero_idx], 'o', color="#FEE0CF", markersize = 4, markeredgewidth = 0, alpha=0.8)
ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
plt.savefig(join(loc, f'model_predicted_p.png'), dpi=600)
plt.savefig(join(loc, f'model_predicted_p.svg'), dpi=600)
plt.show()