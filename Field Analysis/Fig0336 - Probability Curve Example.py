from mylib.statistic_test import *

code_id = '0336 - Probability Curve Example'    
loc = join(figpath, code_id)
mkdir(loc)

# Load sequences from the pickle file
with open(r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\test\demo_seq.pkl", 'rb') as handle:
    sequences = pickle.load(handle)

exp_seq = np.array([1,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1])
print(len(exp_seq))

from mylib.model import EqualRateDriftModel
from mylib.model import TwoProbabilityIndependentModel
from mylib.model import JointProbabilityModel
from mylib.model import HMM
from mylib.model import ContinuousHiddenStateModel
from mylib.model import ProbabilityRNN

M1 = EqualRateDriftModel()
M1.fit(sequences)
M1.calc_loss(sequences)
y1 = M1.get_predicted_prob([exp_seq])

M2 = TwoProbabilityIndependentModel()
M2.fit(sequences)
M2.calc_loss(sequences)
y2 = M2.get_predicted_prob([exp_seq])

M3 = JointProbabilityModel()
M3.fit(sequences)
M3.calc_loss(sequences)
y3 = M3.get_predicted_prob([exp_seq])

M4 = HMM.process_fit(N=5, sequences=sequences)
M4.calc_loss(sequences)
y4 = M4.get_predicted_prob([exp_seq])

M5 = ContinuousHiddenStateModel('reci')
M5.fit(sequences)
M5.calc_loss(sequences)
y5 = M5.get_predicted_prob([exp_seq])

M6 = ProbabilityRNN.process_fit(
    sequences,
    split_ratio=0.8,
    hidden_size=32,
    lr=0.001,
    epochs=1000, 
    batch_size=2048
)
M6.calc_loss(sequences)
y6 = M6.get_predicted_prob([exp_seq])