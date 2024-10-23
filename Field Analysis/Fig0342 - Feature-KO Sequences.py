from mylib.statistic_test import *

code_id = '0342 - Feature-KO Sequences'
loc = os.path.join(figpath, code_id)
mkdir(loc)

with open(r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\test\demo_seq.pkl", 'rb') as handle:
    template_seq = pickle.load(handle)

from mylib.model import ProbabilityRNN, ContinuousHiddenStateModel
from mylib.model import FeatureKnockOut

M53 = ContinuousHiddenStateModel('poly2')
M53.fit(template_seq)

def counts(model: ProbabilityRNN, sequences: list[np.ndarray]):
    seq = np.concatenate([i[1:] for i in sequences])
    n_one = np.sum(seq == 1)
    n_zero = np.sum(seq == 0)
    
    n_one_violate, n_zero_violate = 0, 0
    predicted_prob = model.get_predicted_prob(sequences)
    
    for i in range(len(predicted_prob)):
        prob = np.ediff1d(predicted_prob[i])
        is_correct = (sequences[i][1:-1]-0.5) * prob > 0
        
        n_one_violate += np.sum((is_correct == False) & (sequences[i][1:-1] == 1))
        n_zero_violate += np.sum((is_correct == False) & (sequences[i][1:-1] == 0))
            
    return n_one, n_zero, n_one_violate, n_zero_violate

if exists(os.path.join(figdata, code_id+'.pkl')) == False:
    Data = {"X Percentage": [], "Hidden Size": [], "Y Percentage": []}
    
    for x_perc in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        for hidden_size in [8]:
            for i in range(1):
                print(f" {x_perc}% Feature-KO, Hidden Size: {hidden_size}, Validation {i}")
                feature_ko = FeatureKnockOut(p=x_perc/100, p0=0.5)
                simu_seq = feature_ko.simulate(template_seq)
                print(np.sum(np.concatenate(simu_seq)), np.concatenate(simu_seq).shape)
                
                model = ProbabilityRNN.process_fit(
                    sequences=simu_seq,
                    split_ratio=0.8,
                    hidden_size=hidden_size,
                    lr=0.001,
                    epochs=100, 
                    batch_size=2048
                )
                
                predicted_prob = model.get_predicted_prob(simu_seq)
                p_one_x, p_one_y = [], []
                p_zero_x, p_zero_y = [], []

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
                plt.show()
                
                for i in range(len(predicted_prob)):
                    one_idx = np.where(template_seq[i][1:-1] == 1)[0]
                    zero_idx = np.where(template_seq[i][1:-1] == 0)[0]
    
                    p_one_x.append(predicted_prob[i][one_idx])
                    p_one_y.append(predicted_prob[i][one_idx+1])
    
                    p_zero_x.append(predicted_prob[i][zero_idx])
                    p_zero_y.append(predicted_prob[i][zero_idx+1])
                
                n_one, n_zero, n_one_violate, n_zero_violate = counts(model, simu_seq)
                y_perc = (n_one_violate + n_zero_violate) / (n_one + n_zero)
                print(y_perc)
                Data["X Percentage"].append(x_perc)
                Data["Hidden Size"].append(hidden_size)
                Data["Y Percentage"].append(y_perc)
                print()

    D = pd.DataFrame(Data)
    D.to_excel(join(figdata, code_id+'.xlsx'), index=False)
    
    #with open(join(figdata, code_id+'.pkl'), 'wb') as handle:
    #    pickle.dump(Data, handle)
else:
    with open(os.path.join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)