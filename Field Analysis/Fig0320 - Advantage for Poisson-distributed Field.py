import numpy as np
from scipy.stats import poisson, nbinom, norm, expon
from mylib.statistic_test import *

code_id = '0320 - Advantage for Poisson-distributed Field'
loc = join(figpath, code_id)
mkdir(loc)

# M place fields with N neurons
M = 10000
N = 2000

L = np.arange(10, 10001, 5)

# Poisson-distributed Field Number + Spatially Random
def get_field_number(mode: str = "Poisson", params: list = [], N: int = 2000):
    if mode == "Poisson":
        return poisson.rvs(params[0], size = N)
    elif mode == "NBinom":
        return nbinom.rvs(params[0], params[1], size = N)
    elif mode == "Norm":
        num = (norm.rvs(params[0], params[1], size = N) // 1).astype(int)
        num[num < 0] = 0
        return num
    else:
        raise ValueError
    
def spatial_distributed_fields(mode: str = "Poisson", field_num: np.ndarray = None, L: int = 10, N: int = 2000):
    MAT = np.zeros((N, L), int)
    assert len(field_num) == N
    
    if mode == "Poisson":
        for i in range(N):
            if field_num[i] == 0:
                continue
            IFI = (np.random.exponential(scale=L/field_num[i], size=field_num[i])//1).astype(int)
            cum_IFI = np.cumsum(IFI)
            while cum_IFI[-1] >= L:
                IFI = (np.random.exponential(scale=L/field_num[i], size=field_num[i])//1).astype(int)
                cum_IFI = np.cumsum(IFI)
            
            MAT[i, IFI] = 1

    elif mode == "Random":
        for i in range(N):
            MAT[i, np.random.choice(np.arange(L), size=field_num[i])] = 1
    else:
        raise ValueError("Mode should be 'Poisson' or 'Random'!")
    
    return MAT

def analyze_spatial_map(MAT: np.ndarray, L: int = 10):
    non_coding_frac = np.where(np.sum(MAT, axis=0) == 0)[0].shape[0] / L # Number of null vectors.
    uniq_coding_frac = (np.unique(MAT, axis=1).shape[1]-1) / L # The rank of MAT is equal to the number of unique population vectors.
    
    return non_coding_frac, uniq_coding_frac


def main(
    num_modeL: str, 
    display_mode: str, 
    params: list, 
    L: np.ndarray, 
    N_SIMU: int = 10, 
    N: int = 2000
) -> dict:
    NF, UF = np.zeros((N_SIMU, L.shape[0])), np.zeros((N_SIMU, L.shape[0]))
    field_num = np.zeros((N_SIMU, N), int)
    
    for i in range(N_SIMU):
        print("    Random trial: ", i)
        field_num[i, :] = get_field_number(mode=num_modeL, params=params, N=N)
        for l in tqdm(range(L.shape[0])):
            MAT = spatial_distributed_fields(mode=display_mode, field_num=field_num[i, :], L=L[l], N=N)
            NF[i, l], UF[i, l] = analyze_spatial_map(MAT, L=L[l])
            
    return {
        'L': L,
        'Simu times': N_SIMU,
        'Num Mode': num_modeL,
        'Field Num': field_num,
        'Display Mode': display_mode,
        'params': params,
        'NF': NF,
        'UF': UF
    }

if __name__ == "__main__":
    import pickle
    import os
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import pandas as pd

    sheet = pd.read_excel(join(figdata, code_id+'.xlsx'))
    Data = {}
    N_SIMU = 10
    for i in range(len(sheet)):
        print(sheet['TestID'][i], sheet['Num Mode'][i], sheet['Display Mode'][i], sheet['Note'][i], " -------------------------------")
        if sheet['Num Mode'][i] == 'Poisson':
            params = [sheet['lam'][i]]
        elif sheet['Num Mode'][i] == 'NBinom':
            params = [sheet['r'][i], sheet['p'][i]]
        elif sheet['Num Mode'][i] == 'Norm':
            params = [sheet['mu'][i], sheet['sigma'][i]]
        else:
            raise ValueError
        
        Data[sheet['TestID'][i]] = main(
            num_modeL=sheet['Num Mode'][i], 
            display_mode=sheet['Display Mode'][i], 
            params=params, L=L, N_SIMU=N_SIMU, N = sheet['n_neuron'][i]
        )
        print()
        
    with open(join(loc, code_id+'.pkl'), 'wb') as f:
        pickle.dump(Data, f)
        
