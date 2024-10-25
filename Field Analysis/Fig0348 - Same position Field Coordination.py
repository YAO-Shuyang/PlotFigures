#from mylib.statistic_test import *
#from mylib.field.tracker_v2 import Tracker2d
#from mylib.field.sfer import get_surface, get_data, fit_kww, fit_reci
import os
import h5py
import numpy as np
import pickle

def get_roi_center_dist_matrix(dir_name: str):
    files = os.listdir(dir_name)
    
    dates = []
    for file in files:
        if 'SFP' in file:
            print(file, int(file[3:11]))
            dates.append(int(file[3:11]))
    
    dates = np.array(dates)
    idx = np.argsort(dates)
    
    sfps = []
    for i in idx:
        path = os.path.join(dir_name, f"SFP{dates[i]}.mat")
        with h5py.File(path, 'r') as handle:
            sfp = np.array(handle['SFP'])
            print(sfp.shape)
            sfps.append(sfp)
    
    


if __name__ == '__main__':
    get_roi_center_dist_matrix(r"E:\Data\Cross_maze\10209\Maze1-2-footprint")