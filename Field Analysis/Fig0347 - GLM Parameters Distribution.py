from mylib.statistic_test import *

code_id = '0347 - GLM Parameters Distribution'
loc = os.path.join(figpath, code_id)
mkdir(loc)


if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = {
        "Behavior Progress": [],
        "Session ID": [],
        ""
    }
