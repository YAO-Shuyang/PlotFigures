from mylib.statistic_test import *

code_id = "0836 - Modulated Proportion along Main Track"
loc = join(figpath, "Dsp", code_id)
mkdir(loc)

if exists(join(figdata, code_id+'.pkl')) == False:
    DataFrameEstablish(variable_names=["Position", "Type", "Proportion"], f=f2, file_name=code_id, 
                       behavior_paradigm="DSPMaze", function=ModulatedProportionSpatialDistribution_DSP_Interface)