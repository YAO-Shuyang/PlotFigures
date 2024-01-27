from mylib.statistic_test import *

code_id = "0020 - Influence of Field Criteria to Poisson Distribution"
loc = os.path.join(figpath, code_id)
mkdir(loc)

# Test data against possion distribution
if  os.path.exists(os.path.join(figdata,code_id+'.pkl')) == False:
    Data = DataFrameEstablish(variable_names = ['lambda', 'Poisson-residual', 'Poisson-ChiS Statistics', 'Poisson-ChiS P-Value',
                                                'Poisson-KS Statistics', 'Poisson-KS P-Value', 
                                                'Poisson-KS 2sample Statistics', 'Poisson-KS 2sample P-Value',
                                                'Poisson-AD 2sample Statistics', 'Poisson-AD 2sample P-Value', 
                                                'Poisson-Is Rejected',
                                                
                                                'Mean', 'Sigma', 'Normal-residual', 'Normal-ChiS Statistics', 'Normal-ChiS P-Value',
                                                'Normal-KS Statistics', 'Normal-KS P-Value', 
                                                'Normal-KS 2sample Statistics', 'Normal-KS 2sample P-Value',
                                                'Normal-AD 2sample Statistics', 'Normal-AD 2sample P-Value', 
                                                'Normal-Is Rejected',
                                                
                                                'r', 'p', 'NBinom-residual', 'NBinom-ChiS Statistics', 'NBinom-ChiS P-Value',
                                                'NBinom-KS Statistics', 'NBinom-KS P-Value', 
                                                'NBinom-KS 2sample Statistics', 'NBinom-KS 2sample P-Value',
                                                'NBinom-AD 2sample Statistics', 'NBinom-AD 2sample P-Value', 
                                                'NBinom-Is Rejected',
                                                
                                                'Events Threshold', 'Field Number'], f = f1, pass_i = True,
                              function = FieldDistributionStatistics_DiverseCriteria_Interface, 
                              file_name = code_id, behavior_paradigm = 'HairpinMaze')
else:
    with open(os.path.join(figdata,code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)