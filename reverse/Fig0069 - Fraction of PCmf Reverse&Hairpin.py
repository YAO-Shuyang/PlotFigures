from mylib.statistic_test import *

code_id = '0069 - Fraction of PCmf Reverse&Hairpin'
loc = os.path.join(figpath, code_id)
mkdir(loc)

if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Fraction', 'Direction', 'Criteria'], 
                              f = f3, function = FractionOfPCmf_Reverse_Interface, 
                              file_name = code_id, behavior_paradigm = 'ReverseMaze')
    
if os.path.exists(join(figdata, code_id+' [Hairpin].pkl')):
    with open(join(figdata, code_id+' [Hairpin].pkl'), 'rb') as handle:
        HPData = pickle.load(handle)
else:
    HPData = DataFrameEstablish(variable_names = ['Fraction', 'Direction', 'Criteria'], 
                              f = f4, function = FractionOfPCmf_Reverse_Interface, 
                              file_name = code_id + ' [Hairpin]', behavior_paradigm = 'HairpinMaze')
    
if os.path.exists(join(figdata, code_id+' [Field Num].pkl')):
    with open(join(figdata, code_id+' [Field Num].pkl'), 'rb') as handle:
        FN_Data = pickle.load(handle)
else:
    FN_Data = DataFrameEstablish(variable_names = ['Field Number', 'Direction', 'Criteria'], 
                                 f = f3, function = AverageFieldNumber_Reverse_Interface, 
                                 file_name = code_id+' [Field Num]', 
                                 behavior_paradigm = 'ReverseMaze')
    
if os.path.exists(join(figdata, code_id+' [Hairpin [Field Num]].pkl')):
    with open(join(figdata, code_id+' [Hairpin] [Field Num].pkl'), 'rb') as handle:
        FN_HPData = pickle.load(handle)
else:
    FN_HPData = DataFrameEstablish(variable_names = ['Field Number', 'Direction', 'Criteria'], 
                                   f = f4, function = AverageFieldNumber_Reverse_Interface, 
                                   file_name = code_id + ' [Hairpin] [Field Num]', 
                                   behavior_paradigm = 'HairpinMaze')
    
print_estimator(FN_Data['Field Number'][np.where((FN_Data['Direction'] == 'cis')&(FN_Data['Criteria'] == 'loose'))[0]])
print_estimator(FN_Data['Field Number'][np.where((FN_Data['Direction'] == 'trs')&(FN_Data['Criteria'] == 'loose'))[0]])
print_estimator(FN_Data['Field Number'][np.where((FN_Data['Direction'] == 'cis')&(FN_Data['Criteria'] == 'rigorous'))[0]])
print_estimator(FN_Data['Field Number'][np.where((FN_Data['Direction'] == 'trs')&(FN_Data['Criteria'] == 'rigorous'))[0]], end='\n\n')

print_estimator(FN_HPData['Field Number'][np.where((FN_HPData['Direction'] == 'cis')&(FN_HPData['Criteria'] == 'loose'))[0]])
print_estimator(FN_HPData['Field Number'][np.where((FN_HPData['Direction'] == 'trs')&(FN_HPData['Criteria'] == 'loose'))[0]])
print_estimator(FN_HPData['Field Number'][np.where((FN_HPData['Direction'] == 'cis')&(FN_HPData['Criteria'] == 'rigorous'))[0]])
print_estimator(FN_HPData['Field Number'][np.where((FN_HPData['Direction'] == 'trs')&(FN_HPData['Criteria'] == 'rigorous'))[0]], end='\n\n')
    
print_estimator(Data['Fraction'][np.where((Data['Direction'] == 'cis')&(Data['Criteria'] == 'loose'))[0]])
print_estimator(Data['Fraction'][np.where((Data['Direction'] == 'trs')&(Data['Criteria'] == 'loose'))[0]])
print_estimator(Data['Fraction'][np.where((Data['Direction'] == 'cis')&(Data['Criteria'] == 'rigorous'))[0]])
print_estimator(Data['Fraction'][np.where((Data['Direction'] == 'trs')&(Data['Criteria'] == 'rigorous'))[0]], end='\n\n')

print_estimator(HPData['Fraction'][np.where((HPData['Direction'] == 'cis')&(HPData['Criteria'] == 'loose'))[0]])
print_estimator(HPData['Fraction'][np.where((HPData['Direction'] == 'trs')&(HPData['Criteria'] == 'loose'))[0]])
print_estimator(HPData['Fraction'][np.where((HPData['Direction'] == 'cis')&(HPData['Criteria'] == 'rigorous'))[0]])
print_estimator(HPData['Fraction'][np.where((HPData['Direction'] == 'trs')&(HPData['Criteria'] == 'rigorous'))[0]], end='\n\n')