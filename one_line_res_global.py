import time,re,os,math, sys
import pandas as pd
import numpy as np

phase_1_loandra_res = 'phase_1_out_print.txt'
phase_2_loandra_res = 'phase_2_out_print.txt'

solver_time_r = r'\*solver\* -time: (\d+\.\d*)'
clg_time_r = r'-- SUM CL TIME: (\d+\.\d*)'
sum_time_r = r'SUM TIME: (\d+\.\d*)'
loandra_status_r = r'\*loandra status:\s*(\w+)'
# 
ml_sat_unsat_r = r'-ml sat and violated: (\d+) \| (\d+)'
cl_sat_unsat_r = r'-cl sat and violated: (\d+) \| (\d+)'
# 
b0_r = r'b0_final: (\d+)'
b1_r = r'b1_final: (\d+)'
ARI_r = r'ARI:\s*(-?\d+\.\d*)'

def get_var(pattern, txt, group_ind):
    try:
        return re.compile(pattern).search(txt).group(group_ind)
    except:
        return ''


solution_path = sys.argv[1]
file_name_pattern_b1 = r'out_(\d+)_b1$'


for root, folders, files in os.walk(solution_path):
        for folder in folders:
            folder_path = os.path.join(root, folder)+'/'
            with open(folder_path + phase_1_loandra_res, 'r+') as f:
                line = ''.join([i for i in f])
                p1_solver_time = get_var(solver_time_r, line, 1)
                p1_clg_time = get_var(clg_time_r, line, 1)
                p1_sum_time = get_var(sum_time_r, line, 1)
                ml_sat = get_var(ml_sat_unsat_r, line, 1)
                ml_unsat = get_var(ml_sat_unsat_r, line, 2)
                cl_sat = get_var(cl_sat_unsat_r, line, 1)
                cl_unsat = get_var(cl_sat_unsat_r, line, 2)
                p1_loandra_status = get_var(loandra_status_r, line, 1)
                
                # print(p1_solver_time,p1_clg_time,p1_sum_time,ml_sat,ml_unsat,cl_sat,cl_unsat)


            for filename in os.listdir(folder_path):
                match_b1 = re.match(file_name_pattern_b1, filename)
                if match_b1:
                # if filename.startswith(file_name_pattern): # and filename[4:].isdigit()
                    with open(folder_path + filename, 'r+') as f:
                        line = ''.join([i for i in f])
                        p2_solver_time = get_var(solver_time_r, line, 1)
                        p2_clg_time = get_var(clg_time_r, line, 1)
                        p2_sum_time = get_var(sum_time_r, line, 1)
                        b0 = get_var(b0_r, line, 1)
                        b1 = get_var(b1_r, line, 1)
                        ARI = get_var(ARI_r, line, 1)
                        p2_loandra_status = get_var(loandra_status_r, line, 1)
                        
                        # print(p2_solver_time,p2_clg_time,p2_sum_time,b0,b1,ARI)

                    file_name_pattern_b0 = rf'out_{match_b1.group(1)}_b0$'
                    for filename in os.listdir(folder_path):
                        if re.match(file_name_pattern_b0, filename):
                            with open(folder_path + filename, 'r+') as f:
                                line = ''.join([i for i in f])
                                b0_solver_time = get_var(solver_time_r, line, 1)
                                b0_clg_time = get_var(clg_time_r, line, 1)
                                b0_sum_time = get_var(sum_time_r, line, 1)

                    sol_folder_path = folder_path
                    data, epsilon, kappa, seed, cl_ml_ratio = [re.sub(r'^(mc|s|r)(\d+)', r'\2', i) for i in sol_folder_path.strip(""" ./""").split('/')[-1].split('_')]

                    # b0_sum_time = '0'#b0_sum_time if b0_sum_time!= '' else '0'

                    one_line_res = ','.join([data, seed, epsilon, kappa,cl_ml_ratio,\
                                             match_b1.group(1), #iter digit 
                                            ml_sat,ml_unsat,cl_sat,cl_unsat,\
                                            b0,b1,ARI,\
                                            p1_loandra_status, p1_clg_time, p1_solver_time, p1_sum_time,\
                                            p2_loandra_status, p2_clg_time, p2_solver_time, p2_sum_time,\
                                            b0_sum_time,\
                                            str(float(b0_sum_time)+float(p1_sum_time) + float(p2_sum_time))])

                    # print('\n\n\n')
                    print(one_line_res)