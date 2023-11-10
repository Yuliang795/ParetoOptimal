import re,os


file_name_pattern = 'out_\d+'
file_name_pattern = rf'{file_name_pattern}'
tmp_solution_path = './solutions/chainlink_0.1_mc0.5_s1732_r1' 
for filename in os.listdir(tmp_solution_path):
    if re.match(file_name_pattern, filename):
        print(filename)