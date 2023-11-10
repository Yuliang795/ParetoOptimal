import os,sys,re

def get_res(folder_path,folder_name):
  with open(folder_path +'one_line_res.txt', 'r+') as f:
    lines = f.readlines()
  one_line_res = lines[-1].rstrip('\n')
  print(one_line_res)

dir_path = r'./solutions/' # solutions

# list to store files
file_list = []
# Iterate directory
for path in os.listdir(dir_path):
#   print(path, os.path.join(dir_path, path))
  if os.path.isdir(os.path.join(dir_path, path)) and\
   os.path.exists(os.path.join(dir_path, path)+'/one_line_res.txt'):
      file_list.append(path)
      get_res(os.path.join(dir_path, path)+'/',path)
      # print(111,os.path.join(dir_path, path))