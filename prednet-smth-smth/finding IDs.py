# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:09:32 2021

@author: mikki
"""

import pandas as pd
import numpy as np
import glob, os 
import itertools

path_str="H:\PredNet_Critic"
files= glob.glob(path_str+"\*\*")
name_str="Test_data_for_plt_used_for_checkpoint-"
ids_list=[]
for path, subdirs, files in os.walk(path_str):
   for name in files:
       file=os.path.join(path, name)
       file=file.replace((os.sep), "/")
       if name_str in file.split("/")[-1]:
            data=pd.read_csv(file)
            ids_list.append(data.loc[:, "id"].tolist())

ids_list.sort()
new_num = list(num for num,_ in itertools.groupby(ids_list))

flat_list = []
for sublist in new_num:
    for item in sublist:
        flat_list.append(item)

unique=[]
non_unique=[]
for item in flat_list:
    if item in unique:
        non_unique.append(item)
    else:
        unique.append(item)
print(non_unique)
count=0
for ids in ids_list:
    if non_unique==ids:
        print(count)
    count+=1