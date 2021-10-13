# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:40:21 2021

@author: mikki
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import json
import textwrap
from textwrap import wrap
import math
plt.rcParams.update({'figure.max_open_warning': 0})

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

save_dir="H:/"
name_str="scores_checkpoint-"

Kitti_file="H:/Prednet_kitti/kitti_results/scores_checkpoint-080-loss0.00591.json"
smth_norm_file="H:/PredNet_Critic/fps10_Simulations/Normal/Results/scores_checkpoint-117-loss0.01701.json"
smth_dyn_file="H:/PredNet_Critic/fps10_Simulations/Dynamic_fps/Results/scores_checkpoint-047-loss0.01884.json"
files_list=[Kitti_file, smth_norm_file, smth_dyn_file]

# Conditioned_SSIM_mean, MAE_mean, MAE_mean_prev_frame_copy, MAE_std, MAE_std_prev_frame_copy=([] for i in range(5))
# MSE_mean, MSE_mean_prev_frame_copy, MSE_std, MSE_std_prev_frame_copy, PSNR_mean=([] for i in range(5))
# PSNR_mean_prev_frame_copy, SSIM_mean, SSIM_mean_prev_frame_copy, Sharpness_difference_mean, Sharpness_grad_mean=([] for i in range(5))
#Val_loss, epochs=([] for i in range(2))

fig, ax=plt.subplots(5,3, figsize=(30,30))
fig.suptitle("SMTH Vs. KITTI Results_Dict_PLot for 10-FPS(Hz)", fontsize=35, fontweight='bold')
datasets=['SMTH_Norm.', 'KITTI', 'SMTH_Dyn.']

f= open(Kitti_file)
KITTI_json=json.load(f)
f= open(smth_norm_file)
smth_norm_json=json.load(f)
f= open(smth_dyn_file)
smth_dyn_json=json.load(f)
i=0
j=0
for k in list(KITTI_json.keys()):
    if math.isnan(float(KITTI_json.get(k))):
        KITTI_json.pop(k)
        smth_norm_json.pop(k)
        smth_dyn_json.pop(k)
keys=list(KITTI_json.keys())
k=0
for i in range(5):
    for j in range(3):
       if k<len(keys):
          print("Creating subplot for "+str(i)+" and "+str(j))
          data_list=[]
          data_list.append(smth_norm_json.get(keys[k]))
          data_list.append(KITTI_json.get(keys[k]))
          data_list.append(smth_dyn_json.get(keys[k]))
          ax[i, j].bar(datasets, data_list, width=[0.3,0.3,0.3])
          ax[i, j].set_title(str(i)+"x"+str(j), fontsize='large' ,ha='center')
          ax[i, j].grid(True)
          s=''
          count=0
          new_count=0
          l=keys[k]. split("_")
          for l1 in l:
            if count<2:
                s+=l1+" "
                count+=1
            elif new_count>0:
                s+=l1+" "
            elif count==2:
                s+=l1+"\n"
                new_count+=1
          ax[i, j].set_ylabel(s,wrap=True,fontsize=26)
          k+=1
fig.subplots_adjust(hspace=0.5, wspace=0.5)
fig.savefig("H:/SMTH_Vs._KITTI Results_Dict_PLot_for_10-FPS(Hz).png", dpi=1000)   #, bbox_inches='tight'   
fig.show()
