# -*- coding: utf-8 -*-
"""
Created on Sat May  1 16:23:43 2021

@author: mikki
"""
import pandas as pd
import numpy as np
import glob, os 
import itertools
import matplotlib.pyplot as plt
import json
import textwrap
from textwrap import wrap
plt.rcParams.update({'figure.max_open_warning': 0})
import matplotlib
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

save_dir="H:\PredNet_Critic"
name_str="scores_checkpoint-"
files_list=[]
normal_epochs_list=[]
normal_val_loss_list=[]
dynamic_epochs_list=[]
dynamic_val_loss_list=[]
fps_list=[]
dynamic_plotting_list=[]
normal_plotting_list=[]

#Creating list for each individual key from Results_dict
#Conditioned_SSIM_mean
normal_Conditioned_SSIM_mean_list=[]
dynamic_Conditioned_SSIM_mean_list=[]
#MAE_mean
normal_MAE_mean_list=[]
dynamic_MAE_mean_list=[]
#MAE_mean_prev_frame_copy
normal_MAE_mean_prev_frame_copy_list=[]
dynamic_MAE_mean_prev_frame_copy_list=[]
#MAE_std
normal_MAE_std_list=[]
dynamic_MAE_std_list=[]
#MAE_std_prev_frame_copy
normal_MAE_std_prev_frame_copy_list=[]
dynamic_MAE_std_prev_frame_copy_list=[]
#MSE_mean
normal_MSE_mean_list=[]
dynamic_MSE_mean_list=[]
#MSE_mean_prev_frame_copy
normal_MSE_mean_prev_frame_copy_list=[]
dynamic_MSE_mean_prev_frame_copy_list=[]
#MSE_std
normal_MSE_std_list=[]
dynamic_MSE_std_list=[]
#MSE_std_prev_frame_copy
normal_MSE_std_prev_frame_copy_list=[]
dynamic_MSE_std_prev_frame_copy_list=[]
#PSNR_mean
normal_PSNR_mean_list=[]
dynamic_PSNR_mean_list=[]
#PSNR_mean_prev_frame_copy
normal_PSNR_mean_prev_frame_copy_list=[]
dynamic_PSNR_mean_prev_frame_copy_list=[]
#PSNR_movement_mean
normal_PSNR_movement_mean_list=[]
dynamic_PSNR_movement_mean_list=[]
#PSNR_movement_mean_prev_frame_copy
normal_PSNR_movement_mean_prev_frame_copy_list=[]
dynamic_PSNR_movement_mean_prev_frame_copy_list=[]
#SSIM_mean
normal_SSIM_mean_list=[]
dynamic_SSIM_mean_list=[]
#SSIM_mean_prev_frame_copy
normal_SSIM_mean_prev_frame_copy_list=[]
dynamic_SSIM_mean_prev_frame_copy_list=[]
#SSIM_movement_mean
normal_SSIM_movement_mean_list=[]
dynamic_SSIM_movement_mean_list=[]
#SSIM_movement_mean_prev_frame_copy
normal_SSIM_movement_mean_prev_frame_copy_list=[]
dynamic_SSIM_movement_mean_prev_frame_copy_list=[]
#Sharpness_difference_mean
normal_Sharpness_difference_mean_list=[]
dynamic_Sharpness_difference_mean_list=[]
#Sharpness_grad_mean
normal_Sharpness_grad_mean_list=[]
dynamic_Sharpness_grad_mean_list=[]

for path, subdirs, files in os.walk(save_dir):
    for name in files:
       file=os.path.join(path, name)
       file=file.replace((os.sep), "/")
       if name_str in file.split("/")[-1]:
           files_list.append(file)

_, files_sorted = zip(*sorted([(int(file.split("/")[2].replace("fps","").replace("_Simulations","")), file) for file in files_list]))
for file in files_sorted:
    #print(file)
    with open(file) as f:
        data=json.load(f)
    results_legend=list(data.keys())
    fps=int(file.split("/")[2].replace("fps","").replace("_Simulations",""))
    print(fps)
    if fps not in fps_list:
        fps_list.append(fps)
    if(file.split("/")[3]=="Dynamic_fps"):
        dynamic_epochs_list.append(int(file.split("/")[-1].split(".json")[0].split(name_str)[-1].split("-")[0]))
        dynamic_val_loss_list.append(float(file.split("/")[-1].split(".json")[0].split(name_str)[-1].split("-")[1].replace("loss","")))
        dynamic_Conditioned_SSIM_mean_list.append(float(data.get("Conditioned_SSIM_mean")))
        dynamic_MAE_mean_list.append(float(data.get("MAE_mean")))
        dynamic_MAE_mean_prev_frame_copy_list.append(float(data.get("MAE_mean_prev_frame_copy")))
        dynamic_MAE_std_list.append(float(data.get("MAE_std")))
        dynamic_MAE_std_prev_frame_copy_list.append(float(data.get("MAE_std_prev_frame_copy")))
        dynamic_MSE_mean_list.append(float(data.get("MSE_mean")))
        dynamic_MSE_mean_prev_frame_copy_list.append(float(data.get("MSE_mean_prev_frame_copy")))
        dynamic_MSE_std_list.append(float(data.get("MSE_std")))
        dynamic_MSE_std_prev_frame_copy_list.append(float(data.get("MSE_std_prev_frame_copy")))
        dynamic_PSNR_mean_list.append(float(data.get("PSNR_mean")))
        dynamic_PSNR_mean_prev_frame_copy_list.append(float(data.get("PSNR_mean_prev_frame_copy")))
        dynamic_PSNR_movement_mean_list.append(float(data.get("PSNR_movement_mean")))
        dynamic_PSNR_movement_mean_prev_frame_copy_list.append(float(data.get("PSNR_movement_mean_prev_frame_copy")))
        dynamic_SSIM_mean_list.append(float(data.get("SSIM_mean")))
        dynamic_SSIM_mean_prev_frame_copy_list.append(float(data.get("SSIM_mean_prev_frame_copy")))
        dynamic_SSIM_movement_mean_list.append(float(data.get("SSIM_movement_mean")))
        dynamic_SSIM_movement_mean_prev_frame_copy_list.append(float(data.get("SSIM_movement_mean_prev_frame_copy")))
        dynamic_Sharpness_difference_mean_list.append(float(data.get("Sharpness_difference_mean")))
        dynamic_Sharpness_grad_mean_list.append(float(data.get("Sharpness_grad_mean")))
    else:
        normal_epochs_list.append(int(file.split("/")[-1].split(".json")[0].split(name_str)[-1].split("-")[0]))
        normal_val_loss_list.append(float(file.split("/")[-1].split(".json")[0].split(name_str)[-1].split("-")[1].replace("loss","")))
        normal_Conditioned_SSIM_mean_list.append(float(data.get("Conditioned_SSIM_mean")))
        normal_MAE_mean_list.append(float(data.get("MAE_mean")))
        normal_MAE_mean_prev_frame_copy_list.append(float(data.get("MAE_mean_prev_frame_copy")))
        normal_MAE_std_list.append(float(data.get("MAE_std")))
        normal_MAE_std_prev_frame_copy_list.append(float(data.get("MAE_std_prev_frame_copy")))
        normal_MSE_mean_list.append(float(data.get("MSE_mean")))
        normal_MSE_mean_prev_frame_copy_list.append(float(data.get("MSE_mean_prev_frame_copy")))
        normal_MSE_std_list.append(float(data.get("MSE_std")))
        normal_MSE_std_prev_frame_copy_list.append(float(data.get("MSE_std_prev_frame_copy")))
        normal_PSNR_mean_list.append(float(data.get("PSNR_mean")))
        normal_PSNR_mean_prev_frame_copy_list.append(float(data.get("PSNR_mean_prev_frame_copy")))
        normal_PSNR_movement_mean_list.append(float(data.get("PSNR_movement_mean")))
        normal_PSNR_movement_mean_prev_frame_copy_list.append(float(data.get("PSNR_movement_mean_prev_frame_copy")))
        normal_SSIM_mean_list.append(float(data.get("SSIM_mean")))
        normal_SSIM_mean_prev_frame_copy_list.append(float(data.get("SSIM_mean_prev_frame_copy")))
        normal_SSIM_movement_mean_list.append(float(data.get("SSIM_movement_mean")))
        normal_SSIM_movement_mean_prev_frame_copy_list.append(float(data.get("SSIM_movement_mean_prev_frame_copy")))
        normal_Sharpness_difference_mean_list.append(float(data.get("Sharpness_difference_mean")))
        normal_Sharpness_grad_mean_list.append(float(data.get("Sharpness_grad_mean")))
#Plotting and saving results for Dynamic fps and Normal  
dynamic_plotting_list.append(dynamic_Conditioned_SSIM_mean_list)
dynamic_plotting_list.append(dynamic_MAE_mean_list)
dynamic_plotting_list.append(dynamic_MAE_mean_prev_frame_copy_list)
dynamic_plotting_list.append(dynamic_MAE_std_list)
dynamic_plotting_list.append(dynamic_MAE_std_prev_frame_copy_list)
dynamic_plotting_list.append(dynamic_MSE_mean_list)
dynamic_plotting_list.append(dynamic_MSE_mean_prev_frame_copy_list)
dynamic_plotting_list.append(dynamic_MSE_std_list)
dynamic_plotting_list.append(dynamic_MSE_std_prev_frame_copy_list)
dynamic_plotting_list.append(dynamic_PSNR_mean_list)
dynamic_plotting_list.append(dynamic_PSNR_mean_prev_frame_copy_list)
dynamic_plotting_list.append(dynamic_PSNR_movement_mean_list)
dynamic_plotting_list.append(dynamic_PSNR_movement_mean_prev_frame_copy_list)
dynamic_plotting_list.append(dynamic_SSIM_mean_list)
dynamic_plotting_list.append(dynamic_SSIM_mean_prev_frame_copy_list)
dynamic_plotting_list.append(dynamic_SSIM_movement_mean_list)
dynamic_plotting_list.append(dynamic_SSIM_movement_mean_prev_frame_copy_list)
dynamic_plotting_list.append(dynamic_Sharpness_difference_mean_list)
dynamic_plotting_list.append(dynamic_Sharpness_grad_mean_list)

normal_plotting_list.append(normal_Conditioned_SSIM_mean_list)
normal_plotting_list.append(normal_MAE_mean_list)
normal_plotting_list.append(normal_MAE_mean_prev_frame_copy_list)
normal_plotting_list.append(normal_MAE_std_list)
normal_plotting_list.append(normal_MAE_std_prev_frame_copy_list)
normal_plotting_list.append(normal_MSE_mean_list)
normal_plotting_list.append(normal_MSE_mean_prev_frame_copy_list)
normal_plotting_list.append(normal_MSE_std_list)
normal_plotting_list.append(normal_MSE_std_prev_frame_copy_list)
normal_plotting_list.append(normal_PSNR_mean_list)
normal_plotting_list.append(normal_PSNR_mean_prev_frame_copy_list)
normal_plotting_list.append(normal_PSNR_movement_mean_list)
normal_plotting_list.append(normal_PSNR_movement_mean_prev_frame_copy_list)
normal_plotting_list.append(normal_SSIM_mean_list)
normal_plotting_list.append(normal_SSIM_mean_prev_frame_copy_list)
normal_plotting_list.append(normal_SSIM_movement_mean_list)
normal_plotting_list.append(normal_SSIM_movement_mean_prev_frame_copy_list)
normal_plotting_list.append(normal_Sharpness_difference_mean_list)
normal_plotting_list.append(normal_Sharpness_grad_mean_list)
fig, ax=plt.subplots(5,4,sharex=True, figsize=(30,30))
fig.suptitle("Results_Dict_Plot", fontsize='xx-large', fontweight='bold')
i=0
for j in range(5):
    for k in range(4):
        if i<len(dynamic_plotting_list):
            print("Creating subplot for "+str(j)+" and "+str(k))
            ax[j, k].plot(fps_list, dynamic_plotting_list[i])
            ax[j, k].plot(fps_list, normal_plotting_list[i])
            ax[j, k].grid(True)
            ax[j, k].set_xlabel("FPS", fontsize='large')
            ax[j, k].set_xticks(fps_list)
            s=''
            count=0
            new_count=0
            l=results_legend[i].split('_')
            for l1 in l:
                if count<2:
                    s+=l1+" "
                    count+=1
                elif new_count>0:
                    s+=l1+" "
                elif count==2:
                    s+=l1+"\n"
                    new_count+=1
            ax[j, k].set_ylabel(s,wrap=True,fontsize='large')
            ax[j, k].set_title(str(j)+"x"+str(k), fontsize='large' ,ha='center')
            #ax[j, k].legend(["Dyn.", "Norm."],loc='center left')
            i+=1
        else:
            ax[j,k].plot(fps_list, dynamic_val_loss_list)
            ax[j,k].plot(fps_list, normal_val_loss_list)
            ax[j, k].grid(True)
            ax[j, k].set_xlabel("FPS", fontsize='large')
            ax[j, k].set_xticks(fps_list)
            ax[j, k].set_ylabel("Val_Loss", fontsize='large')
            ax[j, k].set_title(str(j)+"x"+str(k), fontsize=20 ,ha='center')
            #ax[j, k].legend(["Dyn.", "Norm."],loc='center left')
        for tick in ax[j,k].yaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
#plt.xlabel("FPS")
fig.subplots_adjust(hspace=0.5, wspace=0.5)
fig.legend(["Dynamic", "Normal"], fontsize='xx-large' ,loc=8)
fig.savefig(save_dir+"/FPS_Varying_Result_Dict_plot_For_Something-Something_dataset.png", dpi=600)
fig.show()
