'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import glob
import hickle as hkl
import skimage
from skimage.feature import structure_tensor, structure_tensor_eigvals
import json
import numpy as np
from six.moves import cPickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *
from viz_utils import plot_loss_curves, plot_errors, plot_changes_in_r, return_difference
from viz_utils import conditioned_ssim, sharpness_difference_grad, sharpness, sharpness_difference

n_plot = 8 
batch_size = 8
nt = 24
nb_layers=5
im_height, im_width=(48,80)
#weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')
weight_files = glob.glob( WEIGHTS_DIR + "/*.hdf5")
if len(weight_files) > 1:
        # collect (loss, filename) tuples, sort the tuples by the loss, and then collect the filenames only
        _, weights_sorted = zip(*
                             sorted(
                                 [(float(w.split("loss")[-1].split(".hdf5")[0]), w) for w in weight_files]
                             )
                            )
        # select the best 1 or n models with lowest reconstruction loss
        weight_file = weights_sorted[0]
weights_file=weight_file.replace(os.sep,'/')
filename = weights_file.split("/")[-1].split(".hdf5")[0]
# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet, 'nb_layers':nb_layers})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
print(input_shape)
input_shape[0] = nt
print(input_shape)
#quit()
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)
test_model.summary()

test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)

#initialize lists for evaluation        
mse_model_list, mse_prev_list, mae_model_list, mae_prev_list = ([] for i in range(4))
psnr_list, ssim_list, sharpness_grad_list, psnr_prev_list, ssim_prev_list, sharpness_grad_prev_list = ([] for i in range(6))
psnr_movement_list, psnr_movement_prev_list, ssim_movement_list, ssim_movement_prev_list =  ([] for i in range(4))
conditioned_ssim_list, sharpness_list, sharpness_prev_list = ([] for i in range(3))
accuracy_list = []

for index, data in enumerate(test_generator):
    # Only consider steps_test number of steps
    if index > len(test_generator):
        break
    # X_test = test_generator.next()[0]
    X_test = data[0]
    #print(X_test)
    X_hat = test_model.predict(X_test, batch_size)
    if data_format == 'channels_first':
        X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
        X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))
    print(len(X_test))    
    #X_test = test_generator.create_all()
    #print(X_test)
    #quit()
    # X_hat = test_model.predict(X_test, batch_size)
    # if data_format == 'channels_first':
    #     X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    #     X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))
    # mean square error
    mse_model_list.append(
        np.mean((X_test[:, 1:] - X_hat[:, 1:]) ** 2))  # look at all timesteps except the first
    mse_prev_list.append(np.mean((X_test[:, :-1] - X_test[:, 1:]) ** 2))
    # mean absolute error
    mae_model_list.append(
        np.mean(np.abs(X_test[:, 1:] - X_hat[:, 1:])))
    mae_prev_list.append(np.mean(np.abs(X_test[:, :-1] - X_test[:, 1:])))
    # ssim
    ssim_list.append(np.mean([return_difference(X_test[ind][1:], X_hat[ind][1:])[0] for ind in range(X_test.shape[0])]))
    ssim_prev_list.append(np.mean([return_difference(X_test[ind][:-1], X_test[ind][1:])[0] 
                                   for ind in range(X_test.shape[0]-1)]))
    ssim_movement_list.append(np.mean([return_difference(X_test[ind], X_hat[ind])[2] 
                                       for ind in range(X_test.shape[0])]))
    ssim_movement_prev_list.append(np.mean([return_difference(X_test[ind][:-1], X_test[ind][1:])[2] 
                                   for ind in range(X_test.shape[0]-1)])) 
    conditioned_ssim_list.append(np.mean([conditioned_ssim(X_test[ind], X_hat[ind]) 
                                   for ind in range(X_test.shape[0])])) 
    
    # psnr
    psnr_list.append(np.mean([return_difference(X_test[ind][1:], X_hat[ind][1:])[1] for ind in range(X_test.shape[0])]))            
    psnr_prev_list.append(np.mean([return_difference(X_test[ind][:-1], X_test[ind][1:])[1] 
                                   for ind in range(X_test.shape[0]-1)]))
    psnr_movement_list.append(np.mean([return_difference(X_test[ind], X_hat[ind])[3] 
                                       for ind in range(X_test.shape[0])]))
    psnr_movement_prev_list.append(np.mean([return_difference(X_test[ind][:-1], X_test[ind][1:])[3] 
                                   for ind in range(X_test.shape[0]-1)]))
    
    # sharpness
    sharpness_grad_list.append(np.mean([sharpness_difference_grad(X_test[ind][1:], X_hat[ind][1:])
                                   for ind in range(X_test.shape[0])]))
    sharpness_list.append(np.mean([sharpness_difference(X_test[ind][1:], X_hat[ind][1:])
                                  for ind in range(X_test.shape[0])]))
results_dict = {                    
    "MSE_mean": float("{:.6f}".format(np.mean(mse_model_list))), 
    "MSE_std":float(("{:.6f}".format(np.std(mse_model_list)))), 
    "MSE_mean_prev_frame_copy":float("{:.6f}".format(np.mean(mse_prev_list))), 
    "MSE_std_prev_frame_copy":float("{:.6f}".format(np.std(mse_prev_list))),
    "MAE_mean": float("{:.6f}".format(np.mean(mae_model_list))), 
    "MAE_std":float(("{:.6f}".format(np.std(mae_model_list)))), 
    "MAE_mean_prev_frame_copy":float("{:.6f}".format(np.mean(mae_prev_list))), 
    "MAE_std_prev_frame_copy":float("{:.6f}".format(np.std(mae_prev_list))),
    "SSIM_mean": float("{:.6f}".format(np.mean(ssim_list))), 
    "SSIM_mean_prev_frame_copy": float("{:.6f}".format(np.mean(ssim_prev_list))), 
    "SSIM_movement_mean": float("{:.6f}".format(np.mean(ssim_movement_list))), 
    "SSIM_movement_mean_prev_frame_copy": float("{:.6f}".format(np.mean(ssim_movement_prev_list))), 
    "Conditioned_SSIM_mean": float("{:.6f}".format(np.mean(conditioned_ssim_list))),
    "PSNR_mean": float("{:.6f}".format(np.mean(psnr_list))),
    "PSNR_mean_prev_frame_copy": float("{:.6f}".format(np.mean(psnr_prev_list))), 
    "PSNR_movement_mean": float("{:.6f}".format(np.mean(psnr_movement_list))), 
    "PSNR_movement_mean_prev_frame_copy": float("{:.6f}".format(np.mean(psnr_movement_prev_list))), 
    "Sharpness_grad_mean": float("{:.6f}".format(np.mean(sharpness_grad_list))),
    #"Sharpness_grad_mean_prev_frame_copy": float("{:.6f}".format(np.mean(sharpness_grad_prev_list))),
    "Sharpness_difference_mean": float("{:.6f}".format(np.mean(sharpness_list)))
    #"Sharpness_difference_mean_prev_frame_copy" : float("{:.6f}".format(np.mean(sharpness_prev_list)))
    }

if not os.path.exists(RESULTS_SAVE_DIR): os.makedirs(RESULTS_SAVE_DIR,exist_ok=True)

with open(os.path.join(RESULTS_SAVE_DIR, 'scores_' + filename + '.json'), 'w') as f:
        json.dump(results_dict, f, sort_keys=True,  indent=4)
print("========================= Plotting results for model {}=======================".format(filename))
#Plotting extra graphs
no_layers = nb_layers
extra_output_modes = (['E'+str(no) for no in range(no_layers)] + ['A'+str(no) for no in range(no_layers)] 
                                 + ['Ahat'+str(no) for no in range(no_layers)] + ['R'+str(no) for no in range(no_layers)])
extra_test_models = []
for output_mode in extra_output_modes:
    layer_config['output_mode'] = output_mode    
    data_format = (layer_config['data_format'] if 'data_format' in layer_config 
                    else layer_config['dim_ordering'])
    extra_test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
    input_shape = list(train_model.layers[0].batch_input_shape[1:])
    input_shape[0] = nt
    inputs = Input(shape=tuple(input_shape))
    extra_predictions = extra_test_prednet(inputs)
    extra_test_model = Model(inputs=inputs, outputs=extra_predictions)
    extra_test_models.append((extra_test_model, output_mode))
    
#Create outputs for extra plots
error_X_hats = []
R_X_hats = []
A_X_hats = []
Ahat_X_hats = []
total_vids_to_plt=n_plot
for test_model, output_mode in extra_test_models:
    if output_mode[0]=='R':
        R_X_hat = test_model.predict(X_test, total_vids_to_plt) 
        R_X_hats.append((R_X_hat, output_mode))
    elif output_mode[0]=='E':
        error_X_hat = test_model.predict(X_test, total_vids_to_plt) 
        error_X_hats.append((error_X_hat, output_mode))
    elif 'Ahat' in output_mode: 
        Ahat_X_hat = test_model.predict(X_test, total_vids_to_plt) 
        Ahat_X_hats.append((Ahat_X_hat, output_mode))
    else: # output_mode[0]=='A':
        A_X_hat = test_model.predict(X_test, total_vids_to_plt) 
        A_X_hats.append((A_X_hat, output_mode))

plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'predictions/' + filename)
if not os.path.exists(plot_save_dir):
             os.makedirs(plot_save_dir)
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
sources=hkl.load(test_sources)
for i in range(n_plot):
    #computing Frame wise Local Orientation, Coherence, Energy and the Dominant Orientation direction
     eps = 1e-20
     Ori_list=[]
     Dom_Ori_list=[]
     Coh_list=[]
     Ene_list=[]
     #print(len(X_test[i]))
     for t in X_test[i]:
         Axx, Axy, Ayy=structure_tensor(skimage.color.rgb2gray((t * 255).round().astype(np.uint8)), 
                                        mode="reflect", order='xy')
         l1, l2 = structure_tensor_eigvals(Axx, Axy, Ayy)
         ori = np.arctan2(2 * Axy, (Ayy - Axx)) / 2
         dom_ori = np.arctan2(2 * Axy.mean(), (Ayy.mean() - Axx.mean())) / 2
         coh = ((l2 - l1) / (l2 + l1 + eps)) ** 2
         ene = np.sqrt(Axx + Ayy)
         ene /= ene.max()
         
         Ori_list.append(ori)
         Coh_list.append(coh)
         Dom_Ori_list.append(np.rad2deg(dom_ori).round())
         Ene_list.append(ene)
        
     
     fig, ax = plt.subplots(ncols=1, nrows=20, sharex=True, figsize=(nt, 35 * aspect_ratio),
                               gridspec_kw={'height_ratios':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3]})
     source=str(sources[i].split("-")[0]+"_"+sources[i].split("-")[1].split("_", 3)[3])
     # print(source)
     # quit()
     # set the title of the plot
     title =  "{} Max. Dominant Orientation Direction:{} Deg., Min Dominant Orientation Direction:{} Deg.".format(source,
         max(Dom_Ori_list).round(), min(Dom_Ori_list).round())
    
     fig.suptitle(title)
    
     #Plot video
     ax = plt.subplot()
     ax.imshow(np.concatenate([t for t in X_test[i]], axis=1), interpolation='none', aspect="auto")
     ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                                     labelbottom=False, labelleft=False)
     ax.set_ylabel(r'Actual', fontsize=10)
     ax.set_xlim(0,nt*im_width)
    
     #Plot predictions
     divider = make_axes_locatable(ax)
     ax = divider.append_axes("bottom", size="100%", pad=0.0)                                             
     ax.imshow(np.concatenate([t for t in X_hat[i]], axis=1), interpolation='none', aspect="auto")
     ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                                     labelbottom=False, labelleft=False)
     ax.set_ylabel(r'Prediction', fontsize=10)
     ax.set_xlim(0,nt*im_width)
     
     # computing Correlation coeffiecients for each frame in X_test[i], X_hat[i]
     Corr_coeff_list=[]
     for t, t_hat in zip(X_test[i], X_hat[i]):
         cm=np.corrcoef(t.flat, t_hat.flat)
         Corr_coeff_list.append(cm[0,1])
    
    #Creating Plot with Cross correlation coeffecients
     ax = divider.append_axes("bottom", size="300%", pad=0.2) 
     x=[im_width/2+j*im_width for j in range(nt)]
     ax.fill_between(x, Dom_Ori_list, alpha=0.1)
     ax.plot(x, Corr_coeff_list)
     ax.set_xlim(0,nt*im_width)
     ax.set_ylim(0,1)
     ax.set_xticks(np.arange(im_width/2, nt*im_width, step=im_width))                
     ax.set_xticklabels(np.arange(1,nt+1))
     ax.grid(True)
     ax.set_ylabel(r"Corr. Coeff.", fontsize=10)
     ax.xaxis.set_label_position('top')
     
     #Creating Plot with Dom_ori_Direction in Degrees
     ax = divider.append_axes("bottom", size="300%", pad=0.2) 
     x=[im_width/2+j*im_width for j in range(nt)]
     ax.fill_between(x, Dom_Ori_list, alpha=0.1)
     ax.plot(x, Dom_Ori_list)
     ax.set_xlim(0,nt*im_width)
     ax.set_ylim(min(Dom_Ori_list), max(Dom_Ori_list))
     ax.set_xticks(np.arange(im_width/2, nt*im_width, step=im_width))                
     ax.set_xticklabels(np.arange(1,nt+1))
     ax.grid(True)
     ax.set_ylabel(r"D_O_D in Deg.", fontsize=10)
     ax.xaxis.set_label_position('top')
    ######################################### Extra plot #############################################
     std_param=0.5#parameter for the plotting R function: how many times the STD should we shaded     
       #Create values for R plots
     results = plot_changes_in_r(R_X_hats, i, std_param=std_param)
     ax = divider.append_axes("bottom", size="300%", pad=0.2)
     #Plot R plots
     for layer in results:
        (y, x, std) = layer[0]
        x = [im_width/2+item*im_width for item in x]
        ax.fill_between(x, [(val-std_param*dev) for val, dev in zip(y, std)],
                         [(val+std_param*dev) for val, dev in zip(y, std)], alpha=0.1)
        ax.plot(x, y)

     ax.set_xlim(0, nt*im_width)
     ax.set_xticks(np.arange(im_width/2, nt*im_width, step=im_width))
     ax.set_xticklabels(np.arange(1, nt+1))
     ax.grid(True)
     ax.set_ylabel(r"Mean R activations", fontsize=10)
     ax.xaxis.set_label_position('top') 
     ax.legend(['R'+str(no) for no in range(no_layers)], loc='center left')
       
     #Create values for E plots      
     results = plot_changes_in_r(error_X_hats, i, std_param=std_param)
     ax = divider.append_axes("bottom", size="300%", pad=0.2)                                                                
     #Plot E plots
     for layer in results:
        (y,x,std) = layer[0]
        x = [im_width/2+item*im_width for item in x]
        ax.fill_between(x, [(val-std_param*dev) for val,dev in zip(y,std)], 
                        [(val+std_param*dev) for val,dev in zip(y,std)], alpha=0.1)
        ax.plot(x, y)
       
     ax.set_xlim(0,nt*im_width)
     ax.set_xticks(np.arange(im_width/2, nt*im_width, step=im_width))                
     ax.set_xticklabels(np.arange(1,nt+1))
     ax.grid(True)                  
     ax.set_ylabel(r"Mean E activations", fontsize=10)
     ax.xaxis.set_label_position('top') 
     ax.legend(['E'+str(no) for no in range(no_layers)], loc='center left')
       
     #Create error output matrices to plot inside the next loop
     R_matrices = plot_errors(R_X_hats, X_test, ind=i)
     A_matrices =  plot_errors(A_X_hats, X_test, ind=i) 
     Ahat_matrices = plot_errors(Ahat_X_hats, X_test, ind=i)
     error_matrices = plot_errors(error_X_hats, X_test, ind=i)
     #Plot R, A, Ahat and errors for each layer
     for layer in range(len(error_matrices)):   
            ##R
            ax = divider.append_axes("bottom", size="100%", pad=0.2)                                             
            ax.imshow(np.concatenate([t for t in R_matrices[layer]], axis=1), 
                               interpolation='nearest', cmap='gray', aspect="auto")
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, 
                                    right=False, labelbottom=False, labelleft=False)
            ax.set_ylabel(r"R" + str(layer), fontsize=10)
            ax.set_xlabel(r"Layer " + str(layer), fontsize=10)
            ax.xaxis.set_label_position('top') 
            ax.set_xlim(0,nt*im_width)
            ##A
            ax = divider.append_axes("bottom", size="100%", pad=0.0)                                             
            ax.imshow(np.concatenate([t for t in Ahat_matrices[layer]], axis=1), 
                               interpolation='nearest', cmap='gray', aspect="auto")
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, 
                                    right=False, labelbottom=False, labelleft=False)
            ax.set_ylabel(r"Ahat" + str(layer), fontsize=10)
            ax.set_xlim(0,nt*im_width)
            ##Ahat
            ax = divider.append_axes("bottom", size="100%", pad=0.0)                                     
            ax.imshow(np.concatenate([t for t in A_matrices[layer]], axis=1), 
                               interpolation='nearest', cmap='gray', aspect="auto")
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, 
                                    right=False, labelbottom=False, labelleft=False)
            ax.set_ylabel(r"A" + str(layer), fontsize=10)
            ax.set_xlim(0,nt*im_width)
            ##E
            ax = divider.append_axes("bottom", size="100%", pad=0.0)                                             
            ax.imshow(np.concatenate([t for t in error_matrices[layer]], axis=1), 
                               interpolation='nearest', cmap='gray', aspect="auto")
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, 
                                    right=False, labelbottom=False, labelleft=False)
            ax.set_ylabel(r"E" + str(layer), fontsize=10)
            ax.set_xlim(0,nt*im_width)                
    #####################################################################################################################
       
     plt.subplots_adjust(hspace=0., wspace=0., top=0.97)
     plt.savefig(plot_save_dir + "/plot-" +str(i)+source+'.png')
     #plt.savefig(plot_save_dir + "/" + test_data_for_plt.loc[i,'name'] + str(test_data_for_plt.loc[i,'id']) + '.png')
     plt.clf()
     
     
# # Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
# mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
# mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
# if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
# f = open(RESULTS_SAVE_DIR + 'prediction_scores.txt', 'w')
# f.write("Model MSE: %f\n" % mse_model)
# f.write("Previous Frame MSE: %f" % mse_prev)
# f.close()

# # Plot some predictions
# aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
# plt.figure(figsize = (nt, 2*aspect_ratio))
# gs = gridspec.GridSpec(2, nt)
# gs.update(wspace=0., hspace=0.)
# plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
# if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
# plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
# for i in plot_idx:
#     for t in range(nt):
#         plt.subplot(gs[t])
#         plt.imshow(X_test[i,t], interpolation='none')
#         plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
#         if t==0: plt.ylabel('Actual', fontsize=10)

#         plt.subplot(gs[t + nt])
#         plt.imshow(X_hat[i,t], interpolation='none')
#         plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
#         if t==0: plt.ylabel('Predicted', fontsize=10)

#     plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
#     plt.clf()
