from collections import defaultdict
import glob
import imageio
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.transform import rescale
from scipy.fftpack import fft, fftshift
import sys
import time


# Please don't remove this function - Danke schön (Thanks and Regards, Vageesh Saxena)
def plot_loss_curves(history, evaluation_method, model, result_dir):
    # plots the accuracy and loss curve for the mentioned evaluation method
    """
    history: fitted model
    evaluation_method: categorical_crossentropy, threshold_score etc(dtype:string)
    model: neural network architecture(dtype:string)
    result_dir : directory in which results are stored(dtype:string)
    """

    plot_dir = os.path.join(result_dir, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig = plt.figure()
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Reconstruction Loss', fontsize=16)
    plt.title("Loss Curves :" + model + "_for_" + evaluation_method, fontsize=16)
    fig.savefig(os.path.join(plot_dir, "loss_" + model + "_" + evaluation_method + ".png"))


# Please don't remove this function - Danke schön (Thanks and Regards, Vageesh Saxena)
def plot_accuracy_curves(history, evaluation_method, model, result_dir):
    # plots the accuracy and loss curve for the mentioned evaluation method
    """
    history: fitted model
    evaluation_method: categorical_crossentropy, threshold_score etc(dtype:string)
    model: neural network architecture(dtype:string)
    result_dir : directory in which results are stored(dtype:string)
    """

    plot_dir = os.path.join(result_dir, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig = plt.figure()
    plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title("Accuracy Curves :" + model + "_for_" + evaluation_method, fontsize=16)
    fig.savefig(os.path.join(plot_dir, "accuracy_" + model + "_" + evaluation_method + ".png"))


def plot_video(video=None, stats=False, save_pdf=False, RESULTS_SAVE_DIR='plots', vid_path=None):
    '''
    plotting function that shows or saves a video as a sequence of frames on a grid
    can optionally show stats
    index of the frame is on the x label
    either video or path has to be given

    Arguments:
    video:             an array of (n_frames, size_x, size_y)
                       set to None by default in case path is given
    stats:             False by default
                       if True then it will output statistics for each frame (by index)
                       frame.shape, np.min(frame), np.mean(frame), np.max(frame)
    save_pdf:          False by default, it will show the plot only
                       if True, it saves plot to RESULTS_SAVE_DIR
                       the name of the file is current date and time
    RESULTS_SAVE_DIR:  \plots by default
                       folder to save plots into, gets created if doesn't exist
    vid_path:          None by default
                       if given it has to be in the form as seen in data.csv['path']
                       e.g.'/data/videos/something-something-v2/preprocessed/train/51646'
    '''
    assert type(video) == np.ndarray or vid_path, "Please specify a video to plot as an array or path."
    assert not (type(video) == np.ndarray and vid_path), "Please only speficy either a video or a path."

    if vid_path != None:
        files = []
        vid_list = []
        for im_path in glob.glob(vid_path + '/*.png'):
            files.append(im_path)

        for file in sorted(files):
            im = (imageio.imread(file))
            vid_list.append(im)

        video = np.array(vid_list)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)

    height_of_plot = np.shape(video)[0] / 8 if np.shape(video)[0] % 8 == 0 else np.shape(video)[0] // 8 + 1
    figs = []

    fig1 = plt.figure(figsize=((np.shape(video)[2] + 20) / 80 * 8, (np.shape(video)[1] + 50) / 80 * height_of_plot))

    for ind in range(np.shape(video)[0]):
        plt.subplot(height_of_plot, 8, ind + 1)
        plt.imshow(video[ind])
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False,
                        right=False, labelbottom=False, labelleft=False)
        plt.xlabel(ind, fontsize=15)

    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.show()
    figs.append(fig1)

    if stats:
        data = []
        for ind, frame in enumerate(video):
            data.append([(ind), (frame.shape), (np.min(frame)), (np.mean(frame)), (np.max(frame))])
        fig2 = plt.figure(figsize=((np.shape(video)[2] + 20) / 80 * 8, len(video)))

        the_table = plt.table(cellText=data,
                              colLabels=["Index", "Frame shape", "Frame min", "Frame mean", "Frame max"],
                              bbox=[0, 1, 1, 0.8])

        the_table.set_fontsize(15)
        plt.xticks([])
        plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        figs.append(fig2)

    if save_pdf:
        with PdfPages(RESULTS_SAVE_DIR + '/' + timestr + '.pdf') as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches='tight')
    else:
        plt.show()
        
def plot_errors(error_outputs, X_test, ind=0):
    '''
    function used to produce error_matrices for the evaluation mode in the pipeline
    '''
    layer_error = []
    matrices = []
    
    for layer in range(len(error_outputs)):  
        matrices.append([])
        layer_error = error_outputs[layer][0]
        vid_error = layer_error[ind]
        for frame in vid_error:   
            frame = np.transpose(frame, (2,0,1))
            frame_matrix =  np.sum([mat for mat in frame], axis=0)/len(frame)
            frame_matrix_rescaled = rescale(frame_matrix, (2**layer, 2**layer))
            matrices[layer].append(frame_matrix_rescaled)              

    return matrices 

def plot_changes_in_r(X_hats, ind, std_param=0.5):
    '''
    funtion used to produce the values needed for R plots in the evaluation mode in the pipeline
    '''
    results = []
    vid = [(x_hat[0][ind], x_hat[1]) for x_hat in X_hats]
    frames = defaultdict(lambda : defaultdict(float))
     
    for channel in range(len(vid)):
        for ind, frame in enumerate(vid[channel][0]):   
            frames[channel][ind] = (np.average(np.abs(frame)), np.std(np.abs(frame)))
        
        results.append([])        
        y = [tup[0] for tup in frames[channel].values()]
        x = [n for n in range(len(y))]
        std = [tup[1] for tup in frames[channel].values()]
        results[channel].append((y, x, std))
          
    return results
    

def return_difference(X_test, X_hat):
    '''
    returns mean SSIM and PSNR for all frames + only for frames with movement for one video
    '''
    ssim_list = []
    psnr_list = []
    ssim_movement_list = []
    psnr_movement_list = []
        
    for ind in range(len(X_test)):
        ssim_list.append(ssim(X_test[ind], X_hat[ind], multichannel=True))
        #for movement only
        if 0 < ind < len(X_test) and not np.array_equal(X_test[ind], X_test[ind-1]):
            ssim_movement_list.append(ssim(X_test[ind], X_hat[ind], multichannel=True))   
            psnr_movement_list.append(compare_psnr(X_test[ind], X_hat[ind], epsilon=0.00005))
         
        psnr_list.append(compare_psnr(X_test[ind], X_hat[ind], epsilon=0.00005))
    
    return np.mean(ssim_list), np.mean(psnr_list), np.mean(ssim_movement_list), np.mean(psnr_movement_list)
                             
def sharpness_difference_grad(X_test, X_hat, epsilon=0.00005):
    '''
    return sharpness difference for one video based on Mathieu 2016 
    the bigger the better
    '''
    differences = []
    
    for ind, frame in enumerate(X_test): 
       
        frame2 = np.transpose(X_hat[ind], (2,0,1))
        frame = np.transpose(frame, (2,0,1))
        channel_list = []

        for channel in range(frame.shape[0]):
            
            gy, gx = np.gradient(frame[channel])
            sum1 =  (np.abs(gy) + np.abs(gx))
                   
            gy2, gx2 = np.gradient(frame2[channel])
            sum2 =  (np.abs(gy2) + np.abs(gx2))
          
            res = 10 * np.log10((255*255) / ((np.sum(np.abs(sum1-sum2)+epsilon) / gy.shape[0]*gy.shape[1])))
            
            channel_list.append(res)
                        
        
        if channel_list != []:
            differences.append(np.mean(channel_list))
        
    return np.mean(differences)
    
def conditioned_ssim(X_test, X_hat):
    '''
    ssim of a video frame by frame, conditioned on previous frame copy
    (max_SSIM - SSIM(prev_actual, current_pred)) * SSIM(current_actual, current_pred)
    '''
    current = []
    prev = []
    for ind in range(len(X_test)):
        current.append(ssim(X_test[ind], X_hat[ind], multichannel=True))
        if ind < len(X_test)-1:
            prev.append(ssim(X_test[ind], X_hat[ind+1], multichannel=True))
    
    return np.mean([(1-curr)*prv for curr,prv in zip(current,prev)])

def sharpness(vid):
    '''
    returns sharpness of one video based on De and Masilamani 2013
    '''
    FMs = []
    for frame in range(vid.shape[0]):
        for channel in range(vid.shape[3]):
            pic = np.transpose(vid[frame], (2,0,1))[channel]
            F = fft(pic)
            Max = np.max(np.absolute(fftshift(F)))
            TH = (F > Max/1000).sum()
            FM = TH / (pic.shape[0] * pic.shape[1])
            FMs.append(FM)
          
    return np.mean(FMs)

def sharpness_difference(X_test, X_hat):
    '''
    calculates the sharpness difference between prediction and actual video, based on the previous sharpness function
    the smaller the better
    '''
    actual_sharpness = sharpness(X_test)
    pred_sharpness = sharpness(X_hat)
    return (actual_sharpness - pred_sharpness)

#the functions below are from https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/simple_metrics.py#L99 

dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}

def compare_mse(im1, im2):
    """Compute the mean-squared error between two images.
    Parameters
    ----------
    im1, im2 : ndarray
        Image.  Any dimensionality.
    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.
    """
    #_assert_compatible(im1, im2)
    #im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)

def compare_psnr(im_true, im_test, data_range=None, epsilon = 0.00005): ################ added parameter: epsilon
    """ Compute the peak signal to noise ratio (PSNR) for an image.
    Parameters
    ----------
    im_true : ndarray
        Ground-truth image.
    im_test : ndarray
        Test image.
    data_range : int
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.
    noise : float
        Number added to MSE.
    Returns
    -------
    psnr : float
        The PSNR metric.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    #_assert_compatible(im_true, im_test)

    if data_range is None:
        if im_true.dtype != im_test.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "im_true.")
        dmin, dmax = dtype_range[im_true.dtype.type]
        true_min, true_max = np.min(im_true), np.max(im_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "im_true has intensity values outside the range expected for "
                "its data type.  Please manually specify the data_range")
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin

    #im_true, im_test = _as_floats(im_true, im_test)

    err = compare_mse(im_true, im_test) + epsilon ######################### modification: epsilon to mse to avoid zero division issues
    return 10 * np.log10((data_range ** 2) / err)

      