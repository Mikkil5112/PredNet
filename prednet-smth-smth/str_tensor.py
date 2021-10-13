# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:26:20 2021

@author: mikki
"""
import numpy as np
import skimage
import plotly.express as px
import plotly.graph_objects as go
from skimage.feature import structure_tensor
from skimage.feature import structure_tensor_eigenvalues
from PIL import Image as im
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
class Str_Tensor():
        def __init__(self, X_test=[],
            total_vids_to_plt=0):
            
            
            A_elems_whole_video_list=[]
            eigen_whole_video_list=[]
            A_elems_whole_video_by_frame_list=[]
            eigen_whole_video_by_frame_list=[]
            # f_count=0
            # v_count=0
            print("Shape of X_test:"+str(X_test.shape))
            for i in range(total_vids_to_plt):
                #For simplicity of computation we convert the RGB video to a Grayscale video
                video=skimage.color.rgb2gray((X_test[i] * 255).round().astype(np.uint8))
                V_min, V_max=video[:,:,:].min(),video[:,:,:].max()
                print("Shape of the video:"+str(video.shape))
                A_elems=structure_tensor(video)
                eigen=structure_tensor_eigenvalues(A_elems)
                coords=np.unravel_index(eigen.argmax(), eigen.shape)
                print("Coords: "+str(coords))
                n_plane, n_row, n_col=video.shape
                subplots = np.dstack((video[coords[1], :, :],video[:, coords[2], :], video[:, :, coords[3]]))
                fig=px.imshow(subplots,
                    zmin=V_min,
                    zmax=V_max,
                    facet_col=2,
                    labels={'facet_col': 'longitudinal'})
                fig.show()
                # #Computing the range of values in each channel of the RGB video
                # for j in range(n_chan):
                #     V_min_0, V_max_0=video[:,:,:,0].min(),video[:,:,:,0].max()
                #     #print(f'range of Channel 0: ({V_min_0}, {V_max_0})')
                #     V_min_1, V_max_1=video[:,:,:,1].min(),video[:,:,:,1].max()
                #     #print(f'range of Channel 1: ({V_min_1}, {V_max_1})')
                #     V_min_2, V_max_2=video[:,:,:,2].min(),video[:,:,:,2].max()
                #     #print(f'range of Channel 2: ({V_min_2}, {V_max_2})')
                # Z, Y, X=np.mgrid[:n_plane,:n_row,:n_col]
                # fig = go.Figure(data=go.Volume(
                #     x=X.flatten(),
                #     y=Y.flatten(),
                #     z=Z.flatten(),
                #     value=((video * 255).round().astype(np.uint8)).flatten(),
                #     opacity=0.5,
                #     slices_z=dict(show=True, locations=[4])
                #     ))
                # fig.show()
                # #Here We consider each video in the test data as a 3D-Multichannel image of the format [plane, row, column, channel]
                # 
                # 
                
                # A_elems_whole_video_list.append(A_elems)
                # eigen_whole_video_list.append(eigen)
                
                #Converting RGB video to a Gray scale video
                
                # A_elems_video_by_frame_list=[]
                # eigen_video_by_frame_list=[]
                # for t in video:
                #     #Here We consider each frame in the video as a 2D-Multichannel image of the format [row, column, channel]
                #     A_elems=structure_tensor(t)
                    
                #     #A_elems_video_by_frame_list.append(A_elems)
                    
                #     eigen=structure_tensor_eigenvalues(A_elems)
                #     coords=np.unravel_index(eigen.argmax(), eigen.shape)
                #     print("Coords: "+str(coords))
                    
                    #eigen_video_by_frame_list.append(eigen)
                    
                #A_elems_whole_video_by_frame_list.append(A_elems_video_by_frame_list)
                #eigen_whole_video_by_frame_list.append(eigen_video_by_frame_list)    
            # coords=np.unravel_index(eigen.argmax(), eigen.shape)
            # assert coords[0]==0
            # print("Coords: "+str(coords))
            # fig=px.imshow(
            #     eigen[:, coords[1], :, :],
            #     facet_col=0,
            #     labels={'x': 'col', 'y': 'row', 'facet_col': 'rank'},
            #     title=f'Eigenvalues for plane Z = {coords[1]}.')
            # fig.show()
            # Shape_list=[]
            # for video in A_elems_whole_video_list:
            #     if np.shape(video) not in Shape_list:
            #         Shape_list.append(np.shape(video))
            # print("Shapes in A_elems of videos as a whole list are: "+str(Shape_list))
            # Shape_list=[]
            # for video in eigen_whole_video_list:
            #     if np.shape(video) not in Shape_list:
            #         Shape_list.append(np.shape(video))
            # print("Shapes in eigen values of videos as a whole list are: "+str(Shape_list))
            # Shape_list=[]
            # for video in A_elems_whole_video_by_frame_list:
            #     for frame in video:
            #         if np.shape(frame) not in Shape_list:
            #             Shape_list.append(np.shape(frame))
            # print("Shapes in A_elems of videos by frame list are: "+str(Shape_list))
            # Shape_list=[]
            # for video in eigen_whole_video_by_frame_list:
            #     for frame in video:
            #         if np.shape(frame) not in Shape_list:
            #             Shape_list.append(np.shape(frame))
            # print("Shapes in A_elems of videos by frame list are: "+str(Shape_list))
            