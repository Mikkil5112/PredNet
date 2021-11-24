# The Performance of PredNet Using Predictive Coding in the Visual Cortex: An Empirical Analysis ([paper](https://ieeexplore-ieee-org.ezproxyprod.ucs.louisiana.edu/stamp/stamp.jsp?tp=&arnumber=9599098))
PredNet is a deep recurrent convolutional neural network developed by Lotter et al.. The architecture drew inspiration from a Hierarchical Neuroscience model of visual processing described and demonstrated by Rao and Ballard. In 2020, Rane, Roshan Prakash, et al. published a critical review of PredNet stating its lack of performance in the task of next frame prediction in videos on a crowd sourced action classification dataset. While their criticism was nearly coherent, it is dubious, when observed, considering the findings reported by Rao and Ballard. In this paper, we reevaluate their review through the application of the two primary datasets used by Lotter et al. and Rane, Roshan Prakash et al.. We address gaps, drawing reasoning using the findings reported by Rao and Ballard. As such, we provide a more comprehensive picture for future research based on predictive coding theory.

<p align="center">
  <img width="750" height="500" src="https://github.com/RoshanRane/Predictive-video-classification/blob/master/PredNet_Vanilla.jpg"></img>
  As shown in Rane, Roshan Prakash et al.
</p>

# Dataset
! [20bn something something dataset](https://20bn.com/datasets/something-something)

#Note
This code was made possible by building upon the contributions by Lotter et al. and Rane, Rohan Prakash, et al. You can find their papers at: ([Rane, Roshan Prakash et al.] (https://arxiv.org/abs/1906.11902)), ([Lotter et al.] (https://arxiv.org/abs/1605.08104)).
# Usage

1.  We ran the simulations on a Windows-10 system and made appropriate changes to the code.
    a. To extract the videos from the downloaded something-something-v2 dataset and split them into test, train, and validation dataset(data.csv) use extract_20bn.py`  

        Ex:- `python3 extract_20bn.py --data_dir /data/videos/something-something-v2/raw --dest_dir /data/videos/something-something-v2/preprocessed --fps 3`

    b. dataset_smthsmth_analysis.ipynb :- performs data analysis on the raw data.  

    c. plot_results.ipynb :- plot results on different evaluation metrics.  

    d. prednet_sth_sth_channel_viz.ipynb :- generates future frame predictions/errors for different layers and channels.
    
    e. plot_results_dict.py :- Plots all of the metrics from results dictionary.
    
    f. pipeline.py :- Runs the whole pipeline
      
      To use the pipeline:
      
        Ex:- `python pipeline.py --csv_path D:/fps12/data.csv --weight_dir H:/PredNet_Critic/ --result_dir H:/PredNet_Critic/ --batch_size 32 --nframes 24 --fps 12 --im_height 48 --im_width 80 --evaluate_model_flag --extra_plots_flag  --samples_per_epoch 1500 --samples_per_epoch_val 300 --nb_epochs 150 --n_chan_layer 32 48 64 128 --a_filt_sizes 5 5 5 5 --ahat_filt_sizes 3 3 3 3 3 --r_filt_sizes 3 3 3 3 4 --layer_loss 1 0 0 0 0 --data_split_ratio 0.002 --model_checkpoint 1 --samples_test 100 --seed 123 --lr_reduce_epoch 75 --train_model_flag --frame_selection "dynamic-fps" --shutdown --early_stopping --early_stopping_patience 80`
  

# Paper and bibtex Reference
[The Performance of PredNet Using Predictive Coding in the Visual Cortex: An Empirical Analysis](https://ieeexplore-ieee-org.ezproxyprod.ucs.louisiana.edu/stamp/stamp.jsp?tp=&arnumber=9599098), Sai Ranganath Mikkilineni, Michael W. Totaro
```
@INPROCEEDINGS{9599098,  author={Mikkilineni, Sai Ranganath and Totaro, Michael Wayne},  booktitle={2021 IEEE 22nd International Conference on Information Reuse and Integration for Data Science (IRI)},   title={The Performance of PredNet Using Predictive Coding in the Visual Cortex: An Empirical Analysis},   year={2021},  volume={},  number={},  pages={408-415},  doi={10.1109/IRI51335.2021.00064}}
```
