# =============================================================================
# Spiking Neural Network Model of the Primate Ventral Visual System
# -----------------------------------------------------------------------------
# This script runs a simulation by instantiating the model defined in 'model.py'
# and presenting a series of images. 
# -----------------------------------------------------------------------------
# Patrick McCarthy, Imperial College London, 2021
# =============================================================================

# import packages
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import glob
import time
import csv
from brian2 import *
from model import *

# set_device('cpp_standalone') # run in standalone mode 

#%% function definitions
def spikemon_to_raster_exc(spike_monitor):
    fig = plt.figure(dpi=150)
    plt.plot(spike_monitor.t/ms, spike_monitor.i, ',k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.ylim([0,4096])
    plt.savefig(spike_monitor+'.png',dpi=500)

def spikemon_to_raster_inh(spike_monitor):
    fig = plt.figure(dpi=150)
    plt.plot(spike_monitor.t/ms, spike_monitor.i, ',k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.ylim([0,1024])
    plt.savefig(spike_monitor+'.png',dpi=500)
    
def spikemon_to_raster_poisson(spike_monitor):
    fig = plt.figure(dpi=150)
    plt.plot(spike_monitor.t/ms, spike_monitor.i, ',k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.ylim([0,524288])
    plt.savefig(spike_monitor+'.png',dpi=500)
    
def spikes_to_csv(file_name,spike_monitor):
    spike_array = [spike_monitor.i,spike_monitor.t_]
    with open(file_name+".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(spike_array)
        
def weights_to_csv(file_name,synapse_object):
    weight_array = [synapse_object.idx_pre, synapse_object.idx_post, synapse_object.w_]
    with open(file_name+".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(weight_array)
        
def weights_to_csv_poisson(file_name1,file_name2,file_name3,file_name4,file_name5,file_name6,synapse_object):
    idx_pre = [synapse_object.idx_pre]
    x_pre = [synapse_object.x_pre_]
    y_pre = [synapse_object.y_pre_]
    idx_post = [synapse_object.idx_post]
    f_pre = [synapse_object.f_pre]
    w = [synapse_object.w_]
    with open(file_name1+".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(idx_pre)
    with open(file_name2+".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(x_pre) 
    with open(file_name3+".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(y_pre)
    with open(file_name4+".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(idx_post)
    with open(file_name5+".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(f_pre)
    with open(file_name6+".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(w)
        
def rates_to_csv(file_name,neurons_object,spike_monitor,simulation_time):
    N_spikes = [0] * len(neurons_object.i)
    for nrn_idx in spike_monitor.i:
        N_spikes[nrn_idx] += 1 
        rates = [(num_spikes/simulation_time)/Hz for num_spikes in N_spikes]
        spike_and_rate_array = [neurons_object.i, N_spikes, rates]
        with open(file_name+".csv","w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(spike_and_rate_array)

#%% create model

visnet = SpikingVisNet()
visnet.model_summary()

#%% prepare for simulation


start_scope()                                                                          # clear any model variables
ims = [cv2.imread(file, 0) for file in glob.glob("input_data/*.png")]            # read in image set # $HOME/input_data
cwd = os.getcwd()
disp(cwd)
shap = str(shape(ims))
disp(shap)

#%% simulation

# =============================================================================
# training period 
# =============================================================================

visnet.STDP_on(learning_rate=0.0001)
N_epochs = 10

ep = 0
weights_to_csv_poisson("output_data/layer_0_layer_1_exc_weights_8s_idx_pre_epoch_%s"%ep,
                       "output_data/layer_0_layer_1_exc_weights_8s_x_pre_epoch_%s"%ep,
                       "output_data/layer_0_layer_1_exc_weights_8s_y_pre_epoch_%s"%ep,
                       "output_data/layer_0_layer_1_exc_weights_8s_idx_post_epoch_%s"%ep,
                       "output_data/layer_0_layer_1_exc_weights_8s_f_pre_epoch_%s"%ep,
                       "output_data/layer_0_layer_1_exc_weights_8s_w_epoch_%s"%ep,
                       visnet.Syn_L0_L1_exc)
weights_to_csv("output_data/layer_1_exc_layer_2_exc_weights_8s_epoch_%s"%ep,visnet.Syn_L1_exc_L2_exc)
weights_to_csv("output_data/layer_2_exc_layer_3_exc_weights_8s_epoch_%s"%ep,visnet.Syn_L2_exc_L3_exc)
weights_to_csv("output_data/layer_3_exc_layer_4_exc_weights_8s_epoch_%s"%ep,visnet.Syn_L3_exc_L4_exc)
weights_to_csv("output_data/layer_4_exc_layer_4_exc_weights_8s_epoch_%s"%ep,visnet.Syn_L4_exc_L4_exc)
    
disp('TRAINING')
for ep in range(N_epochs):
    print('epoch '+str(ep)+' of '+str(N_epochs))
    count = 0 # keep track of index of image being presented
    for im_num in range(16):
        
        count += 1
        disp('image '+str(count))

        im = ims[im_num]
        plt.figure
        plt.imshow(im)
        plt.savefig("output_data/im_"+str(im_num)+".png")

        visnet.run_simulation(im,1*second)
        disp('done')

        
    # save weights
    disp('saving weights')
    # IDEA: SCRIPT TO PUT ALL DATA INTO ONE BIG FILE - PANDAS DATAFRAMES
    weights_to_csv_poisson("output_data/layer_0_layer_1_exc_weights_8s_idx_pre_epoch_%s"%(ep+1),
                           "output_data/layer_0_layer_1_exc_weights_8s_x_pre_epoch_%s"%(ep+1),
                           "output_data/layer_0_layer_1_exc_weights_8s_y_pre_epoch_%s"%(ep+1),
                           "output_data/layer_0_layer_1_exc_weights_8s_idx_post_epoch_%s"%(ep+1),
                           "output_data/layer_0_layer_1_exc_weights_8s_f_pre_epoch_%s"%(ep+1),
                           "output_data/layer_0_layer_1_exc_weights_8s_w_epoch_%s"%(ep+1),
                           visnet.Syn_L0_L1_exc)
    weights_to_csv("output_data/layer_1_exc_layer_2_exc_weights_8s_epoch_%s"%(ep+1),visnet.Syn_L1_exc_L2_exc)
    weights_to_csv("output_data/layer_2_exc_layer_3_exc_weights_8s_epoch_%s"%(ep+1),visnet.Syn_L2_exc_L3_exc)
    weights_to_csv("output_data/layer_3_exc_layer_4_exc_weights_8s_epoch_%s"%(ep+1),visnet.Syn_L3_exc_L4_exc)
    weights_to_csv("output_data/layer_4_exc_layer_4_exc_weights_8s_epoch_%s"%(ep+1),visnet.Syn_L4_exc_L4_exc)
    disp('done')

# =============================================================================
# test period
# =============================================================================

visnet.STDP_on(learning_rate=0.001)

disp('TESTING')

count = 0 # keep track of index of image being presented
for im_num in range(16):
    
    count +=1
    disp('image '+str(count))
    
    im = ims[im_num]
    visnet.run_simulation(im,1*second)
    
# save spikes
disp('saving spike data')
spikes_to_csv("output_data/layer_0_full_spikes",visnet.L0_mon)
spikes_to_csv("output_data/layer_1_excitatory_full_spikes",visnet.L1_exc_mon)
spikes_to_csv("output_data/layer_1_inhibitory_full_spikes",visnet.L1_inh_mon)
spikes_to_csv("output_data/layer_2_excitatory_full_spikes",visnet.L2_exc_mon)
spikes_to_csv("output_data/layer_2_inhibitory_full_spikes",visnet.L2_inh_mon)
spikes_to_csv("output_data/layer_3_excitatory_full_spikes",visnet.L3_exc_mon)
spikes_to_csv("output_data/layer_3_inhibitory_full_spikes",visnet.L3_inh_mon)
spikes_to_csv("output_data/layer_4_excitatory_full_spikes",visnet.L4_exc_mon)
spikes_to_csv("output_data/layer_4_inhibitory_full_spikes",visnet.L4_inh_mon)
disp('done')

        
            
