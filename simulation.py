#!/usr/bin/env python
# coding: utf-8

# **Simulation Runner**

# Install and import packages

# In[1]:


get_ipython().system('pip install --user brian2')
get_ipython().system('pip install --user brian2tools')
get_ipython().system('pip install --user --upgrade pip setuptools wheel')
get_ipython().system('pip install --user opencv-python')


# In[2]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import glob
import time
import csv
from brian2 import *
from model import *


# Function definitions

# In[3]:


def spikemon_to_raster(spike_monitor):
    fig = plt.figure(dpi=150)
    plt.plot(spike_monitor.t/ms, spike_monitor.i, ',k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    # plt.savefig(spike_monitor+'.png',dpi=500)


# In[4]:


def spikes_to_csv(file_name,spike_monitor):
    spike_array = [spike_monitor.i,spike_monitor.t_]
    with open(file_name+".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(spike_array)


# In[5]:


def weights_to_csv(file_name,synapse_object):
    weight_array = [synapse_object.i, synapse_object.x_pre_, synapse_object.y_pre_, synapse_object.x_post_, synapse_object.y_post_, synapse_object.w_]
    with open(file_name+".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(weight_array)


# In[ ]:


def weights_to_csv_poisson(file_name,synapse_object):
    weight_array = [synapse_object.i, synapse_object.x_pre_, synapse_object.y_pre_, synapse_object.x_post_, synapse_object.y_post_, synapse_object.f_pre, synapse_object.w_]
    with open(file_name+".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(weight_array)


# In[6]:


def rates_to_csv(file_name,neurons_object,spike_monitor,simulation_time):
    N_spikes = [0] * len(neurons_object.i)
    for nrn_idx in spike_monitor.i:
        N_spikes[nrn_idx] += 1
        rates = [(num_spikes/simulation_time)/Hz for num_spikes in N_spikes]
        spike_and_rate_array = [neurons_object.i, N_spikes, rates]
        with open(file_name+".csv","w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(spike_and_rate_array)


# Create network

# In[7]:


visnet = SpikingVisNet()
visnet.model_summary()


# In[8]:


start_scope()


# Train network

# In[ ]:


ims = read_images('input_data/n4p2') # read in image set

# save initial weights to CSV
weights_to_csv("output_data/layer_0_layer_1_exc_weights_0s".format(idx),visnet.Syn_L0_L1_exc)
weights_to_csv("output_data/layer_1_exc_layer_1_exc_weights_0s".format(idx),visnet.Syn_L1_exc_L1_inh)
weights_to_csv("output_data/layer_1_exc_layer_1_inh_weights_0s".format(idx),visnet.Syn_L1_exc_L1_inh)
weights_to_csv("output_data/layer_1_inh_layer_1_exc_weights_0s".format(idx),visnet.Syn_L1_inh_L1_exc)
weights_to_csv("output_data/layer_1_exc_layer_2_exc_weights_0s".format(idx),visnet.Syn_L1_exc_L2_exc)
weights_to_csv("output_data/layer_2_exc_layer_2_exc_weights_0s".format(idx),visnet.Syn_L2_exc_L2_inh)
weights_to_csv("output_data/layer_2_exc_layer_2_inh_weights_0s".format(idx),visnet.Syn_L2_exc_L2_inh)
weights_to_csv("output_data/layer_2_inh_layer_2_exc_weights_0s".format(idx),visnet.Syn_L2_inh_L2_exc)
weights_to_csv("output_data/layer_2_exc_layer_3_exc_weights_0s".format(idx),visnet.Syn_L2_exc_L3_exc)
weights_to_csv("output_data/layer_2_exc_layer_1_exc_weights_0s".format(idx),visnet.Syn_L2_exc_L1_exc)
weights_to_csv("output_data/layer_3_exc_layer_3_exc_weights_0s".format(idx),visnet.Syn_L3_exc_L3_inh)
weights_to_csv("output_data/layer_3_exc_layer_3_inh_weights_0s".format(idx),visnet.Syn_L3_exc_L3_inh)
weights_to_csv("output_data/layer_3_inh_layer_3_exc_weights_0s".format(idx),visnet.Syn_L3_inh_L3_exc)
weights_to_csv("output_data/layer_3_exc_layer_4_exc_weights_0s".format(idx),visnet.Syn_L3_exc_L4_exc)
weights_to_csv("output_data/layer_3_exc_layer_2_exc_weights_0s".format(idx),visnet.Syn_L3_exc_L2_exc)
weights_to_csv("output_data/layer_4_exc_layer_4_exc_weights_0s".format(idx),visnet.Syn_L4_exc_L4_inh)
weights_to_csv("output_data/layer_4_exc_layer_4_inh_weights_0s".format(idx),visnet.Syn_L4_exc_L4_inh)
weights_to_csv("output_data/layer_4_inh_layer_4_exc_weights_0s".format(idx),visnet.Syn_L4_inh_L4_exc)
weights_to_csv("output_data/layer_4_exc_layer_3_exc_weights_0s".format(idx),visnet.Syn_L4_exc_L3_exc)

# present one image at a time, saving weights of STDP synapses to a CSV every 20 seconds
for idx, im in enumerate(ims):
    # create plot of original image
    plt.figure(figsize=[7,5]) 
    plt.imshow(im,cmap='gray', vmin=0, vmax=255) # this line creates the image using the pre-defined sub axes
    plt.title('Stimulus 1')
    for i in range(5):
        time = (i+1)*20
        visnet.run_simulation(im,20*second)
        weights_to_csv("output_data/layer_0_layer_1_exc_weights_im{}_{}s".format(idx),visnet.Syn_L0_L1_exc)
        weights_to_csv("output_data/layer_1_exc_layer_2_exc_weights_im{}_{}s".format(idx),visnet.Syn_L1_exc_L2_exc)
        weights_to_csv("output_data/layer_2_exc_layer_3_exc_weights_im{}_{}s".format(idx),visnet.Syn_L2_exc_L3_exc)
        weights_to_csv("output_data/layer_3_exc_layer_4_exc_weights_im{}_{}s".format(idx),visnet.Syn_L3_exc_L4_exc)


# Test network

# In[ ]:


visnet.STDP_off()


# In[ ]:


start_scope()


# In[ ]:


for im in ims:
    # create plot of original image
    plt.figure(figsize=[7,5])
    plt.imshow(im,cmap='gray', vmin=0, vmax=255) # this line creates the image using the pre-defined sub axes
    visnet.run_simulation(im,2*second)


# Generate plots to visualise spikes

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


spikemon_to_raster(visnet.L0_mon)


# In[ ]:


spikemon_to_raster(visnet.L1_exc_mon)


# In[ ]:


spikemon_to_raster(visnet.L1_inh_mon)


# In[ ]:


spikemon_to_raster(visnet.L2_exc_mon)


# In[ ]:


spikemon_to_raster(visnet.L2_inh_mon)


# In[ ]:


spikemon_to_raster(visnet.L3_exc_mon)


# In[ ]:


spikemon_to_raster(visnet.L3_inh_mon)


# In[ ]:


spikemon_to_raster(visnet.L4_exc_mon)


# In[ ]:


spikemon_to_raster(visnet.L4_inh_mon)


# Save spikes

# In[ ]:


spikes_to_csv("output_data/layer_0_spikes",visnet.L0_mon)


# In[ ]:


spikes_to_csv("output_data/layer_1_excitatory_spikes",visnet.L1_exc_mon)


# In[ ]:


spikes_to_csv("output_data/layer_1_inhibitory_spikes",visnet.L1_inh_mon)


# In[ ]:


spikes_to_csv("output_data/layer_2_excitatory_spikes",visnet.L2_exc_mon)


# In[ ]:


spikes_to_csv("output_data/layer_2_inhibitory_spikes",visnet.L2_inh_mon)


# In[ ]:


spikes_to_csv("output_data/layer_3_excitatory_spikes",visnet.L3_exc_mon)


# In[ ]:


spikes_to_csv("output_data/layer_3_inhibitory_spikes",visnet.L3_inh_mon)


# In[ ]:


spikes_to_csv("output_data/layer_4_excitatory_spikes",visnet.L4_exc_mon)


# In[ ]:


spikes_to_csv("output_data/layer_4_inhibitory_spikes",visnet.L4_inh_mon)

