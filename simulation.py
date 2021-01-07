#!/usr/bin/env python

from model import *
import csv

def spikes_to_csv(file_name,spike_monitor):
    spike_array = [spike_monitor.i,spike_monitor.t]
    with open(file_name+".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(spike_array)
        
if __name__ == '__main__':

    # read in image to array
    # ims = read_images('mini_test_images')
    # im = ims[1]
    
    start_scope()
    
    visnet = SpikingVisNet()
    visnet.model_summary()

    ims = read_images('images')
    im = ims[7]
    visnet.run_simulation(im,2*second)
    
    spikes_to_csv("layer_0_spikes",visnet.L0_mon)
    spikes_to_csv("layer_1_excitatory_spikes",visnet.L1_exc_mon)
    spikes_to_csv("layer_1_inhibitory_spikes",visnet.L1_inh_mon)
    spikes_to_csv("layer_2_excitatory_spikes",visnet.L2_exc_mon)
    spikes_to_csv("layer_2_inhibitory_spikes",visnet.L2_inh_mon)
    spikes_to_csv("layer_3_excitatory_spikes",visnet.L3_exc_mon)
    spikes_to_csv("layer_3_inhibitory_spikes",visnet.L3_inh_mon)
    spikes_to_csv("layer_4_inhibitory_spikes",visnet.L4_exc_mon)
    spikes_to_csv("layer_4_excitatory_spikes",visnet.L4_inh_mon)

    