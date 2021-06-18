import csv
import numpy as np
import matplotlib.pyplot as plt

# read in data
with open('../output_data/simulation_14/layer_1_excitatory_full_spikes.csv', newline='') as csv_file:
    l1_exc_spike_times = list(csv.reader(csv_file))
with open('../output_data/simulation_14/layer_2_excitatory_full_spikes.csv', newline='') as csv_file:
    l2_exc_spike_times = list(csv.reader(csv_file))
with open('../output_data/simulation_14/layer_3_excitatory_full_spikes.csv', newline='') as csv_file:
    l3_exc_spike_times = list(csv.reader(csv_file))
with open('../output_data/simulation_14/layer_4_excitatory_full_spikes.csv', newline='') as csv_file:
    l4_exc_spike_times= list(csv.reader(csv_file))

f_samp  = 10000  # sampling frequency
T = 32           # length of spike data in seconds
N_neurons = 4096 # number of neurons
l1_exc_spikes = l2_exc_spikes = l3_exc_spikes = l4_exc_spikes = np.zeros([N_neurons,T*f_samp]).astype(int)

for neuron in l1_exc_spike_times[0][:]:
    for time in l1_exc_spike_times[1][:]:
        time_idx = int(float(time)/f_samp)
        l1_exc_spikes[int(neuron),time_idx] = 1
# for neuron in l2_exc_spike_times[0][:]:
#     for time in l2_exc_spike_times[1][:]:
#         time_idx = int(float(time)/f_samp)
#         l2_exc_spikes[int(neuron),time_idx] = 1
# for neuron in l3_exc_spike_times[0][:]:
#     for time in l3_exc_spike_times[1][:]:
#         time_idx = int(float(time)/f_samp)
#         l3_exc_spikes[int(neuron),time_idx] = 1
# for neuron in l4_exc_spike_times[0][:]:
#     for time in l4_exc_spike_times[1][:]:
#         time_idx = int(float(time)/f_samp)
#         l4_exc_spikes[int(neuron),time_idx] = 1

#%% 

count = 0
frame = np.empty([64,64])

for x in range(64):
    for y in range(64):
        frame[x,y] = l1_exc_spikes[count,0]
        count += 1
        
plt.figure(dpi=500)
plt.imshow(frame,cmap='hot')
plt.colorbar(cmap)