import brian2
import csv
import matplotlib.pyplot as plt

def csv_to_raster(file_name):
   with open(file_name+'.csv', 'r') as csv_file:
    data = list(csv.reader(csv_file, delimiter=','))
    spike_idx = [int(idx) for idx in data[0]]
    spike_times = [float(time[:-2]) for time in data[1]] # remove last 2 charactes from times as these will be 'ms' to signify milliseconds
    fig = plt.figure()
    plt.plot(spike_times, spike_idx, '.k', markersize=0.1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index');
    plt.savefig(file_name+'.png',dpi=500)

csv_to_raster('simulation_data/layer_2_excitatory_spikes')