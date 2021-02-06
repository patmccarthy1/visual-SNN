import numpy as np

def calc_prob(fan_in_radius_neurons, neuron_spacing, num_conns_per_neuron):
    radius_m = fan_in_radius_neurons*neuron_spacing
    A = np.pi*radius_m**2
    possible_conns = A/neuron_spacing**2
    probability = num_conns_per_neuron/possible_conns
    return probability
    
p = calc_prob(1,12.5e-6,30)