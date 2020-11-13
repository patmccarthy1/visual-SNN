# Patrick McCarthy
# A Spiking Neural Network Model of the Primate Ventral Visual Pathway using the Brian 2 simulator
# Imperial College London
# Copyright © 2020

from brian2 import *
import numpy as np
import math

from brian2 import NeuronGroup

start_scope()

# read in input image


# variables and parameters for Layer 0 (Gabor filter layer)


# variables and parameters for neurons in Layers 1-4
layer_width = 10                                                                                                        # width of Layers 1-4 in neurons, e.g. if 128 we will have 128^2 = 16384 neurons in a layer
N = layer_width**2
neuron_spacing = 50*umetre                                                                                              # required to assign spatial locations of neurons
v_th = 0                                                                                                                # threshold potential
v_0 = 0                                                                                                                 # starting potential
tau_m = 0                                                                                                               # membrane time constant
x_locs = []                                                                                                             # list which will be used to define x locations of neurons
y_locs = []                                                                                                             # list which will be used to define y locations of neurons
for x in range(layer_width):                                                                                            # assign locations such that [x,y] = [0,0],[0,1],[0,2]...[0,layer_width],[1,0],[1,1],[1,2]...[layer_width-1,layer_width],[layer_width,layer_width], e.g. for layer width 128, we would have locations [x,y] = [0,0],[0,1],[0,2]...[0,127],[1,0],[1,1],[1,2]...[126,127],[127,127]
    for y in range(layer_width):
        x_locs.append(x)
        y_locs.append(y)
x_locs = x_locs*neuron_spacing
y_locs = y_locs*neuron_spacing
# variables to enable creation of randomised connections between layers within topologically corresponding regions
num_conn =  math.ceil(layer_width/10)**2                                                                                # number of connections from layer to a single neuron in next layer - 1/10 of layer width, rounding up to nearest integer and then squaring e.g for layer width 128, we would have 13^2 = 169 connections to each postsynaptic neuron
p_conn = 0.5                                                                                                            # probability of connection between neurons - required to randomise connections
neighbourhood_width  = math.sqrt(num_conn/p_conn)*umetre                                                                # define width of square neighbourhood from which to randomly select neurons to connect in each layer - approx. 1/10 of layer width, rounding up to nearest integer
# parameters to enable Gaussian distributed axonal conduction delays
mean_delay = 0                                                                                                          # mean for Gaussian distribution to draw conduction delays from
SD_delay = 1                                                                                                            # SD for Gaussian distribution to draw conduction delays from

# variables and parameters for STDP (trace learning rule)
taupre = taupost = 20*ms
wmax = 0.01
Apre = 0.01
Apost = -Apre*taupre/taupost*1.05

# equations for neurons in Layers 1-4
LIF_neurons = '''
dv/dt = -(v-v_0)/tau_m  : volt   # membrane potential
x                       : metre # x position
y                       : metre # y position
'''

# equations for STDP (trace learning rule)
STDP_ODEs = '''
# ODEs for trace learning rule
w                                : 1
dapre/dt = -apre/taupre       : 1 (event-driven)
dapost/dt = -apost/taupost    : 1 (event-driven)
'''
STDP_pre_update = '''
# update rule for presynaptic spike
v_post += w
apre += Apre
w = clip(w+apost, 0, wmax)
'''
STDP_post_update = '''
# update rule for postsynaptic spike
apost += Apost
w = clip(w+apre, 0, wmax)
'''

# Layer 0 (Gabor filter layer)
def gaborToPoisson():


# Layer 1 (V2)
V2 = NeuronGroup(N, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                                 # create group of LIF neurons
V2.x = x_locs                                                                                                           # define spatial locations of V2 neurons
V2.y = y_locs

# Layer 2 (V4)
V4 = NeuronGroup(N, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                                 # create group of LIF neurons
V4.x = x_locs                                                                                                           # define spatial locations of V4 neurons
V4.y = y_locs
Syn_V2_V4 = Synapses(V2, V4, STDP_ODEs, on_pre=STDP_pre_update, on_post=STDP_post_update)                               # create synapses with STDP learning rule
Syn_V2_V4.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < neighbourhood_width', p=p_conn)                          # connect Layer 1 (V2) neurons to each Layer 2 (V4) neuron (by connecting with p_conn in neighbourhood of neighbourhood_width^2 neurons)
num_Syn_V2_V4 = len(Syn_V2_V4.x_pre)                                                                                    # get number of synapses(can use x_pre or x_post to do this)
Syn_V2_V4.delay = np.random.normal(mean_delay, SD_delay, num_Syn_V2_V4)*ms                                              # set Gaussian-ditributed synaptic delay between

# Layer 3 (TEO)
TEO = NeuronGroup(N, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                                # create group of LIF neurons
TEO.x = x_locs                                                                                                          # define spatial locations of TEO neurons
TEO.y = y_locs
Syn_V4_TEO = Synapses(V4, TEO, STDP_ODEs, on_pre=STDP_pre_update, on_post=STDP_post_update)                             # create synapses with STDP learning rule
Syn_V4_TEO.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < neighbourhood_width', p=p_conn)                         # connect Layer 2 (V4) neurons to each Layer 3 (TEO) neuron (by connecting with p_conn in neighbourhood of neighbourhood_width^2 neurons)
num_Syn_V4_TEO = len(Syn_V4_TEO.x_pre)                                                                                  # get number of synapses(can use x_pre or x_post to do this)
Syn_V4_TEO.delay = np.random.normal(mean_delay, SD_delay, num_Syn_V4_TEO) * ms

# Layer 4 (TE)
TE = NeuronGroup(N, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                                 # create group of LIF neurons
TE.x = x_locs                                                                                                           # define spatial locations of TE neurons
TE.y = y_locs
Syn_TEO_TE = Synapses(TEO, TE, STDP_ODEs, on_pre=STDP_pre_update, on_post=STDP_post_update)
Syn_TEO_TE.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < neighbourhood_width', p=p_conn)                         # connect Layer 3 (TEO) neurons to each Layer 4 (TE) neuron (by connecting with p_conn in neighbourhood of neighbourhood_width^2 neurons)
num_Syn_TEO_TE = len(Syn_TEO_TE.x_pre)                                                                                  # get number of synapses(can use x_pre or x_post to do this)
Syn_TEO_TE.delay = np.random.normal(mean_delay, SD_delay, num_Syn_TEO_TE)*ms
