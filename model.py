#!/usr/bin/env python

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import time
from scipy.stats import truncnorm

class SpikingVisNet:
    
    '''
    Instantiating this class builds 4 layer spiking neural network model of the
    primate ventral visual pathway with topologically corresponding STDP synapses 
    between layers and feedforward, lateral and feedback connections.
    ------------------------------------------------------------------------------
    Layer 0: Retina-LGN-V1 complex as Poisson neurons with Gabor filter modulated firing rates
    Layer 1: V2 as excitatory and inhibitory leaky integrate-and-fire neurons
    Layer 2: V4 as excitatory and inhibitory leaky integrate-and-fire neurons
    Layer 3: TEO as excitatory and inhibitory leaky integrate-and-fire neurons
    Layer 4: TE as excitatory and inhibitory leaky integrate-and-fire neurons
    Synapses: Conductance-based synapses with Gaussian-distributed axonal conduction delays and trace learning rule for STDP
    '''
    
    # =============================================================================
    # internal functions
    # =============================================================================
    
    # class constructor
    def __init__(self):
    
        t1 = time.time() # record time of instantiation so can calculate construction time
        self.filters = self._generate_gabor_filters()
        self._build_layers()
        self._connect_layers()
        self._build_spike_monitors()
        self.filtered_images = [] 
        self.network = Network(self.L0, self.L1, self.L1_exc, self.L1_inh, self.L2, self.L2_exc, self.L2_inh, self.L3, 
                               self.L3_exc, self.L3_inh, self.L4, self.L4_exc, self.L4_inh, self.L0_mon, self.L1_exc_mon, 
                               self.L2_exc_mon, self.L3_exc_mon, self.L4_exc_mon, self.L1_inh_mon, self.L2_inh_mon, 
                               self.L3_inh_mon, self.L4_inh_mon, self.Syn_L0_L1_exc, self.Syn_L1_exc_L2_exc, 
                               self.Syn_L2_exc_L3_exc, self.Syn_L3_exc_L4_exc, self.Syn_L1_exc_L1_inh, self.Syn_L2_exc_L2_inh, 
                               self.Syn_L3_exc_L3_inh, self.Syn_L4_exc_L4_inh, self.Syn_L1_inh_L1_exc, self.Syn_L2_inh_L2_exc, 
                               self.Syn_L3_inh_L3_exc, self.Syn_L4_inh_L4_exc, self.Syn_L1_exc_L1_exc, self.Syn_L2_exc_L2_exc,
                               self.Syn_L3_exc_L3_exc, self.Syn_L4_exc_L4_exc, self.Syn_L4_exc_L3_exc, self.Syn_L3_exc_L2_exc, 
                               self.Syn_L2_exc_L1_exc)
        t2 = time.time()
        print("Construction time: %.1f seconds" % (t2 - t1)) 
        
    # internal function to create layers of neurons
    def _build_layers(self):

        print('BUILDING NEURONS')
        
        # =============================================================================
        # parameters        
        # =============================================================================

        # Poisson neuron parameters
        poisson_layer_width = 256     
        N_poisson = poisson_layer_width**2      # number of Poisson neurons in (for one filter) can change to np.sqrt(len(flattened_filtered_image)/len(self.filter s)) to generalise to different image sizes
        poisson_neuron_spacing = 12.5*umetre    # spacing between neurons
        N_filters = len(self.filters)           # number of filters
        N_poisson_total = N_poisson * N_filters # total number of Poisson neurons for all filters
        x_poisson = [(i%poisson_layer_width)*poisson_neuron_spacing for i in range(N_poisson_total)]                       
        y_poisson = [(int(i/poisson_layer_width))%poisson_layer_width*poisson_neuron_spacing for i in range(N_poisson_total)]
        filt_num = [int(i/N_poisson) for i in range(N_poisson_total)]       
            
        # LIF neuron parameters
        LIF_exc_layer_width = 64                                                                    # width of excitatory neuron sublayer in Layers 1-4
        N_LIF_exc = LIF_exc_layer_width**2                                                          # number of neurons in excitatory sublayer 
        LIF_exc_neuron_spacing = poisson_neuron_spacing*(poisson_layer_width/LIF_exc_layer_width)   # required to assign spatial locations of neurons
        LIF_inh_layer_width = 32                                                                    # width of inhibitory neuron sublayer in Layers 1-4                                                                                                    
        N_LIF_inh = LIF_inh_layer_width**2                                                          # number of neurons in inhibitory sublayer                                                                                                 
        LIF_inh_neuron_spacing = poisson_neuron_spacing*(poisson_layer_width/LIF_inh_layer_width)   # required to assign spatial locations of neurons                                 
        x_exc = [(i%LIF_exc_layer_width)*LIF_exc_neuron_spacing for i in range(N_LIF_exc)]          # excitatory sublayer x position vector
        y_exc = [(int(i/LIF_exc_layer_width))*LIF_exc_neuron_spacing for i in range(N_LIF_exc)]     # excitatory sublayer y position vector
        x_inh = [(i%LIF_inh_layer_width)*LIF_inh_neuron_spacing for i in range(N_LIF_inh)]          # inhibitory sublayer x position vector
        y_inh = [(int(i/LIF_inh_layer_width))*LIF_inh_neuron_spacing for i in range(N_LIF_inh)]     # inhibitory sublayer y position vector
        E_l = -50*mV           # leak reversal potential
        g_l = 25*nS            # leak conductance
        E_e = -20*mV           # excitatory synaptic reversal potential
        E_i = -90*mV           # inhibitory synaptic reversal potential
        C_m = 0.214*nF         # membrane capacitance
        tau_e = 1.5*ms         # excitatory synaptic time constant
        tau_i = 5*ms           # inhibitory synaptic time constant
        tau_r = 15*ms           # refractory period
        V_th = -40*mV          # firing threshold
        V_r = -58*mV           # reset potential
        
        # synapse parameters
        g_init = 50*nS           # initial synaptic conductance
        tau_C = 30*ms
        tau_D = 30*ms
        
        # =============================================================================
        # definitions        
        # =============================================================================
       
        # Poisson neuron equations
        poisson_neurons = '''
        idx = i            : 1       # store index of each neuron (for tracking synaptic connections)
        dC/dt = -C/tau_C   : 1       # concentration of glutamate released into synaptic cleft
        dD/dt = -D/tau_D   : 1       # proportion of NMDA receptors unblocked
        x                  : metre   # x position
        y                  : metre   # y position
        f                  : 1       # filter number
        rate               : Hz      # firing rate to define Poisson distribution
        '''
        
        # LIF neuron equations (only difference between excitatory and inhibitory is spatial locations)
        LIF_neurons='''
        idx = i                                                  : 1                        # store index of each neuron (for tracking synaptic connections)
        dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v))/C_m    : volt (unless refractory) # membrane potential (LIF equation)               
        dg_e/dt = -g_e/tau_e                                     : siemens                  # post-synaptic exc. conductance (incremented when excitatory spike arrives at neuron - see synapse equations)
        dg_i/dt = -g_i/tau_i                                     : siemens                  # post-synaptic inh. conductance (incremented when inhibitory spike arrives at neuron - see synapse equations)
        dC/dt = -C/tau_C                                         : 1                        # concentration of glutamate released into synaptic cleft
        dD/dt = -D/tau_D                                         : 1                        # proportion of NMDA receptors unblocked
        x                                                        : metre                    # x position
        y                                                        : metre                    # y position
        '''
        
        # =============================================================================
        # neurons       
        # =============================================================================
        
        # Layer 0  
        print('Layer 0 (Poisson)')
        self.L0 = NeuronGroup(N_poisson_total, poisson_neurons, # create group of Poisson neurons for input layer
                              threshold='rand() < rate*dt', # multiply rate by second^2 for correct dimensions 
                              method='euler')
        self.L0.x = x_poisson
        self.L0.y = y_poisson
        self.L0.f = filt_num
        
        # Layer 1 
        print('Layer 1')
        self.L1 = NeuronGroup(N_LIF_exc + N_LIF_inh, LIF_neurons, threshold='v > V_th', reset='v = V_r', refractory='tau_r', method='euler') # create group of excitatory LIF neurons           
        self.L1.v = 'E_l + rand()*(V_th-E_l)'  # random initial membrane potentials
        self.L1_exc = self.L1[:N_LIF_exc]      # create variable for excitatory neurons
        self.L1_exc.C = 'rand()/2'
        self.L1_exc.D = 'rand()/2'
        self.L1_exc.g_e = 'rand()*g_init'        # random initial excitatory conductances
        self.L1_exc.g_i = 'rand()*g_init'        # random initial inhibitory conductances
        self.L1_exc.x = x_exc                  # excitatory neurons x locations
        self.L1_exc.y = y_exc                  # excitatory neurons y locations
        self.L1_inh = self.L1[N_LIF_exc:]      # create variable for inhibitory neurons
        self.L1_inh.g_e = 'rand()*g_init'        # random initial excitatory conductances
        self.L1_inh.g_i = '0*siemens'          # inhibitory conductance is zero for inhibitory neurons (only neurons in excitatory layers are capable of being inhibited)
        self.L1_inh.x = x_inh                  # inhibitory neurons x locations
        self.L1_inh.y = y_inh                  # inhibitory neurons y locations

        # Layer 2
        print('Layer 2')
        self.L2 = NeuronGroup(N_LIF_exc + N_LIF_inh, LIF_neurons, threshold='v > V_th', reset='v = V_r', refractory='tau_r', method='euler') # create group of excitatory LIF neurons            
        self.L2.v = 'E_l + rand()*(V_th-E_l)'  # random initial membrane potentials
        self.L2_exc = self.L2[:N_LIF_exc]      # create variable for excitatory neurons
        self.L2_exc.C = 'rand()/2'
        self.L2_exc.D = 'rand()/2'
        self.L2_exc.g_e = 'rand()*g_init'        # random initial excitatory conductances
        self.L2_exc.g_i = 'rand()*g_init'        # random initial inhibitory conductances
        self.L2_exc.x = x_exc                  # excitatory neurons x locations
        self.L2_exc.y = y_exc                  # excitatory neurons y locations
        self.L2_inh = self.L2[N_LIF_exc:]      # create variable for inhibitory neurons
        self.L2_inh.g_e = 'rand()*g_init'        # random initial excitatory conductances
        self.L2_inh.g_i = '0*siemens'          # inhibitory conductance is zero for inhibitory neurons (only neurons in excitatory layers are capable of being inhibited)
        self.L2_inh.x = x_inh                  # inhibitory neurons x locations
        self.L2_inh.y = y_inh                  # inhibitory neurons y locations
        
        # Layer 3 
        print('Layer 3')
        self.L3 = NeuronGroup(N_LIF_exc + N_LIF_inh, LIF_neurons, threshold='v > V_th', reset='v = V_r', refractory='tau_r', method='euler') # create group of excitatory LIF neurons            
        self.L3.v = 'E_l + rand()*(V_th-E_l)'  # random initial membrane potentials
        self.L3_exc = self.L3[:N_LIF_exc]      # create variable for excitatory neurons
        self.L3_exc.C = 'rand()/2'
        self.L3_exc.D = 'rand()/2'
        self.L3_exc.g_e = 'rand()*g_init'        # random initial excitatory conductances
        self.L3_exc.g_i = 'rand()*g_init'        # random initial inhibitory conductances
        self.L3_exc.x = x_exc                  # excitatory neurons x locations
        self.L3_exc.y = y_exc                  # excitatory neurons y locations
        self.L3_inh = self.L3[N_LIF_exc:]      # create variable for inhibitory neurons
        self.L3_inh.g_e = 'rand()*g_init'        # random initial excitatory conductances
        self.L3_inh.g_i = '0*siemens'          # inhibitory conductance is zero for inhibitory neurons (only neurons in excitatory layers are capable of being inhibited)
        self.L3_inh.x = x_inh                  # inhibitory neurons x locations
        self.L3_inh.y = y_inh                  # inhibitory neurons y locations

        # Layer 4 
        print('Layer 4')
        self.L4 = NeuronGroup(N_LIF_exc + N_LIF_inh, LIF_neurons, threshold='v > V_th', reset='v = V_r', refractory='tau_r', method='euler') # create group of excitatory LIF neurons           
        self.L4.v = 'E_l + rand()*(V_th-E_l)'  # random initial membrane potentials
        self.L4_exc = self.L4[:N_LIF_exc]      # create variable for excitatory neurons
        self.L4_exc.C = 'rand()/2'
        self.L4_exc.D = 'rand()/2'
        self.L4_exc.g_e = 'rand()*g_init'        # random initial excitatory conductances
        self.L4_exc.g_i = 'rand()*g_init'        # random initial inhibitory conductances
        self.L4_exc.x = x_exc                  # excitatory neurons x locations
        self.L4_exc.y = y_exc                  # excitatory neurons y locations
        self.L4_inh = self.L4[N_LIF_exc:]      # create variable for inhibitory neurons
        self.L4_inh.g_e = 'rand()*g_init'        # random initial excitatory conductances
        self.L4_inh.g_i = '0*siemens'          # inhibitory conductance is zero for inhibitory neurons (only neurons in excitatory layers are capable of being inhibited)
        self.L4_inh.x = x_inh                  # inhibitory neurons x locations
        self.L4_inh.y = y_inh                  # inhibitory neurons y locations

        # ============================================================================================
        # create class variable copies of variables (required for namespace issues during simulation)
        # ============================================================================================

        self.poisson_layer_width = poisson_layer_width     
        self.N_poisson = N_poisson                                                                                 
        self.poisson_neuron_spacing = poisson_neuron_spacing
        self.LIF_exc_layer_width = LIF_exc_layer_width    
        self.LIF_inh_layer_width = LIF_inh_layer_width                                                                                                                                                                                                    
        self.N_LIF_exc = N_LIF_exc    
        self.N_LIF_inh = N_LIF_inh                                                                                                                                                                        
        self.LIF_exc_neuron_spacing = LIF_exc_neuron_spacing
        self.LIF_inh_neuron_spacing = LIF_inh_neuron_spacing
        self.E_l = E_l
        self.g_l = g_l
        self.E_e = E_e
        self.E_i = E_i
        self.C_m = C_m
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.tau_r = tau_r 
        self.V_th = V_th
        self.V_r = V_r 
        self.g_init = g_init
        self.tau_C = tau_C 
        self.tau_D = tau_D 
        
    # internal function to create spike monitors 
    def _build_spike_monitors(self):
        
        print('BUILDING MONITORS')
        
        self.L0_mon = SpikeMonitor(self.L0)                                                                                                 
        self.L1_exc_mon = SpikeMonitor(self.L1_exc)      
        self.L1_inh_mon = SpikeMonitor(self.L1_inh)                                                                                                                                                                            
        self.L2_exc_mon = SpikeMonitor(self.L2_exc)
        self.L2_inh_mon = SpikeMonitor(self.L2_inh)                                                                                                                                                                                  
        self.L3_exc_mon = SpikeMonitor(self.L3_exc)       
        self.L3_inh_mon = SpikeMonitor(self.L3_inh)                                                                                                                                                                       
        self.L4_exc_mon = SpikeMonitor(self.L4_exc)  
        self.L4_inh_mon = SpikeMonitor(self.L4_inh)  

    # internal function to create synapses and connect layers
    def _connect_layers(self):
        
        # =============================================================================
        # parameters
        # =============================================================================

        # create copies of class variables for local use
        poisson_layer_width = self.poisson_layer_width   
        poisson_neuron_spacing = self.poisson_neuron_spacing
        LIF_exc_layer_width = self.LIF_exc_layer_width
        LIF_inh_layer_width = self.LIF_inh_layer_width                                                                                                                                                                                                                    
        LIF_exc_layer_width = self.LIF_exc_layer_width                                                                                                          
        LIF_inh_layer_width = self.LIF_inh_layer_width                                                                                                          
        LIF_exc_neuron_spacing = self.LIF_exc_neuron_spacing
        LIF_inh_neuron_spacing = self.LIF_inh_neuron_spacing
        
        # fan-in radii
        fan_in_L0_L1 = 2 * poisson_neuron_spacing 
        fan_in_L1_L2 = 4 * LIF_exc_neuron_spacing
        fan_in_L2_L3 = 6 * LIF_exc_neuron_spacing
        fan_in_L3_L4 =  8 * LIF_exc_neuron_spacing
        fan_in_L4_L3 = fan_in_L3_L2 = fan_in_L2_L1 = 5 * LIF_exc_neuron_spacing
        fan_in_L1_exc_L1_inh = fan_in_L2_exc_L2_inh = fan_in_L3_exc_L3_inh = fan_in_L4_exc_L4_inh = 5 * LIF_exc_neuron_spacing
        fan_in_L1_inh_L1_exc = fan_in_L2_inh_L2_exc = fan_in_L3_inh_L3_exc = fan_in_L4_inh_L4_exc = 8 * LIF_inh_neuron_spacing
        fan_in_L1_exc_L1_exc = fan_in_L2_exc_L2_exc = fan_in_L3_exc_L3_exc = fan_in_L4_exc_L4_exc = 1 * LIF_inh_neuron_spacing

        # connection probabilities
        p_L0_L1 = 1
        p_L1_L2 = 0.5
        p_L2_L3 = 0.25
        p_L3_L4 = 0.1
        p_L4_L3 = p_L3_L2 = p_L2_L1 = 0.01
        p_L1_exc_L1_inh = p_L2_exc_L2_inh = p_L3_exc_L3_inh = p_L4_exc_L4_inh = 1
        p_L1_inh_L1_exc = p_L2_inh_L2_exc = p_L3_inh_L3_exc = p_L4_inh_L4_exc = 0.5
        p_L1_exc_L1_exc = p_L2_exc_L2_exc = p_L3_exc_L3_exc = p_L4_exc_L4_exc = 0.3
        
        # parameters to enable Gaussian distributed axonal conduction delays
        mean_delay = 20 # mean delay
        sd_delay = 10   # standard deviation of delays
        #         low_delay = 1000  # lower bound on delays
        #         upp_delay = 2000  # upper bound on delays
        
        # synaptic parameters
        rho = 0.01     # synaptic learning rate
        tau_w = 5*ms   # time constant for ODE describing synaptic weight
        eta_exc = 0.04 # scaling factor for increase in synaptic conductance upon spike
        eta_inh = 0.06 # scaling factor for increase in synaptic conductance upon spike
        alpha_C = 0.1  # synaptic glutamate concentration scaling constant
        alpha_D = 0.1  # proportion of unblocked NMDA receptors scaling constant
        w_init = 20*nS # max initial weight
        w_max = 100*nS # max weight
        
        # =============================================================================
        # definitions       
        # =============================================================================
                
        # synapse equations
        STDP_synapse_eqs = '''
        dw/dt = rho*(((1*siemens-w)*C_pre-w*D_post)/tau_w) : siemens (event-driven) # synaptic weight (STDP function - see variables A and B below)            
        '''
        # action upon excitatory presynaptic spike 
        STDP_on_pre_exc = '''
        g_e_post += eta_exc*w
        C_pre += alpha_C*(1-C_pre)
        '''
        # action upon excitatory postsynaptic spike
        STDP_on_post_exc = '''
        D_post += alpha_D*(1-D_post)
        '''
        non_STDP_synapse_eqs = '''
        w : siemens # synaptic weight (fixed as no STDP in synapses involving inhibitory neurons)
        '''
        non_STDP_on_pre_exc = '''
        g_e_post += eta_exc*w     
        '''
        # action upon inhibitory postsynaptic spike 
        on_pre_inh = '''
        g_i_post += eta_inh*w     
        '''
        
        # =============================================================================
        # synapses       
        # =============================================================================
        
        # feedforward connections       
        # -----------------------
        
        print('BUILDING FEEDFORWARD CONNECTIONS')
                
        # Layer 0 to Layer 1 excitatory
        print('Layer 0 to Layer 1 excitatory')
        self.Syn_L0_L1_exc = Synapses(self.L0, self.L1_exc, model=STDP_synapse_eqs, on_pre=STDP_on_pre_exc, on_post=STDP_on_post_exc, method='euler') # create synapses with STDP learning rule
        self.Syn_L0_L1_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L0_L1',p=p_L0_L1) # connect lower layer neurons to random upper layer neurons with spatial relation (implicitly selects from random filters)
        self.num_Syn_L0_L1_exc = self.Syn_L0_L1_exc.N # get number of synapses(can use x_pre or x_post to do this)
        self.Syn_L0_L1_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #'truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L0_L1_exc)*ms' # set truncated Gaussian-ditributed synaptic delay 
        self.Syn_L0_L1_exc.w = 'rand() * 10 * w_init'
        self.affs_per_neuron_Syn_L0_L1_exc = self.Syn_L0_L1_exc.N_incoming_post

        # Layer 1 excitatory to Layer 2 excitatory
        print('Layer 1 excitatory to Layer 2 excitatory')
        self.Syn_L1_exc_L2_exc = Synapses(self.L1_exc, self.L2_exc, model=STDP_synapse_eqs, on_pre=STDP_on_pre_exc, on_post=STDP_on_post_exc, method='euler')            
        self.Syn_L1_exc_L2_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L1_L2',p=p_L1_L2)                               
        self.num_Syn_L1_exc_L2_exc = self.Syn_L1_exc_L2_exc.N                                                                              
        self.Syn_L1_exc_L2_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay,size=self.num_Syn_L1_exc_L2_exc)*ms                                       
        self.Syn_L1_exc_L2_exc.w = 'rand() * w_init'
        self.affs_per_neuron_Syn_L1_exc_L2_exc = self.Syn_L1_exc_L2_exc.N_incoming_post

        # Layer 2 excitatory to Layer 3 excitatory
        print('Layer 2 excitatory to Layer 3 excitatory')
        self.Syn_L2_exc_L3_exc = Synapses(self.L2_exc, self.L3_exc, model=STDP_synapse_eqs, on_pre=STDP_on_pre_exc, on_post=STDP_on_post_exc, method='euler')             
        self.Syn_L2_exc_L3_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L2_L3', p=p_L2_L3)                             
        self.num_Syn_L2_exc_L3_exc = self.Syn_L2_exc_L3_exc.N                                                                             
        self.Syn_L2_exc_L3_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L2_exc_L3_exc)*ms                                        
        self.Syn_L2_exc_L3_exc.w = 'rand() * w_init'
        self.affs_per_neuron_Syn_L2_exc_L3_exc = self.Syn_L2_exc_L3_exc.N_incoming_post

        # Layer 3 excitatory to Layer 4 excitatory
        print('Layer 3 excitatory to Layer 4 excitatory')
        self.Syn_L3_exc_L4_exc = Synapses(self.L3_exc, self.L4_exc, model=STDP_synapse_eqs, on_pre=STDP_on_pre_exc, on_post=STDP_on_post_exc, method='euler')              
        self.Syn_L3_exc_L4_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L3_L4', p=p_L3_L4)                               
        self.num_Syn_L3_exc_L4_exc = self.Syn_L3_exc_L4_exc.N                                                                              
        self.Syn_L3_exc_L4_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L3_exc_L4_exc)*ms                                       
        self.Syn_L3_exc_L4_exc.w = 'rand() * w_init'
        self.affs_per_neuron_Syn_L3_exc_L4_exc = self.Syn_L3_exc_L4_exc.N_incoming_post

        # lateral connections 
        # -------------------
    
        print('BUILDING LATERAL CONNECTIONS')
                
        # Layer 1 excitatory to Layer 1 inhibitory
        print('Layer 1 excitatory to Layer 1 inhibitory')
        self.Syn_L1_exc_L1_inh = Synapses(self.L1_exc, self.L1_inh, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')
        self.Syn_L1_exc_L1_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L1_exc_L1_inh',p=p_L1_exc_L1_inh)                               
        self.num_Syn_L1_exc_L1_inh = self.Syn_L1_exc_L1_inh.N                                                                              
        self.Syn_L1_exc_L1_inh.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L1_exc_L1_inh)*ms 
        self.Syn_L1_exc_L1_inh.w = 'rand() * 10 * w_init'
        self.affs_per_neuron_Syn_L1_exc_L1_inh = self.Syn_L1_exc_L1_inh.N_incoming_post
        
        # Layer 1 inhibitory to Layer 1 excitatory 
        print('Layer 1 inhibitory to Layer 1 excitatory')
        self.Syn_L1_inh_L1_exc = Synapses(self.L1_inh, self.L1_exc, model=non_STDP_synapse_eqs, on_pre=on_pre_inh, method='euler')
        self.Syn_L1_inh_L1_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L1_inh_L1_exc',p=p_L1_inh_L1_exc)                               
        self.num_Syn_L1_inh_L1_exc = self.Syn_L1_inh_L1_exc.N                                                                               
        self.Syn_L1_inh_L1_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L1_inh_L1_exc)*ms 
        self.Syn_L1_inh_L1_exc.w = 'rand() * 10 * w_init'
        self.affs_per_neuron_Syn_L1_inh_L1_exc = self.Syn_L1_inh_L1_exc.N_incoming_post

        # Layer 2 excitatory to Layer 2 inhibitory
        print('Layer 2 excitatory to Layer 2 inhibitory')
        self.Syn_L2_exc_L2_inh = Synapses(self.L2_exc, self.L2_inh, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')
        self.Syn_L2_exc_L2_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L2_exc_L2_inh',p=p_L2_exc_L2_inh)                               
        self.num_Syn_L2_exc_L2_inh = self.Syn_L2_exc_L2_inh.N                                                                               
        self.Syn_L2_exc_L2_inh.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L2_exc_L2_inh)*ms 
        self.Syn_L2_exc_L2_inh.w = 'rand() * 10 * w_init'
        self.affs_per_neuron_Syn_L2_exc_L2_inh = self.Syn_L2_exc_L2_inh.N_incoming_post

        # Layer 2 inhibitory to Layer 2 excitatory
        print('Layer 2 inhibitory to Layer 2 excitatory')
        self.Syn_L2_inh_L2_exc = Synapses(self.L2_inh, self.L2_exc, model=non_STDP_synapse_eqs, on_pre=on_pre_inh, method='euler')
        self.Syn_L2_inh_L2_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L2_inh_L2_exc',p=p_L2_inh_L2_exc)                               
        self.num_Syn_L2_inh_L2_exc = self.Syn_L2_inh_L2_exc.N                                                                              
        self.Syn_L2_inh_L2_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L2_inh_L2_exc)*ms 
        self.Syn_L2_inh_L2_exc.w = 'rand() * 10 * w_init'
        self.affs_per_neuron_Syn_L2_inh_L2_exc = self.Syn_L2_inh_L2_exc.N_incoming_post

        # Layer 3 excitatory to Layer 3 inhibitory
        print('Layer 3 excitatory to Layer 3 inhibitory ')
        self.Syn_L3_exc_L3_inh = Synapses(self.L3_exc, self.L3_inh, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')
        self.Syn_L3_exc_L3_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L3_exc_L3_inh',p=p_L3_exc_L3_inh)                               
        self.num_Syn_L3_exc_L3_inh = self.Syn_L3_exc_L3_inh.N                                                                              
        self.Syn_L3_exc_L3_inh.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L3_exc_L3_inh)*ms 
        self.Syn_L3_exc_L3_inh.w = 'rand() * 10 * w_init'
        self.affs_per_neuron_Syn_L3_exc_L3_inh = self.Syn_L3_exc_L3_inh.N_incoming_post

        # Layer 3 inhibitory to Layer 3 excitatory
        print('Layer 3 inhibitory to Layer 3 excitatory')
        self.Syn_L3_inh_L3_exc = Synapses(self.L3_inh, self.L3_exc, model=non_STDP_synapse_eqs, on_pre=on_pre_inh, method='euler')
        self.Syn_L3_inh_L3_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L3_inh_L3_exc',p=p_L3_inh_L3_exc)                               
        self.num_Syn_L3_inh_L3_exc = len(self.Syn_L3_inh_L3_exc.x_pre)                                                                               
        self.Syn_L3_inh_L3_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L3_inh_L3_exc)*ms
        self.Syn_L3_inh_L3_exc.w = 'rand() * 10 * w_init'
        self.affs_per_neuron_Syn_L3_inh_L3_exc = self.Syn_L3_inh_L3_exc.N_incoming_post

        # Layer 4 excitatory to Layer 4 inhibitory 
        print('Layer 4 excitatory to Layer 4 inhibitory')
        self.Syn_L4_exc_L4_inh = Synapses(self.L4_exc, self.L4_inh, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')
        self.Syn_L4_exc_L4_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L4_exc_L4_inh',p=p_L4_exc_L4_inh)                               
        self.num_Syn_L4_exc_L4_inh = self.Syn_L4_exc_L4_inh.N                                                                               
        self.Syn_L4_exc_L4_inh.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L4_exc_L4_inh)*ms 
        self.Syn_L4_exc_L4_inh.w = 'rand() * 10 * w_init'
        self.affs_per_neuron_Syn_L4_exc_L4_inh = self.Syn_L4_exc_L4_inh.N_incoming_post

        # Layer 4 inhibitory to Layer 4 excitatory
        print('Layer 4 inhibitory to Layer 4 excitatory')
        self.Syn_L4_inh_L4_exc = Synapses(self.L4_inh, self.L4_exc, model=non_STDP_synapse_eqs, on_pre=on_pre_inh, method='euler')
        self.Syn_L4_inh_L4_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L4_inh_L4_exc',p=p_L4_inh_L4_exc)                               
        self.num_Syn_L4_inh_L4_exc = self.Syn_L4_inh_L4_exc.N                                                                               
        self.Syn_L4_inh_L4_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L4_inh_L4_exc)*ms
        self.Syn_L4_inh_L4_exc.w = 'rand() * 10 * w_init'
        self.affs_per_neuron_Syn_L4_inh_L4_exc = self.Syn_L4_inh_L4_exc.N_incoming_post

        # Layer 1 excitatory to Layer 1 excitatory
        print('Layer 1 excitatory to Layer 1 excitatory')
        self.Syn_L1_exc_L1_exc = Synapses(self.L1_exc, self.L1_exc, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')
        self.Syn_L1_exc_L1_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L1_exc_L1_exc',p=p_L1_exc_L1_exc)                               
        self.num_Syn_L1_exc_L1_exc = self.Syn_L1_exc_L1_exc.N
        self.Syn_L1_exc_L1_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L1_exc_L1_exc)*ms
        self.Syn_L1_exc_L1_exc.w = 'rand() * w_init'
        self.affs_per_neuron_Syn_L1_exc_L1_exc = self.Syn_L1_exc_L1_exc.N_incoming_post

        # Layer 2 excitatory to Layer 2 excitatory
        print('Layer 2 excitatory to Layer 2 excitatory')
        self.Syn_L2_exc_L2_exc = Synapses(self.L2_exc, self.L2_exc, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')
        self.Syn_L2_exc_L2_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L2_exc_L2_exc',p=p_L2_exc_L2_exc)                               
        self.num_Syn_L2_exc_L2_exc = self.Syn_L2_exc_L2_exc.N
        self.Syn_L2_exc_L2_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L2_exc_L2_exc)*ms
        self.Syn_L2_exc_L2_exc.w = 'rand() * w_init'
        self.affs_per_neuron_Syn_L2_exc_L2_exc = self.Syn_L2_exc_L2_exc.N_incoming_post
        
        # Layer 3 excitatory to Layer 3 excitatory
        print('Layer 3 excitatory to Layer 3 excitatory')
        self.Syn_L3_exc_L3_exc = Synapses(self.L2_exc, self.L2_exc, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')
        self.Syn_L3_exc_L3_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L3_exc_L3_exc',p=p_L2_exc_L2_exc)                               
        self.num_Syn_L3_exc_L3_exc = self.Syn_L3_exc_L3_exc.N
        self.Syn_L3_exc_L3_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L3_exc_L3_exc)*ms
        self.Syn_L3_exc_L3_exc.w = 'rand() * w_init'
        self.affs_per_neuron_Syn_L3_exc_L3_exc = self.Syn_L3_exc_L3_exc.N_incoming_post
        
        # Layer 4 excitatory to Layer 4 excitatory
        print('Layer 4 excitatory to Layer 4 excitatory')
        self.Syn_L4_exc_L4_exc = Synapses(self.L2_exc, self.L2_exc, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')
        self.Syn_L4_exc_L4_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L4_exc_L4_exc',p=p_L2_exc_L2_exc)                               
        self.num_Syn_L4_exc_L4_exc = self.Syn_L4_exc_L4_exc.N
        self.Syn_L4_exc_L4_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L4_exc_L4_exc)*ms
        self.Syn_L4_exc_L4_exc.w = 'rand() * w_init'
        self.affs_per_neuron_Syn_L4_exc_L4_exc = self.Syn_L4_exc_L4_exc.N_incoming_post
        
        # feedback connections    
        # --------------------
        
        print('BUILDING FEEDBACK CONNECTIONS')
            
        # Layer 4 excitatory to Layer 3 excitatory
        print('Layer 4 excitatory to Layer 3 excitatory')
        self.Syn_L4_exc_L3_exc = Synapses(self.L4_exc, self.L3_exc, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')              
        self.Syn_L4_exc_L3_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L4_L3', p=p_L4_L3)                               
        self.num_Syn_L4_exc_L3_exc = self.Syn_L4_exc_L3_exc.N                                                                            
        self.Syn_L4_exc_L3_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L4_exc_L3_exc)*ms   
        self.Syn_L4_exc_L3_exc.w = 'rand() * w_init'
        self.affs_per_neuron_Syn_L4_exc_L3_exc = self.Syn_L4_exc_L3_exc.N_incoming_post

        # Layer 3 excitatory to Layer 2 excitatory
        print('Layer 3 excitatory to Layer 2 excitatory')
        self.Syn_L3_exc_L2_exc = Synapses(self.L3_exc, self.L2_exc, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')              
        self.Syn_L3_exc_L2_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L3_L2', p=p_L3_L2)                               
        self.num_Syn_L3_exc_L2_exc = self.Syn_L3_exc_L2_exc.N                                                                              
        self.Syn_L3_exc_L2_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L3_exc_L2_exc)*ms   
        self.Syn_L3_exc_L2_exc.w = 'rand() * w_init'
        self.affs_per_neuron_Syn_L3_exc_L2_exc = self.Syn_L3_exc_L2_exc.N_incoming_post

        # Layer 2 excitatory to Layer 1 excitatory
        print('Layer 2 excitatory to Layer 1 excitatory')
        self.Syn_L2_exc_L1_exc = Synapses(self.L2_exc, self.L1_exc, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')              
        self.Syn_L2_exc_L1_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L2_L1', p=p_L2_L1)                               
        self.num_Syn_L2_exc_L1_exc = self.Syn_L2_exc_L1_exc.N                                                                              
        self.Syn_L2_exc_L1_exc.delay = 'sd_delay*randn()*ms + mean_delay*ms' #truncnorm.rvs(low_delay, upp_delay, size=self.num_Syn_L2_exc_L1_exc)*ms   
        self.Syn_L2_exc_L1_exc.w = 'rand() * w_init'
        self.affs_per_neuron_Syn_L2_exc_L1_exc = self.Syn_L2_exc_L1_exc.N_incoming_post

        # ============================================================================================
        # create class variable copies of variables (required for namespace issues during simulation)
        # ============================================================================================

        self.rho = rho
        self.tau_w = tau_w
        self.eta_exc = eta_exc 
        self.eta_inh = eta_inh 
        self.alpha_C = alpha_C 
        self.alpha_D = alpha_D 
        self.w_init = w_init
        self.w_max = w_max 
        self.mean_delay = mean_delay
        self.sd_delay = sd_delay
#         self.upp_delay = upp_delay
#         self.low_delay = low_delay
        
        
    # internal function to generate Gabor filters to be applied to input image (called inside _gabor_filter)
    def _generate_gabor_filters(self):
        self.filters = []                                                                                                            
        ksize = 11 # kernel size
        phi_list = [np.pi/2, np.pi] # phase offset of sinusoid 
        lamda = 5.2 # wavelength of sinusoid 
        theta_list = [0,np.pi/4, np.pi/2, 3*np.pi/4, np.pi] # filter orientation
        b = 1.1 # spatial bandwidth in octaves (will be used to determine SD)
        sigma = lamda*(2**b+1)/np.pi*(2**b-1) * np.sqrt(np.log(2)/2)
        gamma = 0.5 # filter aspect ratio
        for phi in phi_list:
            for theta in theta_list:
                filt = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
                self.filters.append(filt)
        return self.filters
    
    # internal function to apply Gabor filters to SINGLE IMAGE and generate output image for each filter 
    def _image_to_spikes(self, image, filters):
        filtered_image = np.empty([len(image),len(image),len(filters)],dtype=np.float32) # NumPy array to store filtered images (first dimension is input image, second dimension is filters)                                                                                                                                     # iterate through images and filters
        for filt_idx, filt in enumerate(filters):
            filtered = cv2.filter2D(image, cv2.CV_8UC3, filt) # apply filter
            # show image
            # fig, ax = plt.subplots(1,1)
            # ax.imshow(filtered)
            # ax.set_title('Filter {}'.format(filt_idx+1)) # plot filtered images                               
            # plt.axis('off')
            # plt.show()
            filtered_image[:,:,filt_idx] = filtered # add filtered image to array
        self.filtered_images.append(filtered_image)
        flattened_filtered_image = np.ndarray.flatten(filtered_image) # flatten filtered images
        self.L0.rate = flattened_filtered_image * 1/4000 * Hz # set firing rates of L0 Poisson neurons equal to outputs of Gabor filters - multiply by a coefficient (10e-8) to get biologically realistic values
        return filtered_image
    
    # =============================================================================
    # external functions
    # =============================================================================

    def STDP_off(self):
        self.rho = 0
        
    def STDP_on(self,learning_rate):
        self.rho = learning_rate
        
    # function to print out summary of model architecture as a sanity check
    # update this 
    def model_summary(self):
        print('\nSUMMARY\n')
        print(' layer  | neurons  | dimensions  | spacing (um)')
        print('----------------------------------------------')
        print(' 0      | {}   | {}x{}x{}  | {:.2f}'.format(self.L0.N,self.poisson_layer_width,self.poisson_layer_width,len(self.filters),self.poisson_neuron_spacing*10**6,len(self.filters)))
        print(' 1 exc. | {}     | {}x{}       | {:.2f} '.format(self.L1_exc.N,self.LIF_exc_layer_width,self.LIF_exc_layer_width,self.LIF_exc_neuron_spacing*10**6))
        print(' 1 inh. | {}     | {}x{}       | {:.2f} '.format(self.L1_inh.N,self.LIF_inh_layer_width,self.LIF_inh_layer_width,self.LIF_inh_neuron_spacing*10**6))
        print(' 2 exc. | {}     | {}x{}       | {:.2f}'.format(self.L2_exc.N,self.LIF_exc_layer_width,self.LIF_exc_layer_width,self.LIF_exc_neuron_spacing*10**6))
        print(' 2 inh. | {}     | {}x{}       | {:.2f}'.format(self.L2_inh.N,self.LIF_inh_layer_width,self.LIF_inh_layer_width,self.LIF_inh_neuron_spacing*10**6))
        print(' 3 exc. | {}     | {}x{}       | {:.2f}'.format(self.L3_exc.N,self.LIF_exc_layer_width,self.LIF_exc_layer_width,self.LIF_exc_neuron_spacing*10**6))
        print(' 3 inh. | {}     | {}x{}       | {:.2f}'.format(self.L3_inh.N,self.LIF_inh_layer_width,self.LIF_inh_layer_width,self.LIF_inh_neuron_spacing*10**6))
        print(' 4 exc. | {}     | {}x{}       | {:.2f}'.format(self.L4_exc.N,self.LIF_exc_layer_width,self.LIF_exc_layer_width,self.LIF_exc_neuron_spacing*10**6))
        print(' 4 inh. | {}     | {}x{}       | {:.2f}\n'.format(self.L4_inh.N,self.LIF_inh_layer_width,self.LIF_inh_layer_width,self.LIF_inh_neuron_spacing*10**6))
        print('Feedforward connections')
        print(' source | target | total conn\'s | aff\'s per neuron')
        print('----------------------------------------------------')
        print(' 0      | 1 exc. | {}       | {}'.format(self.num_Syn_L0_L1_exc[:],round(np.mean(self.affs_per_neuron_Syn_L0_L1_exc),2)))
        print(' 1 exc. | 2 exc. | {}        | {}'.format(self.num_Syn_L1_exc_L2_exc[:],round(np.mean(self.affs_per_neuron_Syn_L1_exc_L2_exc),2)))
        print(' 2 exc. | 3 exc. | {}       | {}'.format(self.num_Syn_L2_exc_L3_exc[:],round(np.mean(self.affs_per_neuron_Syn_L2_exc_L3_exc),2)))
        print(' 3 exc. | 4 exc. | {}        | {}\n'.format(self.num_Syn_L3_exc_L4_exc[:],round(np.mean(self.affs_per_neuron_Syn_L3_exc_L4_exc),2)))
        print('Lateral connections')
        print(' source | target | total conn\'s | aff\'s per neuron')
        print('----------------------------------------------------')
        print(' 1 exc. | 1 inh. | {}        | {}'.format(self.num_Syn_L1_exc_L1_inh[:],round(np.mean(self.affs_per_neuron_Syn_L1_exc_L1_inh),2)))
        print(' 2 exc. | 2 inh. | {}        | {}'.format(self.num_Syn_L2_exc_L2_inh[:],round(np.mean(self.affs_per_neuron_Syn_L2_exc_L2_inh),2)))
        print(' 3 exc. | 3 inh. | {}        | {}'.format(self.num_Syn_L3_exc_L3_inh[:],round(np.mean(self.affs_per_neuron_Syn_L3_exc_L3_inh),2)))
        print(' 4 exc. | 4 inh. | {}        | {}'.format(self.num_Syn_L4_exc_L4_inh[:],round(np.mean(self.affs_per_neuron_Syn_L4_exc_L4_inh),2)))
        print(' 1 inh. | 1 exc. | {}        | {}'.format(self.num_Syn_L1_inh_L1_exc[:],round(np.mean(self.affs_per_neuron_Syn_L1_inh_L1_exc),2)))
        print(' 2 inh. | 2 exc. | {}        | {}'.format(self.num_Syn_L2_inh_L2_exc[:],round(np.mean(self.affs_per_neuron_Syn_L2_inh_L2_exc),2)))
        print(' 3 inh. | 3 exc. | {}        | {}'.format(self.num_Syn_L3_inh_L3_exc,round(np.mean(self.affs_per_neuron_Syn_L3_inh_L3_exc),2)))
        print(' 4 inh. | 4 exc. | {}        | {}\n'.format(self.num_Syn_L4_inh_L4_exc[:],round(np.mean(self.affs_per_neuron_Syn_L4_inh_L4_exc),2)))
        print('Self connections')
        print(' source | target | total conn\'s | aff\'s per neuron')
        print('----------------------------------------------------')
        print(' 1 exc. | 1 exc. | {}         | {}'.format(self.num_Syn_L1_exc_L1_exc[:],round(np.mean(self.affs_per_neuron_Syn_L1_exc_L1_exc),2))) 
        print(' 2 exc. | 2 exc. | {}         | {}'.format(self.num_Syn_L2_exc_L2_exc[:],round(np.mean(self.affs_per_neuron_Syn_L2_exc_L2_exc),2))) 
        print(' 3 exc. | 3 exc. | {}         | {}'.format(self.num_Syn_L3_exc_L3_exc[:],round(np.mean(self.affs_per_neuron_Syn_L3_exc_L3_exc),2)))
        print(' 4 exc. | 4 exc. | {}         | {}\n'.format(self.num_Syn_L4_exc_L4_exc[:],round(np.mean(self.affs_per_neuron_Syn_L4_exc_L4_exc),2))) 
        print('Feedback connections')
        print(' source | target | total conn\'s | aff\'s per neuron')
        print('----------------------------------------------------')
        print(' 4 exc. | 3 exc. | {}         | {}'.format(self.num_Syn_L4_exc_L3_exc[:],round(np.mean(self.affs_per_neuron_Syn_L4_exc_L3_exc),2)))
        print(' 3 exc. | 2 exc. | {}         | {}'.format(self.num_Syn_L3_exc_L2_exc[:],round(np.mean(self.affs_per_neuron_Syn_L3_exc_L2_exc),2)))
        print(' 2 exc. | 1 exc. | {}         | {}\n'.format(self.num_Syn_L2_exc_L1_exc[:],round(np.mean(self.affs_per_neuron_Syn_L2_exc_L1_exc),2))) 
        
    # function to pass images into model
    def run_simulation(self, image, length):
        filtered_image = self._image_to_spikes(image,self.filters)
        self.network.run(length, namespace={'poisson_layer_width': self.poisson_layer_width, 
                                            'N_poisson': self.N_poisson,                                                                       
                                            'poisson_neuron_spacing': self.poisson_neuron_spacing,
                                            'LIF_inh_layer_width': self.LIF_inh_layer_width,                                                                                                                                        
                                            'N_LIF_exc': self.N_LIF_exc,
                                            'N_LIF_inh': self.N_LIF_inh,                                                                                                                         
                                            'LIF_exc_neuron_spacing': self.LIF_exc_neuron_spacing,
                                            'LIF_inh_neuron_spacing': self.LIF_inh_neuron_spacing,
                                            'E_l': self.E_l,
                                            'g_l': self.g_l,
                                            'E_e': self.E_e,
                                            'E_i': self.E_i,
                                            'C_m': self.C_m,
                                            'tau_e': self.tau_e,
                                            'tau_i': self.tau_i,
                                            'tau_r': self.tau_r,
                                            'V_th': self.V_th,
                                            'V_r': self.V_r,
                                            'g_init': self.g_init,
                                            'rho': self.rho,
                                            'tau_w': self.tau_w,
                                            'eta_exc': self.eta_exc, 
                                            'eta_inh': self.eta_inh,
                                            'tau_C': self.tau_C, 
                                            'tau_D': self.tau_D, 
                                            'alpha_C': self.alpha_C, 
                                            'alpha_D': self.alpha_D,
                                            'w_init': self.w_init,
                                            'w_max': self.w_max,
                                            'mean_delay': self.mean_delay,
                                            'sd_delay': self.sd_delay
#                                             'upp_delay': self.upp_delay,
#                                             'low_delay': self.low_delay
                                            }, 
                                            report='stdout')                                                                    

# function to read images from file and store as arrays which can be passed into model
def read_images(img_dir):
    images = [cv2.imread(file, 0) for file in glob.glob(img_dir+"/*.png")]
    # for image_idx, image in enumerate(images):
    #     fig, ax = plt.subplots(1,1)
    #     ax.imshow(image, cmap='gray')
    #     ax.set_title('Stimulus {}'.format(image_idx+1))
    #     plt.axis('off')
    #     plt.show()
    return images

# function to isolate a set of neurons' spikes (after simulation run to produce raster plots - e.g. if want to only plot 50th to 100th neurons) 
def get_neurons(mon,lower_i,upper_i):
    neuron_set_i = []
    neuron_set_t = []
    for idx, neuron in enumerate(mon.i):
        if lower_i <= neuron <= upper_i:
            neuron_set_i.append(neuron)
            neuron_set_t.append(mon.t[idx])
    return neuron_set_i, neuron_set_t

# function to visualise connectivity
def visualise_connectivity(synapses):
    Ns = len(synapses.source)
    Nt = len(synapses.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=0.5)
    plot(ones(Nt), arange(Nt), 'ok', ms=0.5)
    for i, j in zip(synapses.i, synapses.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(synapses.i, synapses.j, 'ok', ms=0.5)
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')
