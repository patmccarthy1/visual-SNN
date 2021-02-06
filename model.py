#!/usr/bin/env python

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import time

class SpikingVisNet:
    
    '''
    Instantiating this class builds 4 layer spiking neural network model of the
    primate ventral visual pathway with topologically corresponding STDP synapses 
    between layers and feedforward, lateral and feedback connections.
    
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
        self.network = Network(self.L0, self.L1, self.L1_exc, self.L1_inh, self.L2, self.L2_exc, self.L2_inh, self.L3, self.L3_exc, self.L3_inh, self.L4, self.L4_exc, self.L4_inh, 
                               self.L0_mon, self.L1_exc_mon, self.L2_exc_mon, self.L3_exc_mon, self.L4_exc_mon, self.L1_inh_mon, self.L2_inh_mon, self.L3_inh_mon, self.L4_inh_mon,
                               self.Syn_L0_L1_exc, self.Syn_L1_exc_L2_exc, self.Syn_L2_exc_L3_exc, self.Syn_L3_exc_L4_exc,
                               self.Syn_L1_exc_L1_inh, self.Syn_L2_exc_L2_inh, self.Syn_L3_exc_L3_inh, self.Syn_L4_exc_L4_inh,
                               self.Syn_L1_inh_L1_exc, self.Syn_L2_inh_L2_exc, self.Syn_L3_inh_L3_exc, self.Syn_L4_inh_L4_exc,
                               self.Syn_L4_exc_L3_exc, self.Syn_L3_exc_L2_exc, self.Syn_L2_exc_L1_exc)
        t2 = time.time()
        print("Construction time: %.1f seconds" % (t2 - t1)) 
        
    # internal function to create layers of neurons
    def _build_layers(self):

        print('Building neurons')
        
        # =============================================================================
        # parameters        
        # =============================================================================

        # Poisson neuron parameters
        poisson_layer_width = 256     
        N_poisson = poisson_layer_width**2  # number of Poisson neurons in (for one filter) can change to np.sqrt(len(flattened_filtered_image)/len(self.filter s)) to generalise to different image sizes
        poisson_neuron_spacing = 12.5*umetre  # spacing between neurons
        tau_m_poisson = 20*ms               # membrane time constant
        tau_r_poisson = 5*ms                # refractory period
        V_th_poisson = -20.1*mV               # threshold potential
        V_0_poisson = -20*mV                
        
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
        E_l = -74*mV           # leak reversal potential
        g_l = 25*nS            # leak conductance
        E_e = 10*mV            # excitatory synaptic reversal potential
        E_i = -100*mV           # inhibitory synaptic reversal potential
        C_m = 0.5*nF             # membrane capacitance
        tau_e = 5*ms          # excitatory synaptic time constant
        tau_i = 5*ms          # inhibitory synaptic time constant
        tau_r = 2*ms           # refractory period
        V_th = -53*mV          # firing threshold
        V_r = -80*mV              # reset potential
        
        # synapse parameters
        cond = 1*nS          # for setting of initial synaptic conductance
        
        tau_C = 25*ms
        tau_D = 25*ms
        
        # =============================================================================
        # definitions        
        # =============================================================================
       
        # Poisson neuron equations
        poisson_neurons = '''
        dC/dt = -C/tau_C                  : 1  # concentration of glutamate released into synaptic cleft
        dD/dt = -D/tau_D                  : 1  # proportion of NMDA receptors unblocked
        x = (i%poisson_layer_width)*poisson_neuron_spacing                            : metre                     # x position
        y = (int(i/poisson_layer_width))%poisson_layer_width*poisson_neuron_spacing   : metre                     # y position
        f = int(i/N_poisson)                                                          : 1                         # filter number
        rate                                                                          : Hz                        # firing rate to define Poisson distribution
        '''
        
        # LIF neuron equations (only difference between excitatory and inhibitory is spatial locations)
        LIF_neurons='''
        dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v))/C_m    : volt (unless refractory) # membrane potential (LIF equation)               
        dg_e/dt = -g_e/tau_e                                     : siemens                  # post-synaptic exc. conductance (will be incremented when excitatory spike arrives at neuron - see synapse equations)
        dg_i/dt = -g_i/tau_i                                     : siemens                  # post-synaptic inh. conductance (will be incremented when inhibitory spike arrives at neuron - see synapse equations)
        dC/dt = -C/tau_C                  : 1  # concentration of glutamate released into synaptic cleft
        dD/dt = -D/tau_D                  : 1  # proportion of NMDA receptors unblocked
        x                                                        : metre                    # x position
        y                                                        : metre                    # y position
        '''
        
        # =============================================================================
        # neurons       
        # =============================================================================
        
        # Layer 0  
        self.L0 = NeuronGroup(len(self.filters)*N_poisson, poisson_neurons, # create group of Poisson neurons for input layer
                              threshold='rand() < rate*dt', # multiply rate by second^2 for correct dimensions 
                              method='euler')
        # self.L0.v = 'V_0_poisson + rand()*(V_th_poisson-V_0_poisson)'  # random initial membrane potentials
        
        # Layer 1 
        self.L1 = NeuronGroup(N_LIF_exc + N_LIF_inh, LIF_neurons, threshold='v > V_th', reset='v = V_r', refractory='tau_r', method='euler') # create group of excitatory LIF neurons                       
        self.L1.v = 'E_l + rand()*(V_th-E_l)'  # random initial membrane potentials
        self.L1_exc = self.L1[:N_LIF_exc]      # create variable for excitatory neurons
        self.L1_exc.C = 'rand()/2'
        self.L1_exc.D = 'rand()/2'
        self.L1_exc.g_e = 'rand()*cond'        # random initial excitatory conductances
        self.L1_exc.g_i = 'rand()*cond'        # random initial inhibitory conductances
        self.L1_exc.x = x_exc                  # excitatory neurons x locations
        self.L1_exc.y = y_exc                  # excitatory neurons y locations
        self.L1_inh = self.L1[N_LIF_exc:]      # create variable for inhibitory neurons
        self.L1_inh.g_e = 'rand()*cond'        # random initial excitatory conductances
        self.L1_inh.g_i = '0*siemens'                  # inhibitory conductance is zero for inhibitory neurons (only neurons in excitatory layers are capable of being inhibited)
        self.L1_inh.x = x_inh                  # inhibitory neurons x locations
        self.L1_inh.y = y_inh                  # inhibitory neurons y locations
        
        # Layer 2 
        self.L2 = NeuronGroup(N_LIF_exc + N_LIF_inh, LIF_neurons, threshold='v > V_th', reset='v = V_r', refractory='tau_r', method='euler') # create group of excitatory LIF neurons                       
        self.L2.v = 'E_l + rand()*(V_th-E_l)'  # random initial membrane potentials
        self.L2_exc = self.L2[:N_LIF_exc]      # create variable for excitatory neurons
        self.L2_exc.C = 'rand()/2'
        self.L2_exc.D = 'rand()/2'
        self.L2_exc.g_e = 'rand()*cond'        # random initial excitatory conductances
        self.L2_exc.g_i = 'rand()*cond'        # random initial inhibitory conductances
        self.L2_exc.x = x_exc                  # excitatory neurons x locations
        self.L2_exc.y = y_exc                  # excitatory neurons y locations
        self.L2_inh = self.L2[N_LIF_exc:]      # create variable for inhibitory neurons
        self.L2_inh.g_e = 'rand()*cond'        # random initial excitatory conductances
        self.L2_inh.g_i = '0*siemens'                  # inhibitory conductance is zero for inhibitory neurons (only neurons in excitatory layers are capable of being inhibited)
        self.L2_inh.x = x_inh                  # inhibitory neurons x locations
        self.L2_inh.y = y_inh                  # inhibitory neurons y locations

        # Layer 3 
        self.L3 = NeuronGroup(N_LIF_exc + N_LIF_inh, LIF_neurons, threshold='v > V_th', reset='v = V_r', refractory='tau_r', method='euler') # create group of excitatory LIF neurons                       
        self.L3.v = 'E_l + rand()*(V_th-E_l)'  # random initial membrane potentials
        self.L3_exc = self.L3[:N_LIF_exc]      # create variable for excitatory neurons
        self.L3_exc.C = 'rand()/2'
        self.L3_exc.D = 'rand()/2'
        self.L3_exc.g_e = 'rand()*cond'        # random initial excitatory conductances
        self.L3_exc.g_i = 'rand()*cond'        # random initial inhibitory conductances
        self.L3_exc.x = x_exc                  # excitatory neurons x locations
        self.L3_exc.y = y_exc                  # excitatory neurons y locations
        self.L3_inh = self.L3[N_LIF_exc:]      # create variable for inhibitory neurons
        self.L3_inh.g_e = 'rand()*cond'        # random initial excitatory conductances
        self.L3_inh.g_i = '0*siemens'                  # inhibitory conductance is zero for inhibitory neurons (only neurons in excitatory layers are capable of being inhibited)
        self.L3_inh.x = x_inh                  # inhibitory neurons x locations
        self.L3_inh.y = y_inh                  # inhibitory neurons y locations
        
        # Layer 4 
        self.L4 = NeuronGroup(N_LIF_exc + N_LIF_inh, LIF_neurons, threshold='v > V_th', reset='v = V_r', refractory='tau_r', method='euler') # create group of excitatory LIF neurons                       
        self.L4.v = 'E_l + rand()*(V_th-E_l)'  # random initial membrane potentials
        self.L4_exc = self.L4[:N_LIF_exc]      # create variable for excitatory neurons
        self.L4_exc.C = 'rand()/2'
        self.L4_exc.D = 'rand()/2'
        self.L4_exc.g_e = 'rand()*cond'        # random initial excitatory conductances
        self.L4_exc.g_i = 'rand()*cond'        # random initial inhibitory conductances
        self.L4_exc.x = x_exc                  # excitatory neurons x locations
        self.L4_exc.y = y_exc                  # excitatory neurons y locations
        self.L4_inh = self.L4[N_LIF_exc:]      # create variable for inhibitory neurons
        self.L4_inh.g_e = 'rand()*cond'        # random initial excitatory conductances
        self.L4_inh.g_i = '0*siemens'                  # inhibitory conductance is zero for inhibitory neurons (only neurons in excitatory layers are capable of being inhibited)
        self.L4_inh.x = x_inh                  # inhibitory neurons x locations
        self.L4_inh.y = y_inh                  # inhibitory neurons y locations

        # ============================================================================================
        # create class variable copies of variables (required for namespace issues during simulation)
        # ============================================================================================
                                                                                                      
        self.tau_m_poisson = tau_m_poisson
        self.tau_r_poisson = tau_r_poisson
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
        self.cond = cond
        self.tau_C = tau_C 
        self.tau_D = tau_D 
        
    # internal function to create spike monitors 
    def _build_spike_monitors(self):
        
        print('Building spike monitors')
        
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
        fan_in_L0_L1 = 3 * poisson_neuron_spacing
        fan_in_L1_L2 = 3 * LIF_exc_neuron_spacing
        fan_in_L2_L3 = 3 * LIF_exc_neuron_spacing
        fan_in_L3_L4 =  3 * LIF_exc_neuron_spacing
        fan_in_L4_L3 = fan_in_L3_L2 = fan_in_L2_L1 = 1 * LIF_exc_neuron_spacing
        fan_in_L1_exc_L1_inh = fan_in_L2_exc_L2_inh = fan_in_L3_exc_L3_inh = fan_in_L4_exc_L4_inh = 1 * LIF_exc_neuron_spacing
        fan_in_L1_inh_L1_exc = fan_in_L2_inh_L2_exc = fan_in_L3_inh_L3_exc = fan_in_L4_inh_L4_exc = 8 * LIF_inh_neuron_spacing

        # connection probabilities
        p_L0_L1 = 1
        p_L1_L2 = 0.1
        p_L2_L3 = 0.1
        p_L3_L4 = 0.1
        p_L4_L3 = p_L3_L2 = p_L2_L1 = 0.02
        p_L1_exc_L1_inh = p_L2_exc_L2_inh = p_L3_exc_L3_inh = p_L4_exc_L4_inh = 1
        p_L1_inh_L1_exc = p_L2_inh_L2_exc = p_L3_inh_L3_exc = p_L4_inh_L4_exc = 1
        
        # parameters to enable Gaussian distributed axonal conduction delays
        mean_delay = 5                                                                                                                   
        SD_delay = 3                                                                                                                      
                
        # synaptic parameters
        rho = 0.1       # synaptic learning rate
        tau_w = 10*ms  # time constant for ODE describing synaptic weight
        eta_exc = 0.01      # scaling factor for increase in synaptic conductance upon spike
        eta_inh = 10      # scaling factor for increase in synaptic conductance upon spike
        alpha_C = 0.1   # synaptic glutamate concentration scaling constant
        alpha_D = 0.1   # proportion of unblocked NMDA receptors scaling constant
        gmax = 50*nS    # max synaptic conductance
        
        # =============================================================================
        # definitions       
        # =============================================================================
        
        # rho*(((1-w)*C-w*D)/tau_w
        
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
        
        print('Building bottom-up connections')
                
        # Layer 0 to Layer 1 excitatory
        self.Syn_L0_L1_exc = Synapses(self.L0, self.L1_exc, model=STDP_synapse_eqs, on_pre=STDP_on_pre_exc, on_post=STDP_on_post_exc, method='euler') # create synapses with STDP learning rule
        self.Syn_L0_L1_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L0_L1',p=p_L0_L1) # connect lower layer neurons to random upper layer neurons with spatial relation (implicitly selects from random filters)
        self.num_Syn_L0_L1_exc = len(self.Syn_L0_L1_exc.x_pre) # get number of synapses(can use x_pre or x_post to do this)
        self.Syn_L0_L1_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L0_L1_exc)*ms # set Gaussian-ditributed synaptic delay 
        self.Syn_L0_L1_exc.w = 'rand()*gmax'
        
        # Layer 1 excitatory to Layer 2 excitatory
        self.Syn_L1_exc_L2_exc = Synapses(self.L1_exc, self.L2_exc, model=STDP_synapse_eqs, on_pre=STDP_on_pre_exc, on_post=STDP_on_post_exc, method='euler')            
        self.Syn_L1_exc_L2_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L1_L2',p=p_L1_L2)                               
        self.num_Syn_L1_exc_L2_exc = len(self.Syn_L1_exc_L2_exc.x_pre)                                                                               
        self.Syn_L1_exc_L2_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L1_exc_L2_exc)*ms                                       
        self.Syn_L1_exc_L2_exc.w = 'rand() * gmax'

        # Layer 2 excitatory to Layer 3 excitatory
        self.Syn_L2_exc_L3_exc = Synapses(self.L2_exc, self.L3_exc, model=STDP_synapse_eqs, on_pre=STDP_on_pre_exc, on_post=STDP_on_post_exc, method='euler')             
        self.Syn_L2_exc_L3_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L2_L3', p=p_L2_L3)                             
        self.num_Syn_L2_exc_L3_exc = len(self.Syn_L2_exc_L3_exc.x_pre)                                                                             
        self.Syn_L2_exc_L3_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L2_exc_L3_exc)*ms                                        
        self.Syn_L2_exc_L3_exc.w = 'rand() * gmax'

        # Layer 3 excitatory to Layer 4 excitatory
        self.Syn_L3_exc_L4_exc = Synapses(self.L3_exc, self.L4_exc, model=STDP_synapse_eqs, on_pre=STDP_on_pre_exc, on_post=STDP_on_post_exc, method='euler')              
        self.Syn_L3_exc_L4_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L3_L4', p=p_L3_L4)                               
        self.num_Syn_L3_exc_L4_exc = len(self.Syn_L3_exc_L4_exc.x_pre)                                                                              
        self.Syn_L3_exc_L4_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L3_exc_L4_exc)*ms                                       
        self.Syn_L3_exc_L4_exc.w = 'rand() * gmax'

        # lateral connections 
        # -------------------
    
        print('Building lateral connections')
                
        # Layer 1 excitatory to Layer 1 inhibitory 
        self.Syn_L1_exc_L1_inh = Synapses(self.L1_exc, self.L1_inh, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')
        self.Syn_L1_exc_L1_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L1_exc_L1_inh',p=p_L1_exc_L1_inh)                               
        self.num_Syn_L1_exc_L1_inh = len(self.Syn_L1_exc_L1_inh.x_pre)                                                                               
        # self.Syn_L1_exc_L1_inh.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L1_exc_L1_inh)*ms 
        self.Syn_L1_exc_L1_inh.w = 'rand() * gmax'
            
        # Layer 1 inhibitory to Layer 1 excitatory 
        self.Syn_L1_inh_L1_exc = Synapses(self.L1_inh, self.L1_exc, model=non_STDP_synapse_eqs, on_pre=on_pre_inh, method='euler')
        self.Syn_L1_inh_L1_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L1_inh_L1_exc',p=p_L1_inh_L1_exc)                               
        self.num_Syn_L1_inh_L1_exc = len(self.Syn_L1_inh_L1_exc.x_pre)                                                                               
        self.Syn_L1_inh_L1_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L1_inh_L1_exc)*ms 
        self.Syn_L1_inh_L1_exc.w = 'rand() * gmax'

        # Layer 2 excitatory to Layer 2 inhibitory 
        self.Syn_L2_exc_L2_inh = Synapses(self.L2_exc, self.L2_inh, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')
        self.Syn_L2_exc_L2_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L2_exc_L2_inh',p=p_L2_exc_L2_inh)                               
        self.num_Syn_L2_exc_L2_inh = len(self.Syn_L2_exc_L2_inh.x_pre)                                                                               
        self.Syn_L2_exc_L2_inh.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L2_exc_L2_inh)*ms 
        self.Syn_L2_exc_L2_inh.w = 'rand() * gmax'

        # Layer 2 inhibitory to Layer 2 excitatory 
        self.Syn_L2_inh_L2_exc = Synapses(self.L2_inh, self.L2_exc, model=non_STDP_synapse_eqs, on_pre=on_pre_inh, method='euler')
        self.Syn_L2_inh_L2_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L2_inh_L2_exc',p=p_L2_inh_L2_exc)                               
        self.num_Syn_L2_inh_L2_exc = len(self.Syn_L2_inh_L2_exc.x_pre)                                                                               
        # self.Syn_L2_inh_L2_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L2_inh_L2_exc)*ms 
        self.Syn_L2_inh_L2_exc.w = 'rand() * gmax'

        # Layer 3 excitatory to Layer 3 inhibitory 
        self.Syn_L3_exc_L3_inh = Synapses(self.L3_exc, self.L3_inh, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')
        self.Syn_L3_exc_L3_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L3_exc_L3_inh',p=p_L3_exc_L3_inh)                               
        self.num_Syn_L3_exc_L3_inh = len(self.Syn_L3_exc_L3_inh.x_pre)                                                                               
        self.Syn_L3_exc_L3_inh.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L3_exc_L3_inh)*ms 
        self.Syn_L3_exc_L3_inh.w = 'rand() * gmax'

        # Layer 3 inhibitory to Layer 3 excitatory 
        self.Syn_L3_inh_L3_exc = Synapses(self.L3_inh, self.L3_exc, model=non_STDP_synapse_eqs, on_pre=on_pre_inh, method='euler')
        self.Syn_L3_inh_L3_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L3_inh_L3_exc',p=p_L3_inh_L3_exc)                               
        self.num_Syn_L3_inh_L3_exc = len(self.Syn_L3_inh_L3_exc.x_pre)                                                                               
        self.Syn_L3_inh_L3_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L3_inh_L3_exc)*ms
        self.Syn_L3_inh_L3_exc.w = 'rand() * gmax'

        # Layer 4 excitatory to Layer 4 inhibitory 
        self.Syn_L4_exc_L4_inh = Synapses(self.L4_exc, self.L4_inh, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')
        self.Syn_L4_exc_L4_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L4_exc_L4_inh',p=p_L4_exc_L4_inh)                               
        self.num_Syn_L4_exc_L4_inh = len(self.Syn_L4_exc_L4_inh.x_pre)                                                                               
        self.Syn_L4_exc_L4_inh.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L4_exc_L4_inh)*ms 
        self.Syn_L4_exc_L4_inh.w = 'rand() * gmax'

        # Layer 4 inhibitory to Layer 4 excitatory 
        self.Syn_L4_inh_L4_exc = Synapses(self.L4_inh, self.L4_exc, model=non_STDP_synapse_eqs, on_pre=on_pre_inh, method='euler')
        self.Syn_L4_inh_L4_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L4_inh_L4_exc',p=p_L4_inh_L4_exc)                               
        self.num_Syn_L4_inh_L4_exc = len(self.Syn_L4_inh_L4_exc.x_pre)                                                                               
        self.Syn_L4_inh_L4_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L4_inh_L4_exc)*ms
        self.Syn_L4_inh_L4_exc.w = 'rand() * gmax'

        # feedback connections    
        # --------------------
        
        print('Building top-down connections')
            
        # Layer 4 excitatory to Layer 3 excitatory
        self.Syn_L4_exc_L3_exc = Synapses(self.L4_exc, self.L3_exc, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')              
        self.Syn_L4_exc_L3_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L4_L3', p=p_L4_L3)                               
        self.num_Syn_L4_exc_L3_exc = len(self.Syn_L4_exc_L3_exc.x_pre)                                                                              
        self.Syn_L4_exc_L3_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L4_exc_L3_exc)*ms   
        self.Syn_L4_exc_L3_exc.w = 'rand() * gmax'
        
        # Layer 3 excitatory to Layer 2 excitatory
        self.Syn_L3_exc_L2_exc = Synapses(self.L3_exc, self.L2_exc, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')              
        self.Syn_L3_exc_L2_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L3_L2', p=p_L3_L2)                               
        self.num_Syn_L3_exc_L2_exc = len(self.Syn_L3_exc_L2_exc.x_pre)                                                                              
        self.Syn_L3_exc_L2_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L3_exc_L2_exc)*ms   
        self.Syn_L3_exc_L2_exc.w = 'rand() * gmax'
        
        # Layer 2 excitatory to Layer 1 excitatory
        self.Syn_L2_exc_L1_exc = Synapses(self.L2_exc, self.L1_exc, model=non_STDP_synapse_eqs, on_pre=non_STDP_on_pre_exc, method='euler')              
        self.Syn_L2_exc_L1_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L2_L1', p=p_L2_L1)                               
        self.num_Syn_L2_exc_L1_exc = len(self.Syn_L2_exc_L1_exc.x_pre)                                                                              
        self.Syn_L2_exc_L1_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L2_exc_L1_exc)*ms   
        self.Syn_L2_exc_L1_exc.w = 'rand() * gmax'
        
        # ============================================================================================
        # create class variable copies of variables (required for namespace issues during simulation)
        # ============================================================================================

        self.rho = rho
        self.tau_w = tau_w
        self.eta_exc = eta_exc 
        self.eta_inh = eta_inh 
        self.alpha_C = alpha_C 
        self.alpha_D = alpha_D 
        self.gmax = gmax 
        
    # internal function to generate Gabor filters to be applied to input image (called inside _gabor_filter)
    def _generate_gabor_filters(self):
        self.filters = []                                                                                                            
        ksize = 10 # kernel size
        phi_list = [np.pi/2, 3*np.pi/4, np.pi, np.pi/2, 3*np.pi/4, np.pi, np.pi/2, 3*np.pi/4, np.pi] # phase offset of sinusoid 
        lamda = 5 # wavelength of sinusoid 
        theta_list = [0, np.pi/2, np.pi, 3*np.pi/2] # filter orientation
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
        self.L0.rate = flattened_filtered_image * 5/12750 * Hz # set firing rates of L0 Poisson neurons equal to outputs of Gabor filters - multiply by a coefficient (10e-8) to get biologically realistic values
        return filtered_image
    
    # =============================================================================
    # external functions
    # =============================================================================

    def STDP_off(self):
        self.rho = 0
        
    # function to print out summary of model architecture as a sanity check
    def model_summary(self):
        print('Layers\n')
        print(' layer | neurons | dimensions  | spacing (um)\n')
        print('----------------------------------------------\n')
        print(' 0      | {}   | {}x{}x{} | {:.2f}\n'.format(self.L0.N,self.poisson_layer_width,self.poisson_layer_width,len(self.filters),self.poisson_neuron_spacing*10**6,len(self.filters)))
        print(' 1 exc. | {}     | {}x{}      | {:.2f} \n'.format(self.L1_exc.N,self.LIF_exc_layer_width,self.LIF_exc_layer_width,self.LIF_exc_neuron_spacing*10**6))
        print(' 1 inh. | {}     | {}x{}      | {:.2f} \n'.format(self.L1_inh.N,self.LIF_inh_layer_width,self.LIF_inh_layer_width,self.LIF_inh_neuron_spacing*10**6))
        print(' 2 exc. | {}     | {}x{}      | {:.2f}\n'.format(self.L2_exc.N,self.LIF_exc_layer_width,self.LIF_exc_layer_width,self.LIF_exc_neuron_spacing*10**6))
        print(' 2 inh. | {}     | {}x{}      | {:.2f}\n'.format(self.L2_inh.N,self.LIF_inh_layer_width,self.LIF_inh_layer_width,self.LIF_inh_neuron_spacing*10**6))
        print(' 3 exc. | {}     | {}x{}      | {:.2f}\n'.format(self.L3_exc.N,self.LIF_exc_layer_width,self.LIF_exc_layer_width,self.LIF_exc_neuron_spacing*10**6))
        print(' 3 inh. | {}     | {}x{}      | {:.2f}\n'.format(self.L3_inh.N,self.LIF_inh_layer_width,self.LIF_inh_layer_width,self.LIF_inh_neuron_spacing*10**6))
        print(' 4 exc. | {}     | {}x{}      | {:.2f}\n\n'.format(self.L4_exc.N,self.LIF_exc_layer_width,self.LIF_exc_layer_width,self.LIF_exc_neuron_spacing*10**6))
        print(' 4 inh. | {}     | {}x{}      | {:.2f}\n\n'.format(self.L4_inh.N,self.LIF_inh_layer_width,self.LIF_inh_layer_width,self.LIF_inh_neuron_spacing*10**6))
        print('Bottom-up connections\n')
        print(' source | target | connections\n')
        print('-------------------------------\n')
        print(' 0      | 1 exc. | {}\n'.format(self.num_Syn_L0_L1_exc))
        print(' 1 exc. | 2 exc. | {}\n'.format(self.num_Syn_L1_exc_L2_exc))
        print(' 2 exc. | 3 exc. | {}\n'.format(self.num_Syn_L2_exc_L3_exc))
        print(' 3 exc. | 4 exc. | {}\n'.format(self.num_Syn_L3_exc_L4_exc))
        print('Lateral connections\n')
        print(' source | target | connections\n')
        print('-------------------------------\n')
        print(' 1 exc. | 1 inh. | {}\n'.format(self.num_Syn_L1_exc_L1_inh))
        print(' 2 exc. | 2 inh. | {}\n'.format(self.num_Syn_L2_exc_L2_inh))
        print(' 3 exc. | 3 inh. | {}\n'.format(self.num_Syn_L3_exc_L3_inh))
        print(' 4 exc. | 4 inh. | {}\n'.format(self.num_Syn_L4_exc_L4_inh))
        print(' 1 inh. | 1 exc. | {}\n'.format(self.num_Syn_L1_inh_L1_exc))
        print(' 2 inh. | 2 exc. | {}\n'.format(self.num_Syn_L2_inh_L2_exc))
        print(' 3 inh. | 3 exc. | {}\n'.format(self.num_Syn_L3_inh_L3_exc))
        print(' 4 inh. | 4 exc. | {}\n'.format(self.num_Syn_L4_inh_L4_exc)) 
        print('Top-down connections\n')
        print(' source | target | connections\n')
        print('-------------------------------\n')
        print(' 4 exc. | 3 exc. | {}\n'.format(self.num_Syn_L4_exc_L3_exc))
        print(' 3 exc. | 2 exc. | {}\n'.format(self.num_Syn_L3_exc_L2_exc))
        print(' 2 exc. | 1 exc. | {}\n'.format(self.num_Syn_L2_exc_L1_exc)) 
        
    # function to pass images into model - EVENTUALLY REPLACE WITH TRAIN AND TEST FUNCTIONS WHERE STDP IS ON AND OFF, RESPECITVELY
    def run_simulation(self, image, length):
        filtered_image = self._image_to_spikes(image,self.filters)
        self.network.run(length, namespace={'tau_m_poisson': self.tau_m_poisson,
                                            'tau_r_poisson': self.tau_r_poisson,
                                            'poisson_layer_width': self.poisson_layer_width, 
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
                                            'cond': self.cond,
                                            'rho': self.rho,
                                            'tau_w': self.tau_w,
                                            'eta_exc': self.eta_exc, 
                                            'eta_inh': self.eta_inh,
                                            'tau_C': self.tau_C, 
                                            'tau_D': self.tau_D, 
                                            'alpha_C': self.alpha_C, 
                                            'alpha_D': self.alpha_D,
                                            'gmax': self.gmax}, 
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
