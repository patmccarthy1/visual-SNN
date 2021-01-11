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
        self.network = Network(self.L0, self.L1_exc, self.L2_exc, self.L3_exc, self.L4_exc, self.L1_inh, self.L2_inh, self.L3_inh, self.L4_inh, 
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
        
        # Poisson neuron parameters
        tau_m_poisson = 0.9* ms # membrane time constant
        poisson_layer_width = 64     
        N_poisson = poisson_layer_width**2 # can change to np.sqrt(len(flattened_filtered_image)/len(self.filters)) to generalise to different image sizes
        poisson_neuron_spacing = 50*umetre
        v_th_poisson = -10*mV # threshold potential
        v_0_poisson = -70*mV # starting potential
        
        # Poisson neuron equations
        poisson_neurons = '''
        dv/dt = -(v-v_0_poisson)/tau_m_poisson                                        : volt (unless refractory)  # membrane potential
        x = (i%poisson_layer_width)*poisson_neuron_spacing                            : metre  # x position
        y = (int(i/poisson_layer_width))%poisson_layer_width*poisson_neuron_spacing   : metre  # y position
        f = int(i/N_poisson)                                                          : 1      # filter number
        rate                                                                          : Hz     # firing rate to define Poisson distribution
        '''
        
        # LIF neuron parameters
        tau_m_LIF = 1*ms # membrane time constant
        tau_NMDA = 50*ms # excitatory synaptic time constant
        tau_GABA = 100*ms # inhibitory synaptic time constant
        El = -70*mV
        v_th_LIF = -30*mV
        LIF_exc_layer_width = 32 # width of square Layers 1-4 in neurons, e.g. if 128 we will have 128^2 = 16384 neurons in a layer
        N_LIF_exc = LIF_exc_layer_width**2 # number of neurons in a layer
        LIF_exc_neuron_spacing = poisson_neuron_spacing*(poisson_layer_width/LIF_exc_layer_width) # required to assign spatial locations of neurons
        LIF_inh_layer_width = 16                                                                                                    
        N_LIF_inh = LIF_inh_layer_width**2                                                                                                 
        LIF_inh_neuron_spacing = poisson_neuron_spacing*(poisson_layer_width/LIF_inh_layer_width)                                           
        g_GABA = 10*mV
        g_NMDA = 2*mV
        
        # LIF neuron equations (only difference between excitatory and inhibitory is spatial locations)
        LIF_exc_neurons='''dv/dt=-(g_NMDA+g_GABA+El-v)/tau_m_LIF   : volt (unless refractory) # membrane potential
        dg_NMDA/dt=-g_NMDA/tau_NMDA                                : volt  # excitatory/NMDA conductance
        dg_GABA/dt=-g_GABA/tau_GABA                                : volt   # inhibitory/GABA conductance
        x = (i%LIF_exc_layer_width)*LIF_exc_neuron_spacing         : metre  # x position
        y = (int(i/LIF_exc_layer_width))*LIF_exc_neuron_spacing    : metre  # y position
        '''
        LIF_inh_neurons='''dv/dt=-(g_NMDA+g_GABA+El-v)/tau_m_LIF    : volt (unless refractory)
        dg_NMDA/dt=-g_NMDA/tau_NMDA                                : volt
        dg_GABA/dt=-g_GABA/tau_GABA                                : volt
        x = (i%LIF_inh_layer_width)*LIF_inh_neuron_spacing         : metre  
        y = (int(i/LIF_inh_layer_width))*LIF_inh_neuron_spacing    : metre 
        '''

        # Layer 0  
        self.L0 = NeuronGroup(len(self.filters)*N_poisson, poisson_neurons, # create group of Poisson neurons for input layer
                              threshold='rand()*dt < rate*second**2', # multiply rate by second^2 for correct dimensions 
                              reset='v = v_0_poisson', refractory=5*ms, method='euler')

        # Layer 1 
        self.L1_exc = NeuronGroup(N_LIF_exc, LIF_exc_neurons, threshold='v > v_th_LIF', reset='v = El', refractory=0.5*ms, method='euler') # create group of excitatory LIF neurons                       
        self.L1_inh = NeuronGroup(N_LIF_inh, LIF_inh_neurons, threshold='v > v_th_LIF', reset='v = El', refractory=0.5*ms, method='euler') # creatr group of inhibitory LIF neurons                            

        # Layer 2 
        self.L2_exc = NeuronGroup(N_LIF_exc, LIF_exc_neurons, threshold='v > v_th_LIF', reset='v = El', refractory=0.5*ms, method='euler') 
        self.L2_inh = NeuronGroup(N_LIF_inh, LIF_inh_neurons, threshold='v > v_th_LIF', reset='v = El', refractory=0.5*ms, method='euler')

        # Layer 3 
        self.L3_exc = NeuronGroup(N_LIF_exc, LIF_exc_neurons, threshold='v > v_th_LIF', reset='v = El', refractory=0.5*ms, method='euler')                         
        self.L3_inh = NeuronGroup(N_LIF_inh, LIF_inh_neurons, threshold='v > v_th_LIF', reset='v = El', refractory=0.5*ms, method='euler')

        # Layer 4 
        self.L4_exc = NeuronGroup(N_LIF_exc, LIF_exc_neurons, threshold='v > v_th_LIF', reset='v = El', refractory=0.5*ms, method='euler')                         
        self.L4_inh = NeuronGroup(N_LIF_inh, LIF_inh_neurons, threshold='v > v_th_LIF', reset='v = El', refractory=0.5*ms, method='euler')

        # create class variable copies of variables (required for namespace access during simulation)
        self.v_th_poisson = v_th_poisson                                                                                                           
        self.v_0_poisson = v_0_poisson                                                                                                               
        self.tau_m_poisson = tau_m_poisson
        self.tau_m_LIF = tau_m_LIF
        self.v_th_LIF = v_th_LIF
        self.tau_NMDA = tau_NMDA
        self.tau_GABA = tau_GABA
        self.El = El
        self.poisson_layer_width = poisson_layer_width     
        self.N_poisson = N_poisson                                                                                 
        self.poisson_neuron_spacing = poisson_neuron_spacing
        self.LIF_exc_layer_width = LIF_exc_layer_width    
        self.LIF_inh_layer_width = LIF_inh_layer_width                                                                                                                                                                                                    
        self.N_LIF_exc = N_LIF_exc    
        self.N_LIF_inh = N_LIF_inh                                                                                                                                                                        
        self.LIF_exc_neuron_spacing = LIF_exc_neuron_spacing
        self.LIF_inh_neuron_spacing = LIF_inh_neuron_spacing
        
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
        
        # variables to enable creation of randomised connections between layers within topologically corresponding regions
        # num_conn = 2000 # number of connections in given 1um area
        # p_conn = 0.9 # probability of connection between neurons - required to randomise connections, essentially defines sparsity of connections in a region
        # fan_in_radius = np.sqrt(num_conn/(np.pi*p_conn)) * umetre  
        
        
        # create local copies of class variables for local use
        poisson_layer_width = self.poisson_layer_width   
        poisson_neuron_spacing = self.poisson_neuron_spacing
        LIF_exc_layer_width = self.LIF_exc_layer_width
        LIF_inh_layer_width = self.LIF_inh_layer_width                                                                                                                                                                                                                    
        LIF_exc_layer_width = self.LIF_exc_layer_width                                                                                                          
        LIF_inh_layer_width = self.LIF_inh_layer_width                                                                                                          
        LIF_exc_neuron_spacing = self.LIF_exc_neuron_spacing
        LIF_inh_neuron_spacing = self.LIF_inh_neuron_spacing
        tau_m_LIF = self.tau_m_LIF
        tau_NMDA = self.tau_NMDA
        tau_GABA = self.tau_GABA
        
        # fan-in radii
        fan_in_L0_L1 = 1 * poisson_neuron_spacing
        fan_in_L1_L2 = 8 * LIF_exc_neuron_spacing
        fan_in_L2_L3 = 12 * LIF_exc_neuron_spacing
        fan_in_L3_L4 =  16 * LIF_exc_neuron_spacing
        fan_in_L4_L3 = fan_in_L3_L2 = fan_in_L2_L1 = 8 * LIF_exc_neuron_spacing
        fan_in_L1_exc_L1_inh = fan_in_L2_exc_L2_inh = fan_in_L3_exc_L3_inh = fan_in_L4_exc_L4_inh = 1 * LIF_exc_neuron_spacing
        fan_in_L1_inh_L1_exc = fan_in_L2_inh_L2_exc = fan_in_L3_inh_L3_exc = fan_in_L4_inh_L4_exc = 8 * LIF_inh_neuron_spacing

        # connection probabilities
        p_L0_L1 = 1
        p_L1_L2 = 0.5
        p_L2_L3 = 0.22
        p_L3_L4 = 0.12
        p_L4_L3 = p_L3_L2 = p_L2_L1 = 0.02
        p_L1_exc_L1_inh = p_L2_exc_L2_inh = p_L3_exc_L3_inh = p_L4_exc_L4_inh = 1
        p_L1_inh_L1_exc = p_L2_inh_L2_exc = p_L3_inh_L3_exc = p_L4_inh_L4_exc = 0.15
        
        # parameters to enable Gaussian distributed axonal conduction delays
        mean_delay = 0.5                                                                                                                   
        SD_delay = 1                                                                                                                      
        
        # STDP parameters
        taupre = 5*ms
        taupost = 25*ms
        Apre = .05
        Apost = -.04
        
        # EPSPs/IPSPs
        EPSP = 1*mV
        IPSP = -1*mV
        EPSC = EPSP * (tau_NMDA/tau_m_LIF)**(tau_m_LIF/(tau_NMDA-tau_m_LIF))
        IPSC = IPSP * (tau_GABA/tau_m_LIF)**(tau_m_LIF/(tau_GABA-tau_m_LIF))
        Apre = Apre*EPSC
        Apost = Apost*EPSC
        
        w = 10*mV # initial synaptic weight
        
        # STDP equations
        STDP_ODEs = '''
        # ODEs for trace learning rule
        w                                 : volt                 # synaptic weight
        dA_source/dt = -A_source/taupre   : volt (event-driven)  # trace learning rule variable
        dA_target/dt = -A_target/taupost  : volt (event-driven)  # trace learning rule variable
        '''
        # update rule for presynaptic spike for excitatory connections
        STDP_exc_presyn_update = '''
        g_NMDA+=w
        A_source += Apre
        w = clip(w+A_target, 0*volt, EPSC)
        '''
        # update rule for postsynaptic spike for excitatory connections
        STDP_exc_postsyn_update = '''
        A_target += Apost
        w = clip(w+A_source, 0*volt, EPSC)
        '''
        # update rule for postsynaptic spike for inhibitory connections
        STDP_inh_postsyn_update = '''
        g_GABA+=IPSC
        '''
        
        # =============================================================================
        # feedforward connections       
        # =============================================================================
        
        print('Building bottom-up connections')
                
        # Layer 0 to Layer 1 excitatory
        self.Syn_L0_L1_exc = Synapses(self.L0, self.L1_exc, STDP_ODEs, on_pre=STDP_exc_presyn_update, on_post=STDP_exc_postsyn_update) # create synapses with STDP learning rule
        self.Syn_L0_L1_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L0_L1',p=p_L0_L1) # connect lower layer neurons to random upper layer neurons with spatial relation (implicitly selects from random filters)
        self.num_Syn_L0_L1_exc = len(self.Syn_L0_L1_exc.x_pre) # get number of synapses(can use x_pre or x_post to do this)
        self.Syn_L0_L1_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L0_L1_exc)*ms # set Gaussian-ditributed synaptic delay 

        # Layer 1 excitatory to Layer 2 excitatory
        self.Syn_L1_exc_L2_exc = Synapses(self.L1_exc, self.L2_exc, STDP_ODEs, on_pre=STDP_exc_presyn_update, on_post=STDP_exc_postsyn_update)            
        self.Syn_L1_exc_L2_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L1_L2',p=p_L1_L2)                               
        self.num_Syn_L1_exc_L2_exc = len(self.Syn_L1_exc_L2_exc.x_pre)                                                                               
        self.Syn_L1_exc_L2_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L1_exc_L2_exc)*ms                                       
        
        # Layer 2 excitatory to Layer 3 excitatory
        self.Syn_L2_exc_L3_exc = Synapses(self.L2_exc, self.L3_exc, STDP_ODEs, on_pre=STDP_exc_presyn_update, on_post=STDP_exc_postsyn_update)             
        self.Syn_L2_exc_L3_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L2_L3', p=p_L2_L3)                             
        self.num_Syn_L2_exc_L3_exc = len(self.Syn_L2_exc_L3_exc.x_pre)                                                                             
        self.Syn_L2_exc_L3_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L2_exc_L3_exc)*ms                                        
        
        # Layer 3 excitatory to Layer 4 excitatory
        self.Syn_L3_exc_L4_exc = Synapses(self.L3_exc, self.L4_exc, STDP_ODEs, on_pre=STDP_exc_presyn_update, on_post=STDP_exc_postsyn_update)              
        self.Syn_L3_exc_L4_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L3_L4', p=p_L3_L4)                               
        self.num_Syn_L3_exc_L4_exc = len(self.Syn_L3_exc_L4_exc.x_pre)                                                                              
        self.Syn_L3_exc_L4_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L3_exc_L4_exc)*ms                                       
    
        # =============================================================================
        # lateral connections        
        # =============================================================================
    
        print('Building lateral connections')
                
        # Layer 1 excitatory to Layer 1 inhibitory 
        self.Syn_L1_exc_L1_inh = Synapses(self.L1_exc, self.L1_inh, STDP_ODEs, on_pre=STDP_exc_presyn_update, on_post=STDP_exc_postsyn_update)
        self.Syn_L1_exc_L1_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L1_exc_L1_inh',p=p_L1_exc_L1_inh)                               
        self.num_Syn_L1_exc_L1_inh = len(self.Syn_L1_exc_L1_inh.x_pre)                                                                               
        self.Syn_L1_exc_L1_inh.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L1_exc_L1_inh)*ms 
        
        # Layer 1 inhibitory to Layer 1 excitatory 
        self.Syn_L1_inh_L1_exc = Synapses(self.L1_inh, self.L1_exc, model='w:volt', on_post=STDP_inh_postsyn_update)
        self.Syn_L1_inh_L1_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L1_inh_L1_exc',p=p_L1_inh_L1_exc)                               
        self.num_Syn_L1_inh_L1_exc = len(self.Syn_L1_inh_L1_exc.x_pre)                                                                               
        # self.Syn_L1_inh_L1_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L1_inh_L1_exc)*ms 
        
        # Layer 2 excitatory to Layer 2 inhibitory 
        self.Syn_L2_exc_L2_inh = Synapses(self.L2_exc, self.L2_inh, STDP_ODEs, on_pre=STDP_exc_presyn_update, on_post=STDP_exc_postsyn_update)
        self.Syn_L2_exc_L2_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L2_exc_L2_inh',p=p_L2_exc_L2_inh)                               
        self.num_Syn_L2_exc_L2_inh = len(self.Syn_L2_exc_L2_inh.x_pre)                                                                               
        self.Syn_L2_exc_L2_inh.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L2_exc_L2_inh)*ms 
        
        # Layer 2 inhibitory to Layer 2 excitatory 
        self.Syn_L2_inh_L2_exc = Synapses(self.L2_inh, self.L2_exc, model='w:volt', on_post=STDP_inh_postsyn_update)
        self.Syn_L2_inh_L2_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L2_inh_L2_exc',p=p_L2_inh_L2_exc)                               
        self.num_Syn_L2_inh_L2_exc = len(self.Syn_L2_inh_L2_exc.x_pre)                                                                               
        # self.Syn_L2_inh_L2_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L2_inh_L2_exc)*ms 
        
        # Layer 3 excitatory to Layer 3 inhibitory 
        self.Syn_L3_exc_L3_inh = Synapses(self.L3_exc, self.L3_inh, STDP_ODEs,on_pre=STDP_exc_presyn_update, on_post=STDP_exc_postsyn_update)
        self.Syn_L3_exc_L3_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L3_exc_L3_inh',p=p_L3_exc_L3_inh)                               
        self.num_Syn_L3_exc_L3_inh = len(self.Syn_L3_exc_L3_inh.x_pre)                                                                               
        self.Syn_L3_exc_L3_inh.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L3_exc_L3_inh)*ms 
        
        # Layer 3 inhibitory to Layer 3 excitatory 
        self.Syn_L3_inh_L3_exc = Synapses(self.L3_inh, self.L3_exc,model='w:volt', on_post=STDP_inh_postsyn_update)
        self.Syn_L3_inh_L3_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L3_inh_L3_exc',p=p_L3_inh_L3_exc)                               
        self.num_Syn_L3_inh_L3_exc = len(self.Syn_L3_inh_L3_exc.x_pre)                                                                               
        # self.Syn_L3_inh_L3_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L3_inh_L3_exc)*ms

        # Layer 4 excitatory to Layer 4 inhibitory 
        self.Syn_L4_exc_L4_inh = Synapses(self.L4_exc, self.L4_inh, STDP_ODEs, on_pre=STDP_exc_presyn_update, on_post=STDP_exc_postsyn_update)
        self.Syn_L4_exc_L4_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L4_exc_L4_inh',p=p_L4_exc_L4_inh)                               
        self.num_Syn_L4_exc_L4_inh = len(self.Syn_L4_exc_L4_inh.x_pre)                                                                               
        self.Syn_L4_exc_L4_inh.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L4_exc_L4_inh)*ms 
        
        # Layer 4 inhibitory to Layer 4 excitatory 
        self.Syn_L4_inh_L4_exc = Synapses(self.L4_inh, self.L4_exc, model='w:volt', on_post=STDP_inh_postsyn_update)
        self.Syn_L4_inh_L4_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L4_inh_L4_exc',p=p_L4_inh_L4_exc)                               
        self.num_Syn_L4_inh_L4_exc = len(self.Syn_L4_inh_L4_exc.x_pre)                                                                               
        # self.Syn_L4_inh_L4_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L4_inh_L4_exc)*ms

        # =============================================================================
        # feedback connections    
        # =============================================================================
    
        print('Building top-down connections')
            
        # Layer 4 excitatory to Layer 3 excitatory
        self.Syn_L4_exc_L3_exc = Synapses(self.L4_exc, self.L3_exc, STDP_ODEs, on_pre=STDP_exc_presyn_update, on_post=STDP_exc_postsyn_update)              
        self.Syn_L4_exc_L3_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L4_L3', p=p_L4_L3)                               
        self.num_Syn_L4_exc_L3_exc = len(self.Syn_L4_exc_L3_exc.x_pre)                                                                              
        self.Syn_L4_exc_L3_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L4_exc_L3_exc)*ms   
        
        # Layer 3 excitatory to Layer 2 excitatory
        self.Syn_L3_exc_L2_exc = Synapses(self.L3_exc, self.L2_exc, STDP_ODEs, on_pre=STDP_exc_presyn_update, on_post=STDP_exc_postsyn_update)              
        self.Syn_L3_exc_L2_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L3_L2', p=p_L3_L2)                               
        self.num_Syn_L3_exc_L2_exc = len(self.Syn_L3_exc_L2_exc.x_pre)                                                                              
        self.Syn_L3_exc_L2_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L3_exc_L2_exc)*ms   
        
        # Layer 2 excitatory to Layer 1 excitatory
        self.Syn_L2_exc_L1_exc = Synapses(self.L2_exc, self.L1_exc, STDP_ODEs, on_pre=STDP_exc_presyn_update, on_post=STDP_exc_postsyn_update)              
        self.Syn_L2_exc_L1_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_L2_L1', p=p_L2_L1)                               
        self.num_Syn_L2_exc_L1_exc = len(self.Syn_L2_exc_L1_exc.x_pre)                                                                              
        self.Syn_L2_exc_L1_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L2_exc_L1_exc)*ms   
        
        # create class variable copies of variables (required for namespace access during simulation)
        self.taupre = taupre
        self.taupost = taupost
        self.Apre = Apre
        self.Apost = Apost
        self.EPSC = EPSC
        self.IPSC = IPSC

    # internal function to generate Gabor filters to be applied to input image (called inside _gabor_filter)
    def _generate_gabor_filters(self):
        self.filters = []                                                                                                            
        ksize = 4 # kernel size
        phi_list = [0, np.pi/2, np.pi] # phase offset of sinusoid 
        lamda = 5 # wavelength of sinusoid 
        theta_list = [0,np.pi/4, np.pi/2, 3*np.pi/4] # filter orientation
        b = 1.5 # spatial bandwidth in octaves (will be used to determine SD)
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
        self.L0.rate = flattened_filtered_image * 1/255000 * Hz # set firing rates of L0 Poisson neurons equal to outputs of Gabor filters - multiply by a coefficient (10e-8) to get biologically realistic values
        return filtered_image
    
    # =============================================================================
    # external functions
    # =============================================================================

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
        self.network.run(length, namespace={'v_th_poisson': self.v_th_poisson, # run simulations, passing dictionary of necessary parameters into namespace argument (simulation will raise error otherwise)
                                            'v_th_LIF': self.v_th_LIF,
                                            'v_0_poisson': self.v_0_poisson,
                                            'tau_m_poisson': self.tau_m_poisson, 
                                            'tau_m_LIF': self.tau_m_poisson,
                                            'poisson_layer_width': self.poisson_layer_width,
                                            'N_poisson': self.N_poisson,
                                            'poisson_neuron_spacing': self.poisson_neuron_spacing,
                                            'LIF_exc_layer_width': self.LIF_exc_layer_width, 
                                            'LIF_inh_layer_width': self.LIF_inh_layer_width,                                                                                                                                                                                                                 
                                            'N_LIF_exc': self.N_LIF_exc,  
                                            'N_LIF_inh': self.N_LIF_inh,                                                                                                                                                                                                  
                                            'LIF_exc_neuron_spacing': self.LIF_exc_neuron_spacing,
                                            'LIF_inh_neuron_spacing': self.LIF_inh_neuron_spacing,
                                            'taupre': self.taupre,
                                            'taupost': self.taupost,
                                            'Apre': self.Apre,
                                            'Apost': self.Apost,
                                            'EPSC': self.EPSC,
                                            'IPSC': self.IPSC,
                                            'tau_NMDA': self.tau_NMDA,
                                            'tau_GABA': self.tau_GABA,
                                            'El': self.El},
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
