from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

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
    '''
    
    # =============================================================================
    # internal functions
    # =============================================================================
    
    # class constructor
    def __init__(self):
    
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

        
    # internal function to create and connect model layers upon class instantiation (called inside __init__)
    def _build_layers(self):
    
        v_th = -0.01 # threshold potential
        v_0 = -0.07 # starting potential
        
        # Poisson neuron parameters
        tau_m_poisson = 10 * ms 
        poisson_layer_width = 16     
        N_poisson = poisson_layer_width**2 # can change to np.sqrt(len(flattened_filtered_image)/len(self.filters)) to generalise to different image sizes
        poisson_neuron_spacing = 12.5*umetre
        
        # equations for neurons in L0
        poisson_neurons = '''
        dv/dt = -(v-v_0)/tau_m_poisson                                                : 1      # membrane potential
        x = (i%poisson_layer_width)*poisson_neuron_spacing                            : metre  # x position
        y = (int(i/poisson_layer_width))%poisson_layer_width*poisson_neuron_spacing   : metre  # y position
        f = int(i/N_poisson)                                                          : 1      # filter number
        rate                                                                          : Hz     # firing rate to define Poisson distribution
        '''
        
        # LIF neuron parameters
        tau_m_LIF = 5 * ms 
        LIF_exc_layer_width = 8 # width of Layers 1-4 in neurons, e.g. if 128 we will have 128^2 = 16384 neurons in a layer
        N_LIF_exc = LIF_exc_layer_width**2 # number of neurons in a layer
        LIF_exc_neuron_spacing = poisson_neuron_spacing*(poisson_layer_width/LIF_exc_layer_width) # required to assign spatial locations of neurons
        LIF_inh_layer_width = 4                                                                                                    
        N_LIF_inh = LIF_inh_layer_width**2                                                                                                 
        LIF_inh_neuron_spacing = poisson_neuron_spacing*(poisson_layer_width/LIF_inh_layer_width)                                           

        # parameters for STDP (trace learning rule)
        taupre = taupost = 20*ms
        wmax = 0.5
        Apre = 0.5
        Apost = -Apre*taupre/taupost*1.05
        
        # equations for neurons in Layers 1-4
        LIF_exc_neurons = '''
        dv/dt = -(v-v_0)/tau_m_LIF                               : 1      # membrane potential
        x = (i%LIF_exc_layer_width)*LIF_exc_neuron_spacing       : metre  # x position
        y = (int(i/LIF_exc_layer_width))*LIF_exc_neuron_spacing  : metre  # y position
        '''
        LIF_inh_neurons = '''
        dv/dt = -(v-v_0)/tau_m_LIF                                : 1      # membrane potential
        x = (i%LIF_inh_layer_width)*LIF_inh_neuron_spacing       : metre  # x position
        y = (int(i/LIF_inh_layer_width))*LIF_inh_neuron_spacing   : metre  # y position
        '''

        # Layer 0  
        self.L0 = NeuronGroup(len(self.filters)*N_poisson, poisson_neurons, # create group of Poisson neurons for input layer
                              threshold='rand()*dt/second < rate*second', 
                              reset='v = v_0', method='euler')

        # Layer 1 
        self.L1_exc = NeuronGroup(N_LIF_exc, LIF_exc_neurons, threshold='v > v_th', reset='v = v_0', method='euler') # create group of excitatory LIF neurons                       
        self.L1_inh = NeuronGroup(N_LIF_inh, LIF_inh_neurons, threshold='v > v_th', reset='v = v_0', method='euler') # creatr group of inhibitory LIF neurons                            

        # Layer 2 
        self.L2_exc = NeuronGroup(N_LIF_exc, LIF_exc_neurons, threshold='v > v_th', reset='v = v_0', method='euler') 
        self.L2_inh = NeuronGroup(N_LIF_inh, LIF_inh_neurons, threshold='v > v_th', reset='v = v_0', method='euler')

        # Layer 3 
        self.L3_exc = NeuronGroup(N_LIF_exc, LIF_exc_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                         
        self.L3_inh = NeuronGroup(N_LIF_inh, LIF_inh_neurons, threshold='v > v_th', reset='v = v_0', method='euler')

        # Layer 4 
        self.L4_exc = NeuronGroup(N_LIF_exc, LIF_exc_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                         
        self.L4_inh = NeuronGroup(N_LIF_inh, LIF_inh_neurons, threshold='v > v_th', reset='v = v_0', method='euler')

        # create class variable copies of variables (required for namespace access for simulation)
        self.v_th = v_th                                                                                                           
        self.v_0 = v_0                                                                                                               
        self.tau_m_poisson = tau_m_poisson
        self.tau_m_LIF = tau_m_LIF
        self.poisson_layer_width = poisson_layer_width     
        self.N_poisson = N_poisson                                                                                 
        self.poisson_neuron_spacing = poisson_neuron_spacing
        self.LIF_exc_layer_width = LIF_exc_layer_width    
        self.LIF_inh_layer_width = LIF_inh_layer_width                                                                                                                                                                                                    
        self.N_LIF_exc = N_LIF_exc    
        self.N_LIF_inh = N_LIF_inh                                                                                                                                                                        
        self.LIF_exc_neuron_spacing = LIF_exc_neuron_spacing
        self.LIF_inh_neuron_spacing = LIF_inh_neuron_spacing
        self.taupre = taupre
        self.taupost = taupost
        self.wmax = wmax
        self.Apre = Apre
        self.Apost = Apost
        
    # internal function to create spike monitors
    def _build_spike_monitors(self):
        self.L0_mon = SpikeMonitor(self.L0)                                                                                                 
        self.L1_exc_mon = SpikeMonitor(self.L1_exc)                                                                                         
        self.L2_exc_mon = SpikeMonitor(self.L2_exc)                                                                                         
        self.L3_exc_mon = SpikeMonitor(self.L3_exc)                                                                                       
        self.L4_exc_mon = SpikeMonitor(self.L4_exc)                                                                                          

    # internal function to create synapses and connect layers
    def _connect_layers(self):
        
        # variables to enable creation of randomised connections between layers within topologically corresponding regions
        num_conn = 5000 # number of connections from layer to a neuron in next layer - WRONG, NEED TO FIGURE OUT HOW TO DEFINE THIS EXACTLY
        p_conn = 0.5 # probability of connection between neurons - required to randomise connections, essentially defines sparsity of connections in a region
        fan_in_radius = np.sqrt(num_conn/(np.pi*p_conn)) * umetre  
        poisson_layer_width = self.poisson_layer_width   
        poisson_neuron_spacing = self.poisson_neuron_spacing
        LIF_exc_layer_width = self.LIF_exc_layer_width
        LIF_inh_layer_width = self.LIF_inh_layer_width                                                                                                                                                                                                                    
        LIF_exc_layer_width = self.LIF_exc_layer_width                                                                                                          
        LIF_inh_layer_width = self.LIF_inh_layer_width                                                                                                          
        LIF_exc_neuron_spacing = self.LIF_exc_neuron_spacing
        LIF_inh_neuron_spacing = self.LIF_inh_neuron_spacing
        
        # parameters to enable Gaussian distributed axonal conduction delays
        mean_delay = 0.01                                                                                                                    
        SD_delay = 3                                                                                                                      
        
        # variables and parameters for STDP (trace learning rule)
        taupre = self.taupre
        taupost = self.taupost
        wmax = self.wmax
        Apre = self.Apre
        Apost = self.Apost
        
        # equations for STDP (trace learning rule)
        STDP_ODEs = '''
        # ODEs for trace learning rule
        w                             : 1
        dapre/dt = -apre/taupre       : 1 (event-driven)
        dapost/dt = -apost/taupost    : 1 (event-driven)
        '''
        # update rule for presynaptic spike
        STDP_presyn_update = '''
        v_post += w
        apre += Apre
        w = clip(w+apost, 0, wmax)
        '''
        # update rule for postsynaptic spike
        STDP_postsyn_update = '''
        apost += Apost
        w = clip(w+apre, 0, wmax)
        '''
        
        # =============================================================================
        # feedforward connections       
        # =============================================================================
        
        # Layer 0 to Layer 1 excitatory
        self.Syn_L0_L1_exc = Synapses(self.L0, self.L1_exc, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update) # create synapses with STDP learning rule
        self.Syn_L0_L1_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius',p=p_conn) # connect lower layer neurons to random upper layer neurons with spatial relation (implicitly selects from random filters)
        self.num_Syn_L0_L1_exc = len(self.Syn_L0_L1_exc.x_pre) # get number of synapses(can use x_pre or x_post to do this)
        self.Syn_L0_L1_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L0_L1_exc)*ms # set Gaussian-ditributed synaptic delay 

        # Layer 1 excitatory to Layer 2 excitatory
        self.Syn_L1_exc_L2_exc = Synapses(self.L1_exc, self.L2_exc, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)            
        self.Syn_L1_exc_L2_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius',p=p_conn)                               
        self.num_Syn_L1_exc_L2_exc = len(self.Syn_L1_exc_L2_exc.x_pre)                                                                               
        self.Syn_L1_exc_L2_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L1_exc_L2_exc)*ms                                       
        
        # Layer 2 excitatory to Layer 3 excitatory
        self.Syn_L2_exc_L3_exc = Synapses(self.L2_exc, self.L3_exc, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)             
        self.Syn_L2_exc_L3_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius', p=p_conn)                             
        self.num_Syn_L2_exc_L3_exc = len(self.Syn_L2_exc_L3_exc.x_pre)                                                                             
        self.Syn_L2_exc_L3_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L2_exc_L3_exc)*ms                                        
        
        # Layer 3 excitatory to Layer 4 excitatory
        self.Syn_L3_exc_L4_exc = Synapses(self.L3_exc, self.L4_exc, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)              
        self.Syn_L3_exc_L4_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius', p=p_conn)                               
        self.num_Syn_L3_exc_L4_exc = len(self.Syn_L3_exc_L4_exc.x_pre)                                                                              
        self.Syn_L3_exc_L4_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L3_exc_L4_exc)*ms                                       
    
        # =============================================================================
        # lateral connections        
        # =============================================================================
        
        # Layer 1 excitatory to Layer 1 inhibitory 
        self.Syn_L1_exc_L1_inh = Synapses(self.L1_exc, self.L1_inh, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)
        self.Syn_L1_exc_L1_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius',p=p_conn)                               
        self.num_Syn_L1_exc_L1_inh = len(self.Syn_L1_exc_L1_inh.x_pre)                                                                               
        self.Syn_L1_exc_L1_inh.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L1_exc_L1_inh)*ms 
        
        # Layer 1 inhibitory to Layer 1 excitatory 
        self.Syn_L1_inh_L1_exc = Synapses(self.L1_inh, self.L1_exc, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)
        self.Syn_L1_inh_L1_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius',p=p_conn)                               
        self.num_Syn_L1_inh_L1_exc = len(self.Syn_L1_inh_L1_exc.x_pre)                                                                               
        self.Syn_L1_inh_L1_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L1_inh_L1_exc)*ms 
        
        # Layer 2 excitatory to Layer 2 inhibitory 
        self.Syn_L2_exc_L2_inh = Synapses(self.L2_exc, self.L2_inh, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)
        self.Syn_L2_exc_L2_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius',p=p_conn)                               
        self.num_Syn_L2_exc_L2_inh = len(self.Syn_L2_exc_L2_inh.x_pre)                                                                               
        self.Syn_L2_exc_L2_inh.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L2_exc_L2_inh)*ms 
        
        # Layer 2 inhibitory to Layer 2 excitatory 
        self.Syn_L2_inh_L2_exc = Synapses(self.L2_inh, self.L2_exc, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)
        self.Syn_L2_inh_L2_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius',p=p_conn)                               
        self.num_Syn_L2_inh_L2_exc = len(self.Syn_L2_inh_L2_exc.x_pre)                                                                               
        self.Syn_L2_inh_L2_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L2_inh_L2_exc)*ms 
        
        # Layer 3 excitatory to Layer 3 inhibitory 
        self.Syn_L3_exc_L3_inh = Synapses(self.L3_exc, self.L3_inh, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)
        self.Syn_L3_exc_L3_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius',p=p_conn)                               
        self.num_Syn_L3_exc_L3_inh = len(self.Syn_L3_exc_L3_inh.x_pre)                                                                               
        self.Syn_L3_exc_L3_inh.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L3_exc_L3_inh)*ms 
        
        # Layer 3 inhibitory to Layer 3 excitatory 
        self.Syn_L3_inh_L3_exc = Synapses(self.L3_inh, self.L3_exc, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)
        self.Syn_L3_inh_L3_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius',p=p_conn)                               
        self.num_Syn_L3_inh_L3_exc = len(self.Syn_L3_inh_L3_exc.x_pre)                                                                               
        self.Syn_L3_inh_L3_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L3_inh_L3_exc)*ms

        # Layer 4 excitatory to Layer 4 inhibitory 
        self.Syn_L4_exc_L4_inh = Synapses(self.L4_exc, self.L4_inh, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)
        self.Syn_L4_exc_L4_inh.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius',p=p_conn)                               
        self.num_Syn_L4_exc_L4_inh = len(self.Syn_L4_exc_L4_inh.x_pre)                                                                               
        self.Syn_L4_exc_L4_inh.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L4_exc_L4_inh)*ms 
        
        # Layer 4 inhibitory to Layer 4 excitatory 
        self.Syn_L4_inh_L4_exc = Synapses(self.L4_inh, self.L4_exc, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)
        self.Syn_L4_inh_L4_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius',p=p_conn)                               
        self.num_Syn_L4_inh_L4_exc = len(self.Syn_L4_inh_L4_exc.x_pre)                                                                               
        self.Syn_L4_inh_L4_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L4_inh_L4_exc)*ms

        # =============================================================================
        # feedback connections    
        # =============================================================================
    
        # Layer 4 excitatory to Layer 3 excitatory
        self.Syn_L4_exc_L3_exc = Synapses(self.L4_exc, self.L3_exc, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)              
        self.Syn_L4_exc_L3_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius', p=p_conn)                               
        self.num_Syn_L4_exc_L3_exc = len(self.Syn_L4_exc_L3_exc.x_pre)                                                                              
        self.Syn_L4_exc_L3_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L4_exc_L3_exc)*ms   
        
        # Layer 3 excitatory to Layer 2 excitatory
        self.Syn_L3_exc_L2_exc = Synapses(self.L3_exc, self.L2_exc, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)              
        self.Syn_L3_exc_L2_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius', p=p_conn)                               
        self.num_Syn_L3_exc_L2_exc = len(self.Syn_L3_exc_L2_exc.x_pre)                                                                              
        self.Syn_L3_exc_L2_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L3_exc_L2_exc)*ms   
        
        # Layer 2 excitatory to Layer 1 excitatory
        self.Syn_L2_exc_L1_exc = Synapses(self.L2_exc, self.L1_exc, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)              
        self.Syn_L2_exc_L1_exc.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius', p=p_conn)                               
        self.num_Syn_L2_exc_L1_exc = len(self.Syn_L2_exc_L1_exc.x_pre)                                                                              
        self.Syn_L2_exc_L1_exc.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L2_exc_L1_exc)*ms   
        
    # internal function to generate Gabor filters to be applied to input image (called inside _gabor_filter)
    def _generate_gabor_filters(self):
        self.filters = []                                                                                                            
        ksize = 5 # kernel size
        phi_list = [0, np.pi/2, np.pi] # phase offset of sinusoid 
        lamda = 2 # wavelength of sinusoid 
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
        self.L0.rate = flattened_filtered_image * 10e-8 * Hz # set firing rates of L0 Poisson neurons equal to outputs of Gabor filters - multiply by a coefficient (10e-8) to get biologically realistic values
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
        self.network.run(length, namespace={'v_th': self.v_th, # run simulations, passing dictionary of necessary parameters into namespace argument (simulation will raise error otherwise)
                                            'v_0': self.v_0,
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
                                            'wmax': self.wmax,
                                            'Apre': self.Apre,
                                            'Apost': self.Apost},
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

# function to isolate a set of neurons' spikes (after simulation run to produce raster plots) 
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