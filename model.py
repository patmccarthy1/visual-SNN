# A Spiking Neural Network Model of the Primate Ventral Visual Pathway using the Brian 2 simulator
# Patrick McCarthy
# Imperial College London
# Copyright © 2020

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

class SpikingVisNet:
    
    # class constructor
    def __init__(self):
    
        self.filters = self._generate_gabor_filters()
        self._build_layers()
        self._build_spike_monitors()
        self._connect_layers()
        
    # internal function to create and connect model layers upon class instantiation (called inside __init__)
    def _build_layers(self):
    
        v_th = 0.8                                                                                                                  # threshold potential
        v_0 = 0                                                                                                                     # starting potential
        tau_m = 1*ms 
        dt = 0.02
        
        # variables for neurons in L0
        self.poisson_layer_width = 256     
        self.N_poisson = self.poisson_layer_width**2                                                                                          # can change to np.sqrt(len(flattened_filtered_image)/len(self.filters)) to generalise to different image sizes
        self.poisson_neuron_spacing = 12.5*umetre
        
        # equations for neurons in L0
        poisson_neurons = '''
        dv/dt = -(v-v_0)/tau_m                                   : volt   # membrane potential
        x = (i%poisson_layer_width)*poisson_neuron_spacing       : metre  # x position
        y = (int(i/poisson_layer_width))*poisson_neuron_spacing  : metre  # y position
        f = int(i/N_poisson)                                     : 1      # filter number
        rates                                                    : 1      # firing rate to define Poisson distribution
        '''
        
        # variables for neurons in Layers 1-4
        self.LIF_layer_width = 64                                                                                                        # width of Layers 1-4 in neurons, e.g. if 128 we will have 128^2 = 16384 neurons in a layer
        self.N_LIF = self.LIF_layer_width**2                                                                                                  # number of neurons in a layer
        self.LIF_neuron_spacing = self.poisson_neuron_spacing*(self.poisson_layer_width/self.LIF_layer_width)                                                                                           # required to assign spatial locations of neurons

        # variables and parameters for STDP (trace learning rule)
        taupre = taupost = 20*ms
        wmax = 0.01
        Apre = 0.01
        Apost = -Apre*taupre/taupost*1.05
        
        # equations for neurons in Layers 1-4
        LIF_neurons = '''
        dv/dt = -(v-v_0)/tau_m                           : volt   # membrane potential
        x = (i%LIF_layer_width)*LIF_neuron_spacing       : metre  # x position
        y = (int(i/LIF_layer_width))*LIF_neuron_spacing  : metre  # y position
        '''

        # Layer 0  
        self.L0 = NeuronGroup(len(self.filters)*self.N_poisson, poisson_neurons, threshold='np.random.random()*dt < rate')                # create group of Poisson neurons with STDP learning rule
        
        # Layer 1 
        self.L1 = NeuronGroup(self.N_LIF, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                             # create group of LIF neurons with STDP learning rule

        # Layer 2 
        self.L2 = NeuronGroup(self.N_LIF, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                             # create group of LIF neurons with STDP learning rule
        
        # Layer 3 
        self.L3 = NeuronGroup(self.N_LIF, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                             # create group of LIF neurons with STDP learning rule
        
        # Layer 4 
        self.L4 = NeuronGroup(self.N_LIF, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                             # create group of LIF neurons with STDP learning rule
       
    # internal function to create spike monitors
    def _build_spike_monitors(self):
        self.L0_mon = SpikeMonitor(self.L0)                                                                                          # create object to monitor Layer 0 spike times
        self.L1_mon = SpikeMonitor(self.L1)                                                                                          # create object to monitor Layer 1 spike times
        self.L2_mon = SpikeMonitor(self.L2)                                                                                          # create object to monitor Layer 2 spike times
        self.L3_mon = SpikeMonitor(self.L3)                                                                                          # create object to monitor Layer 3 spike times
        self.L4_mon = SpikeMonitor(self.L4)                                                                                          # create object to monitor Layer 4 spike times

    # internal function to create synapses and connect layers
    def _connect_layers(self):
        
        # variables to enable creation of randomised connections between layers within topologically corresponding regions
        num_conn =  100                                                                                                              # number of connections from layer to a single neuron in next layer 
        p_conn = 0.5                                                                                                                 # probability of connection between neurons - required to randomise connections, essentially defines sparsity of connections in a region
        fan_in_radius = np.sqrt(num_conn/(np.pi*p_conn))*umetre     
        poisson_layer_width = 256   
        poisson_neuron_spacing = 12.5*umetre
        LIF_layer_width = 64                                                                                                          
        LIF_neuron_spacing = 50*umetre
        
        # parameters to enable Gaussian distributed axonal conduction delays
        mean_delay = 5                                                                                                               # mean for Gaussian distribution to draw conduction delays from - units will be ms
        SD_delay = 5                                                                                                                 # SD for Gaussian distribution to draw conduction delays from - units will be ms
        
        # variables and parameters for STDP (trace learning rule)
        taupre = taupost = 20*ms
        wmax = 0.01
        Apre = 0.01
        Apost = -Apre*taupre/taupost*1.05
        
        # equations for STDP (trace learning rule)
        STDP_ODEs = '''
        # ODEs for trace learning rule
        w                             : 1
        dapre/dt = -apre/taupre       : 1 (event-driven)
        dapost/dt = -apost/taupost    : 1 (event-driven)
        '''
        STDP_presyn_update = '''
        # update rule for presynaptic spike
        v_post += w
        apre += Apre
        w = clip(w+apost, 0, wmax)
        '''
        STDP_postsyn_update = '''
        # update rule for postsynaptic spike
        apost += Apost
        w = clip(w+apre, 0, wmax)
        '''
        
        # Layer 0 (Poisson) neurons to Layer 1 (LIF) neurons
        self.Syn_L0_L1 = Synapses(self.L0, self.L1, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)               # create synapses with STDP learning rule
        self.Syn_L0_L1.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius',p=0.5)                                    # connect random L0 Poisson neurons to random L1 neurons with spatial relation (implicitly selects from random filters)
        self.num_Syn_L0_L1 = len(self.Syn_L0_L1.x_pre)                                                                               # get number of synapses(can use x_pre or x_post to do this)

        # Layer 1 neurons to Layer 2 neurons
        self.Syn_L1_L2 = Synapses(self.L1, self.L2, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)               # create synapses with STDP learning rule
        self.Syn_L1_L2.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius', p=p_conn)                                # connect 100 Layer 1 neurons to each Layer 2  neuron (by connecting with p_conn in neighbourhood of neighbourhood_width^2 neurons)
        self.num_Syn_L1_L2 = len(self.Syn_L1_L2.x_pre)                                                                               # get number of synapses(can use x_pre or x_post to do this)
        # print('number of connections: ', self.num_Syn_L1_L2)
        # print('x locations: {}\n'.format(self.L2.x[:]))
        # print('y locations: {}\n'.format(self.L2.y[:]))
        self.Syn_L1_L2.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L1_L2)*ms                                         # set Gaussian-ditributed synaptic delay 
        
        # Layer 2 neurons to Layer 3 neurons
        self.Syn_L2_L3 = Synapses(self.L2, self.L3, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)               # create synapses with STDP learning rule
        self.Syn_L2_L3.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius', p=p_conn)                                # connect 100 Layer 2 neurons to each Layer 3 neuron (by connecting with p_conn in neighbourhood of neighbourhood_width^2 neurons)
        self.num_Syn_L2_L3 = len(self.Syn_L2_L3.x_pre)                                                                               # get number of synapses(can use x_pre or x_post to do this)
        self.Syn_L2_L3.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L2_L3)*ms                                         # set Gaussian-ditributed synaptic delay
        
        # Layer 3 neurons to Layer 4 neurons
        self.Syn_L3_L4 = Synapses(self.L3, self.L4, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)               # create group of LIF neurons
        self.Syn_L3_L4.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius', p=p_conn)                                # connect 100 Layer 3 neurons to each Layer 4 neuron (by connecting with p_conn in neighbourhood of neighbourhood_width^2 neurons)
        self.num_Syn_L3_L4 = len(self.Syn_L3_L4.x_pre)                                                                               # get number of synapses(can use x_pre or x_post to do this)
        self.Syn_L3_L4.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L3_L4)*ms                                         # set Gaussian-ditributed synaptic delay
    
    # internal function to generate Gabor filters to be applied to input image (called inside _gabor_filter)
    def _generate_gabor_filters(self):
        self.filters = []                                                                                                            # list to hold filters
        ksize = 5                                                                                                                    # kernel size
        phi_list = [0, np.pi/2, np.pi]                                                                                               # phase offset of sinusoid 
        lamda = 2                                                                                                                    # wavelength of sinusoid 
        theta_list = [0,np.pi/4,np.pi/2,3*np.pi/4]                                                                                   # filter orientation
        b = 1.5                                                                                                                      # spatial bandwidth in octaves (will be used to determine SD)
        sigma = lamda*(2**b+1)/np.pi*(2**b-1) * np.sqrt(np.log(2)/2)
        gamma = 0.5                                                                                                                  # filter aspect ratio
        for phi in phi_list:
            for theta in theta_list:
                filt = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
                self.filters.append(filt)
        return self.filters
    
    # internal function to apply Gabor filters to SINGLE IMAGE and generate output image for each filter 
    def _gabor_filter(self, image, filters):
        filtered_image = np.empty([len(image),len(image),len(filters)],dtype=np.float32)                                             # NumPy array to store filtered images (first dimension is input image, second dimension is filters)                                                                                                                                     # iterate through images and filters
        for filt_idx, filt in enumerate(filters):
            filtered = cv2.filter2D(image, cv2.CV_8UC3, filt)                                                                        # apply filter
            # show image
            fig, ax = plt.subplots(1,1)
            ax.imshow(filtered)
            ax.set_title('Filter {}'.format(filt_idx+1))                                                                             # plot filtered images                               
            plt.axis('off')
            plt.show()
            filtered_image[:,:,filt_idx] = filtered                                                                                  # add filtered image to array
        flattened_filtered_image = np.ndarray.flatten(filtered_image)                                                                # flatten filtered images
        return flattened_filtered_image
    
    # internal function to generate Poisson spike trains from Gabor-filtered images 
    def _gabor_filtered_to_spikes(self,flattened_filtered_image):
        self.L0.rates = flattened_filtered_image                                                                                     # set firing rates of L0 Poisson neurons equal to outputs of Gabor filters

    # function to pass images into model - EVENTUALLY REPLACE WITH TRAIN AND TEST FUNCTIONS WHERE STDP IS ON AND OFF, RESPECITVELY
    def run_simulation(self, image):
        filtered_im = self._gabor_filter(image,self.filters)                                                                         # filter image
        self._gabor_filtered_to_spikes(filtered_im)                                                                                  # convert filtered image to Poisson spikes and connect random Poisson neurons
        run(10*second)
        return filtered_im
    
    # function to print out summary of model architecture as a sanity check
    def model_summary(self):
        print('MODEL SUMMARY\n\n')
        print('Layers\n\n')
        print(' layer | neurons | dimensions  | spacing (um) | filters\n')
        print('------------------------------------------------------------\n')
        print(' 0     | {}   | {}x{}x{} | {:.2f}        | {} \n'.format(self.L0.N,self.poisson_layer_width,self.poisson_layer_width,len(self.filters),self.poisson_neuron_spacing*10**6,len(self.filters)))
        print(' 1     | {}     | {}x{}      | {:.2f}        | n/a\n'.format(self.L1.N,self.LIF_layer_width,self.LIF_layer_width,self.LIF_neuron_spacing*10**6))
        print(' 2     | {}     | {}x{}      | {:.2f}        | n/a\n'.format(self.L2.N,self.LIF_layer_width,self.LIF_layer_width,self.LIF_neuron_spacing*10**6))
        print(' 3     | {}     | {}x{}      | {:.2f}        | n/a\n'.format(self.L3.N,self.LIF_layer_width,self.LIF_layer_width,self.LIF_neuron_spacing*10**6))
        print(' 4     | {}     | {}x{}      | {:.2f}        | n/a\n\n'.format(self.L4.N,self.LIF_layer_width,self.LIF_layer_width,self.LIF_neuron_spacing*10**6))
        print('Connections\n\n')
        print(' source | target | connections\n')
        print('-------------------------------\n')
        print(' 0      | 1      | {}\n'.format(self.num_Syn_L0_L1))
        print(' 1      | 2      | {}\n'.format(self.num_Syn_L1_L2))
        print(' 2      | 3      | {}\n'.format(self.num_Syn_L2_L3))
        print(' 3      | 4      | {}\n'.format(self.num_Syn_L3_L4))

# function to read images from file and store as arrays which can be passed into model
def read_images(img_dir):
    images = [cv2.imread(file, 0) for file in glob.glob(img_dir+"/*.png")]
    for image_idx, image in enumerate(images):
        fig, ax = plt.subplots(1,1)
        ax.imshow(image, cmap='gray')
        ax.set_title('Stimulus {}'.format(image_idx+1))
        plt.axis('off')
        plt.show()
    return images


if __name__ == '__main__':

    # read in image to array
    ims = read_images('data')
    im = ims[0]
    
    start_scope()
    
    visnet = SpikingVisNet()
    visnet.model_summary()
    
    for im in ims:
        filtered_image = visnet.run_simulation(im)
    # mon = visnet.L2_mon


    