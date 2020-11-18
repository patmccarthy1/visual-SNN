# Patrick McCarthy
# A Spiking Neural Network Model of the Primate Ventral Visual Pathway using the Brian 2 simulator
# Imperial College London
# Copyright © 2020

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

# =============================================================================
# Model class
# =============================================================================

class SpikingVisNet:
    
    # class initialisation function
    def __init__(self):
    
        self._build_layers()
        self._build_spike_monitors()
        
    # internal function to create and connect model layers upon class instantiation (called inside __init__)
    def _build_layers(self):
    
        # EVENTUALLY HAVE BOTH EXCITATORY AND INHIBITORY NEURONS AS WELL AS TOP-DOWN AND LATERAL CONNECTIONS
        
        # variables and parameters for neurons in Layers 1-4
        layer_width = 20                                                                                                          # width of Layers 1-4 in neurons, e.g. if 128 we will have 128^2 = 16384 neurons in a layer
        N = layer_width**2
        neuron_spacing = 50*umetre                                                                                                # required to assign spatial locations of neurons
        v_th = 0                                                                                                                  # threshold potential
        v_0 = 0                                                                                                                   # starting potential
        tau_m = 0                                                                                                                 # membrane time constant
        x_locs = []                                                                                                               # list which will be used to define x locations of neurons
        y_locs = []                                                                                                               # list which will be used to define y locations of neurons
        for x in range(layer_width):                                                                                              # assign locations such that [x,y] = [0,0],[0,1],[0,2]...[0,layer_width],[1,0],[1,1],[1,2]...[layer_width-1,layer_width],[layer_width,layer_width], e.g. for layer width 128, we would have locations [x,y] = [0,0],[0,1],[0,2]...[0,127],[1,0],[1,1],[1,2]...[126,127],[127,127]
            for y in range(layer_width):
                x_locs.append(x)
                y_locs.append(y)
        x_locs = x_locs*neuron_spacing                                                                                            # multiply by neuron spacing to give actual spatial dimensions
        y_locs = y_locs*neuron_spacing 
        # variables to enable creation of randomised connections between layers within topologically corresponding regions
        num_conn =  np.ceil(layer_width/10)**2                                                                                      # number of connections from layer to a single neuron in next layer - 1/10 of layer width, rounding up to nearest integer and then squaring e.g for layer width 128, we would have 13^2 = 169 connections to each postsynaptic neuron
        p_conn = 0.5                                                                                                                # probability of connection between neurons - required to randomise connections, essentially defines sparsity of connections in a region
        neighbourhood_width  = np.sqrt(num_conn/p_conn)*umetre                                                                      # define width of square neighbourhood from which to randomly select neurons to connect in each layer - approx. 1/10 of layer width, rounding up to nearest integer
        # parameters to enable Gaussian distributed axonal conduction delays
        mean_delay = 0                                                                                                              # mean for Gaussian distribution to draw conduction delays from
        SD_delay = 1                                                                                                                # SD for Gaussian distribution to draw conduction delays from
        
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
        
        # Layer 0 (Gabor filter/ Poisson spikes input layer)
        self.L0 = PoissonGroup(N,  rates=np.random.randn(1,400)*Hz)
 
        
        # Layer 1 (V2)
        self.L1 = NeuronGroup(N, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                                 # create group of LIF neurons with STDP learning rule
        self.L1.x = x_locs                                                                                                           # define spatial locations of Layer 1 neurons
        self.L1.y = y_locs
        # self.Syn_L0_L1 = Synapses(self.L0, self.L1, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)
        # self.Syn_L0_L1.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < neighbourhood_width', p=p_conn)                        # connect 30 Layer 0 neurons to each Layer 1  neuron (by connecting with p_conn in neighbourhood of neighbourhood_width^2 neurons)
        # self.num_Syn_L0_L1 = len(self.Syn_L0_L1.x_pre)                                                                             # get number of synapses(can use x_pre or x_post to do this)
        # self.Syn_L0_L1.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L1_L2)*ms                                       # set Gaussian-ditributed synaptic delay 
        
        # Layer 2 (V4)
        self.L2 = NeuronGroup(N, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                                 # create group of LIF neurons with STDP learning rule
        self.L2.x = x_locs                                                                                                           # define spatial locations of Layer 2 neurons
        self.L2.y = y_locs
        self.Syn_L1_L2 = Synapses(self.L1, self.L2, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)               # create synapses with STDP learning rule
        self.Syn_L1_L2.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < neighbourhood_width', p=p_conn)                          # connect Layer 1 neurons to each Layer 2  neuron (by connecting with p_conn in neighbourhood of neighbourhood_width^2 neurons)
        self.num_Syn_L1_L2 = len(self.Syn_L1_L2.x_pre)                                                                               # get number of synapses(can use x_pre or x_post to do this)
        self.Syn_L1_L2.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L1_L2)*ms                                         # set Gaussian-ditributed synaptic delay 
        
        # Layer 3 (TEO)
        self.L3 = NeuronGroup(N, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                                 # create group of LIF neurons with STDP learning rule
        self.L3.x = x_locs                                                                                                           # define spatial locations of Layer 3 neurons
        self.L3.y = y_locs
        self.Syn_L2_L3 = Synapses(self.L2, self.L3, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)               # create synapses with STDP learning rule
        self.Syn_L2_L3.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < neighbourhood_width', p=p_conn)                          # connect Layer 2 neurons to each Layer 3 neuron (by connecting with p_conn in neighbourhood of neighbourhood_width^2 neurons)
        self.num_Syn_L2_L3 = len(self.Syn_L2_L3.x_pre)                                                                               # get number of synapses(can use x_pre or x_post to do this)
        self.Syn_L2_L3.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L2_L3)*ms                                         # set Gaussian-ditributed synaptic delay
        
        # Layer 4 (TE)
        self.L4 = NeuronGroup(N, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                                 # create group of LIF neurons with STDP learning rule
        self.L4.x = x_locs                                                                                                           # define spatial locations of Layer 4 neurons
        self.L4.y = y_locs
        self.Syn_L3_L4 = Synapses(self.L3, self.L4, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)               # create group of LIF neurons
        self.Syn_L3_L4.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < neighbourhood_width', p=p_conn)                          # connect Layer 3 neurons to each Layer 4 neuron (by connecting with p_conn in neighbourhood of neighbourhood_width^2 neurons)
        self.num_Syn_L3_L4 = len(self.Syn_L3_L4.x_pre)                                                                               # get number of synapses(can use x_pre or x_post to do this)
        self.Syn_L3_L4.delay = np.random.normal(mean_delay, SD_delay, self.num_Syn_L3_L4)*ms                                         # set Gaussian-ditributed synaptic delay
        
    # internal function to create spike monitors
    def _build_spike_monitors(self):
        self.L1_mon = SpikeMonitor(self.L1)                                                                                          # create object to monitor Layer 1 spike times
        self.L2_mon = SpikeMonitor(self.L2)                                                                                          # create object to monitor Layer 2 spike times
        self.L3_mon = SpikeMonitor(self.L3)                                                                                          # create object to monitor Layer 3 spike times
        self.L4_mon = SpikeMonitor(self.L4)                                                                                          # create object to monitor Layer 4 spike times
    
    # internal function to generate Poisson spike trains from images
    def _image_to_spikes(self):
        return 0
    
    # internal function to generate Gabor filters to be applied to input image (called inside _gabor_filter)
    def _generate_gabor_filters(self):
        self.filters = [] # list to hold filters
        ksize = 5 # kernel size
        phi_list = [0, np.pi] # phase offset of sinusoid 
        lamda = 2 # wavelength of sinusoid 
        theta_list = [0,np.pi/4,np.pi/2,3*np.pi/4] # filter orientation
        b = 1.5 # spatial bandwidth in octaves (will be used to determine SD)
        sigma = lamda*(2**b+1)/np.pi*(2**b-1) * np.sqrt(np.log(2)/2)
        gamma = 0.5 # filter aspect ratio
        for phi in phi_list:
            for theta in theta_list:
                filt = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
                self.filters.append(filt)
        return self.filters
    
    # internal function to apply Gabor filters to image and generate output image for each filter (called inside _gabor_filtered_to_spikes)
    def _gabor_filter(self, images, filters):
        filtered_images = np.empty((len(images),len(filters)), dtype=object)                                                        # array to store filtered images (first dimension is input image, second dimension is filters)                                                   
        for image_idx, image in enumerate(images):                                                                                  # iterate through images and filters
            for filt_idx, filt in enumerate(filters):
                filtered_image = cv2.filter2D(image, cv2.CV_8UC3, filt)                                                             # apply filter
                fig, ax = plt.subplots(1,1)
                ax.imshow(filtered_image)
                ax.set_title('Image {}, Filter {}'.format(image_idx+1,filt_idx+1))                                                  # plot filtered images                               
                plt.show()
                filtered_images[image_idx,filt_idx] = filtered_images                                                               # add filtered image to array
        return filtered_images 
    
    # internal function to generate Poisson spike trains from Gabor-filtered images (called inside _image_to_spikes)
    def _gabor_filtered_to_spikes(self):
        return 0

    # function to pass images into model - EVENTUALLY REPLACE WITH TRAIN AND TEST FUNCTIONS WHERE STDP IS ON AND OFF, RESPECITVELY
    def run_simulation(self, images):
        filters = self._generate_gabor_filters()
        filtered_ims = self._gabor_filter(images,filters)
        return filtered_ims
    
    # function to print out summary of model architecture as a sanity check
    def model_summary(self):
        return 0
    
# function to read images from file and store as arrays which can be passed into model
def read_images(img_dir):
    images = [cv2.imread(file, 0) for file in glob.glob(img_dir+"/*.png")]
    return images

# =============================================================================
# Main function
# =============================================================================

if __name__ == '__main__':

    # read in image to array
    ims = read_images('data')
    for im_idx, im in enumerate(ims):
        fig, ax = plt.subplots(1,1)
        ax.imshow(im, cmap='gray')
        ax.set_title('Image {}'.format(im_idx+1))
        plt.show()
        
    start_scope()
    
    visnet = SpikingVisNet()
    filtered_images = visnet.run_simulation(ims)
    