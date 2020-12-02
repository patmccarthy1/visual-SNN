from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

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

# internal function to generate Gabor filters to be applied to input image (called inside _gabor_filter)
def generate_gabor_filters():
    filters = []                                                                                                            # list to hold filters
    ksize = 4                                                                                                                    # kernel size
    phi_list = [0, np.pi]                                                                                               # phase offset of sinusoid 
    lamda = 2                                                                                                                    # wavelength of sinusoid 
    theta_list = [0,np.pi/2]                                                                                   # filter orientation
    b = 1.5                                                                                                                      # spatial bandwidth in octaves (will be used to determine SD)
    sigma = lamda*(2**b+1)/np.pi*(2**b-1) * np.sqrt(np.log(2)/2)
    gamma = 0.5                                                                                                                  # filter aspect ratio
    for phi in phi_list:
        for theta in theta_list:
            filt = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
            filters.append(filt)
    return filters
    
images = read_images('mini_test_images')
image = images[1]

start_scope()

filters = generate_gabor_filters()

v_th = -0.1                                                                                                                  # threshold potential
v_0 = -0.7                                                                                                                      # starting potential
tau_m_poisson = 50 * ms 
tau_m_LIF = 0.00001 * ms

# variables for neurons in L0
poisson_layer_width = 16     
N_poisson = poisson_layer_width**2                                                                                 # can change to np.sqrt(len(flattened_filtered_image)/len(self.filters)) to generalise to different image sizes
poisson_neuron_spacing = 12.5*umetre

# equations for neurons in L0
poisson_neurons = '''
dv/dt = -(v-v_0)/tau_m_poisson                           : 1   # membrane potential
x = (i%poisson_layer_width)*poisson_neuron_spacing       : metre  # x position
y = (int(i/poisson_layer_width))*poisson_neuron_spacing  : metre  # y position
f = int(i/N_poisson)                                     : 1      # filter number
rate                                                     : Hz      # firing rate to define Poisson distribution
'''

# variables for neurons in Layers 1-4
LIF_layer_width = 16                                                                                                   # width of Layers 1-4 in neurons, e.g. if 128 we will have 128^2 = 16384 neurons in a layer
N_LIF = LIF_layer_width**2                                                                                         # number of neurons in a layer
LIF_neuron_spacing = poisson_neuron_spacing*(poisson_layer_width/LIF_layer_width)                                                                                           # required to assign spatial locations of neurons

# equations for neurons in Layers 1-4
LIF_neurons = '''
dv/dt = -(v-v_0)/tau_m_LIF                       : 1   # membrane potential
x = (i%LIF_layer_width)*LIF_neuron_spacing       : metre  # x position
y = (int(i/LIF_layer_width))*LIF_neuron_spacing  : metre  # y position
'''

# Layer 0  
L0 = NeuronGroup(len(filters)*N_poisson, poisson_neurons, threshold='rand()*dt/second < rate*second', reset='v = v_0', method='euler')           # create group of Poisson neurons with STDP learning rule
# for i in range(L0.N):
#     print('L0 neuron {}: [{:.5f},{:.5f},{}]'.format(L0.i[i],L0.x[i],L0.y[i],L0.f[i]))

# Layer 1 
L1 = NeuronGroup(N_LIF, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                        # create group of LIF neurons with STDP learning rule

# Layer 2 
L2 = NeuronGroup(N_LIF, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                        # create group of LIF neurons with STDP learning rule

# Layer 3 
L3 = NeuronGroup(N_LIF, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                        # create group of LIF neurons with STDP learning rule

# Layer 4 
L4 = NeuronGroup(N_LIF, LIF_neurons, threshold='v > v_th', reset='v = v_0', method='euler')                        # create group of LIF neurons with STDP learning rule


L0_mon = SpikeMonitor(L0)                                                                                          # create object to monitor Layer 0 spike times
L1_mon = SpikeMonitor(L1)                                                                                          # create object to monitor Layer 1 spike times
L2_mon = SpikeMonitor(L2)                                                                                          # create object to monitor Layer 2 spike times
L3_mon = SpikeMonitor(L3)                                                                                          # create object to monitor Layer 3 spike times
L4_mon = SpikeMonitor(L4)     

# variables to enable creation of randomised connections between layers within topologically corresponding regions
num_conn =  20                                                                                                              # number of connections from layer to a single neuron in next layer 
p_conn = 0.75                                                                                                                 # probability of connection between neurons - required to randomise connections, essentially defines sparsity of connections in a region
fan_in_radius = np.sqrt(num_conn/(np.pi*p_conn))*umetre     
        
# parameters to enable Gaussian distributed axonal conduction delays
mean_delay = 0.1                                                                                                               # mean for Gaussian distribution to draw conduction delays from - units will be ms
SD_delay = 3                                                                                                                 # SD for Gaussian distribution to draw conduction delays from - units will be ms

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

# Layer 0 (Poisson) neurons to Layer 1 (LIF) neurons
Syn_L0_L1 = Synapses(L0, L1, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)               # create synapses with STDP learning rule
Syn_L0_L1.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius',p=p_conn)                                    # connect random L0 Poisson neurons to random L1 neurons with spatial relation (implicitly selects from random filters)
num_Syn_L0_L1 = len(Syn_L0_L1.x_pre)                                                                               # get number of synapses(can use x_pre or x_post to do this)

# Layer 1 neurons to Layer 2 neurons
Syn_L1_L2 = Synapses(L1, L2, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)               # create synapses with STDP learning rule
Syn_L1_L2.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius', p=p_conn)                                # connect 100 Layer 1 neurons to each Layer 2  neuron (by connecting with p_conn in neighbourhood of neighbourhood_width^2 neurons)
num_Syn_L1_L2 = len(Syn_L1_L2.x_pre)                                                                               # get number of synapses(can use x_pre or x_post to do this)
# print('number of connections: ', self.num_Syn_L1_L2)
# print('x locations: {}\n'.format(self.L2.x[:]))
# print('y locations: {}\n'.format(self.L2.y[:]))
Syn_L1_L2.delay = np.random.normal(mean_delay, SD_delay, num_Syn_L1_L2)*ms                                         # set Gaussian-ditributed synaptic delay 

# Layer 2 neurons to Layer 3 neurons
Syn_L2_L3 = Synapses(L2, L3, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)               # create synapses with STDP learning rule
Syn_L2_L3.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius', p=p_conn)                                # connect 100 Layer 2 neurons to each Layer 3 neuron (by connecting with p_conn in neighbourhood of neighbourhood_width^2 neurons)
num_Syn_L2_L3 = len(Syn_L2_L3.x_pre)                                                                               # get number of synapses(can use x_pre or x_post to do this)
Syn_L2_L3.delay = np.random.normal(mean_delay, SD_delay, num_Syn_L2_L3)*ms                                         # set Gaussian-ditributed synaptic delay

# Layer 3 neurons to Layer 4 neurons
Syn_L3_L4 = Synapses(L3, L4, STDP_ODEs, on_pre=STDP_presyn_update, on_post=STDP_postsyn_update)               # create group of LIF neurons
Syn_L3_L4.connect('sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < fan_in_radius', p=p_conn)                                # connect 100 Layer 3 neurons to each Layer 4 neuron (by connecting with p_conn in neighbourhood of neighbourhood_width^2 neurons)
num_Syn_L3_L4 = len(Syn_L3_L4.x_pre)                                                                               # get number of synapses(can use x_pre or x_post to do this)
Syn_L3_L4.delay = np.random.normal(mean_delay, SD_delay, num_Syn_L3_L4)*ms                 

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
L0.rate = flattened_filtered_image * 0.0000001 * Hz     

run(0.04*second)

#%% isolate a set of neurons

def get_neurons(mon,lower_i,upper_i):
    neuron_set_i = []
    neuron_set_t = []
    for idx, neuron in enumerate(mon.i):
        if lower_i <= neuron <= upper_i:
            neuron_set_i.append(neuron)
            neuron_set_t.append(mon.t[idx])
    return neuron_set_i, neuron_set_t

L0_set_i, L0_set_t = get_neurons(L0_mon,0,300)
L1_set_i, L1_set_t = get_neurons(L1_mon,0,300)
L2_set_i, L2_set_t = get_neurons(L2_mon,0,300)
L3_set_i, L3_set_t = get_neurons(L3_mon,0,300)
L4_set_i, L4_set_t = get_neurons(L4_mon,0,300)

#%% raster plots 
figure()
plot(L0_set_t/ms, L0_set_i, '.k',markersize=2)
xlabel('Time (ms)')
ylabel('Neuron index')
title('Layer 0')
savefig('L0_b.png',dpi=500)

figure()
plot(L1_set_t/ms, L1_set_i, '.k',markersize=2)
xlabel('Time (ms)')
ylabel('Neuron index')
title('Layer 1')
savefig('L1_b.png',dpi=500)

figure()
plot(L2_set_t/ms, L2_set_i, '.k',markersize=2)
xlabel('Time (ms)')
ylabel('Neuron index')
title('Layer 2')
savefig('L2_b.png',dpi=100)

figure()
plot(L3_set_t/ms, L3_set_i, '.k',markersize=2)
xlabel('Time (ms)')
ylabel('Neuron index')
title('Layer 3')
savefig('L3_b.png',dpi=100)

figure()
plot(L4_set_t/ms, L4_set_i, '.k',markersize=2)
xlabel('Time (ms)')
ylabel('Neuron index')
title('Layer 4')
savefig('L4_b.png',dpi=100)

