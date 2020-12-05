from model import *

if __name__ == '__main__':

    # read in image to array
    ims = read_images('mini_test_images')
    im = ims[0]
    
    start_scope()
    
    visnet = SpikingVisNet()
    visnet.model_summary()
    visnet.run_simulation(im,0.5*second)
    
    L0_set_i, L0_set_t = get_neurons(visnet.L0_mon,0,300)
    L1_set_i, L1_set_t = get_neurons(visnet.L1_mon,0,300)
    L2_set_i, L2_set_t = get_neurons(visnet.L2_mon,0,300)
    L3_set_i, L3_set_t = get_neurons(visnet.L3_mon,0,300)
    L4_set_i, L4_set_t = get_neurons(visnet.L4_mon,0,300)
    
    figure()
    plot(L0_set_t/ms, L0_set_i, '.k',markersize=2)
    xlabel('Time (ms)')
    ylabel('Neuron index')
    title('Layer 0')
    show()
    savefig('L0_b.png',dpi=500)
    
    figure()
    plot(L1_set_t/ms, L1_set_i, '.k',markersize=2)
    xlabel('Time (ms)')
    ylabel('Neuron index')
    title('Layer 1')
    show()
    savefig('L1_b.png',dpi=500)
    
    figure()
    plot(L2_set_t/ms, L2_set_i, '.k',markersize=2)
    xlabel('Time (ms)')
    ylabel('Neuron index')
    title('Layer 2')
    show()
    savefig('L2_b.png',dpi=100)
    
    figure()
    plot(L3_set_t/ms, L3_set_i, '.k',markersize=2)
    xlabel('Time (ms)')
    ylabel('Neuron index')
    title('Layer 3')
    show()
    savefig('L3_b.png',dpi=100)
    
    figure()
    plot(L4_set_t/ms, L4_set_i, '.k',markersize=2)
    xlabel('Time (ms)')
    ylabel('Neuron index')
    title('Layer 4')
    show()
    savefig('L4_b.png',dpi=100)