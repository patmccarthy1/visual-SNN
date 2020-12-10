from model import *

if __name__ == '__main__':

    # read in image to array
    # ims = read_images('mini_test_images')
    # im = ims[1]
    
    start_scope()
    
    visnet = SpikingVisNet()
    visnet.model_summary()
    
     #%%
    visualise_connectivity(visnet.Syn_L0_L1_exc)
    suptitle('Layer 0 - Layer 1 exc. connectivity')
    savefig('L0_L1_connectivity.png',dpi=500)

    visualise_connectivity(visnet.Syn_L1_exc_L2_exc)
    suptitle('Layer 1 exc. - Layer 2 exc. connectivity')
    savefig('L1_L2_connectivity.png',dpi=500)
    
    visualise_connectivity(visnet.Syn_L2_exc_L3_exc)
    suptitle('Layer 2 exc. - Layer 3 exc. connectivity')
    savefig('L2_L3_connectivity.png',dpi=500)
    
    visualise_connectivity(visnet.Syn_L3_exc_L4_exc)
    suptitle('Layer 3 exc. - Layer 4 exc. connectivity')
    savefig('L4_L5_connectivity.png',dpi=500)

    #%%
    ims = read_images('mini_test_images')
    im = ims[0]
    visnet.run_simulation(im,0.1*second)
    
    L0_set_i, L0_set_t = get_neurons(visnet.L0_mon,0,300)
    L1_set_i, L1_set_t = get_neurons(visnet.L1_exc_mon,0,300)
    L2_set_i, L2_set_t = get_neurons(visnet.L2_exc_mon,0,300)
    L3_set_i, L3_set_t = get_neurons(visnet.L3_exc_mon,0,300)
    L4_set_i, L4_set_t = get_neurons(visnet.L4_exc_mon,0,300)
    
    figure()
    plot(L0_set_t/ms, L0_set_i, '.k',markersize=0.25)
    xlabel('Time (ms)')
    ylabel('Neuron index')
    title('Layer 0')
    savefig('L0_raster.png',dpi=500)
    
    figure()
    plot(L1_set_t/ms, L1_set_i, '.k',markersize=0.25)
    xlabel('Time (ms)')
    ylabel('Neuron index')
    title('Layer 1')
    savefig('L1_raster.png',dpi=500)
    
    figure()
    plot(L2_set_t/ms, L2_set_i, '.k',markersize=0.25)
    xlabel('Time (ms)')
    ylabel('Neuron index')
    title('Layer 2')
    savefig('L2_raster.png',dpi=500)
    
    figure()
    plot(L3_set_t/ms, L3_set_i, '.k',markersize=0.25)
    xlabel('Time (ms)')
    ylabel('Neuron index')
    title('Layer 3')
    savefig('L3_raster.png',dpi=500)
    
    figure()
    plot(L4_set_t/ms, L4_set_i, '.k',markersize=0.25)
    xlabel('Time (ms)')
    ylabel('Neuron index')
    title('Layer 4')
    savefig('L4_raster.png',dpi=500)
    show()