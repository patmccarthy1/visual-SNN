L1_exc_spikes = csvread('../output_data/layer_1_excitatory_spikes.csv');
L1_exc_L2_exc_weights =  csvread('../output_data/layer_1_exc_layer_2_exc_weights.csv');
L1_exc_L2_exc_weights = L1_exc_L2_exc_weights(6,:);

figure()
scatter(L1_exc_spikes(2,:),L1_exc_spikes(1,:),0.5,'r')
title('layer 1 excitatory neurons')
xlabel('time (ms)')
ylabel('neuron index')

figure()
hist(L1_exc_L2_exc_weights)
title('weights of synapses from layer 1 excitatory to layer 2 excitatory neurons')
xlabel('synaptic weight (Siemens)')
xlabel('number of neurons')
grid('on')